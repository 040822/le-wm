"""Trace exact Stage-B CEM replans without changing solver random draws."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from stable_worldmodel.solver.callbacks import Callback


SLOT_ID_KEY = "_diagnostic_slot_id"


@dataclass(frozen=True)
class DiagnosticRun:
    label: str
    task: str
    family: str
    model_kind: str
    epoch: int
    run_config: Path
    checkpoint: Path
    reference_result: Path


@dataclass(frozen=True)
class DiagnosticManifest:
    path: Path
    root: Path
    protocol: dict[str, Any]
    runs: dict[str, DiagnosticRun]


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def load_diagnostic_manifest(path: str | Path) -> DiagnosticManifest:
    """Load and validate the fixed run matrix and reference evaluation protocol."""
    path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if int(raw.get("version", 0)) != 1:
        raise ValueError("diagnostic manifest version must be 1")
    root = (path.parent / str(raw["repository_root"])).resolve()
    protocol = dict(raw["protocol"])
    runs: dict[str, DiagnosticRun] = {}
    for label, value in raw["runs"].items():
        run = DiagnosticRun(
            label=str(label),
            task=str(value["task"]),
            family=str(value["family"]),
            model_kind=str(value["model_kind"]),
            epoch=int(value["epoch"]),
            run_config=(root / str(value["run_config"])).resolve(),
            checkpoint=(root / str(value["checkpoint"])).resolve(),
            reference_result=(root / str(value["reference_result"])).resolve(),
        )
        for artifact in (
            run.run_config,
            run.checkpoint,
            run.reference_result,
        ):
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
        reference = json.loads(run.reference_result.read_text(encoding="utf-8"))
        parameters = reference["parameters"]
        solver = parameters["solver"]
        plan = parameters["plan_config"]
        checks = {
            "seed": parameters["seed"],
            "num_eval": parameters["num_eval"],
            "goal_offset_steps": parameters["goal_offset_steps"],
            "eval_budget": parameters["eval_budget"],
            "horizon": plan["horizon"],
            "receding_horizon": plan["receding_horizon"],
            "action_block": plan["action_block"],
            "num_samples": solver["num_samples"],
            "n_steps": solver["n_steps"],
            "topk": solver["topk"],
        }
        mismatches = {
            key: (value, protocol[key])
            for key, value in checks.items()
            if int(value) != int(protocol[key])
        }
        if mismatches:
            raise ValueError(
                f"{label} reference protocol differs from manifest: {mismatches}"
            )
        if len(reference["episodes"]) != int(protocol["num_eval"]):
            raise ValueError(f"{label} reference does not contain 50 episodes")
        runs[str(label)] = run
    return DiagnosticManifest(path=path, root=root, protocol=protocol, runs=runs)


@dataclass
class CEMTraceCollector:
    """Own trace state behind one interface shared by solver/world-policy proxies."""

    detailed_steps: set[int] = field(
        default_factory=lambda: {0, 1, 2, 5, 10, 20, 29}
    )
    summaries: list[dict[str, Any]] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)
    plans: list[dict[str, Any]] = field(default_factory=list)
    executed_raw: dict[int, list[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    executed_normalized: dict[int, list[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    replan_counts: dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def next_context(self, slots: Iterable[int]) -> tuple[list[int], list[int]]:
        slots = [int(slot) for slot in slots]
        replans = [self.replan_counts[slot] for slot in slots]
        for slot in slots:
            self.replan_counts[slot] += 1
        return slots, replans

    def executed_prefix(self, slot: int) -> np.ndarray:
        values = self.executed_raw.get(int(slot), [])
        if not values:
            return np.empty((0,), dtype=np.float32)
        return np.stack(values).astype(np.float32, copy=False)


class CEMTraceCallback(Callback):
    """Record CEM state only by detaching existing tensors; never sample RNG."""

    name = "stage_b_episode_trace"

    def __init__(self, collector: CEMTraceCollector, batch_size: int):
        super().__init__(reduction="none")
        self.collector = collector
        self.batch_size = int(batch_size)
        self._slots: list[int] = []
        self._replans: list[int] = []
        self._batch_cursor = 0
        self._batch_slots: list[int] = []
        self._batch_replans: list[int] = []
        self._best_cost: dict[tuple[int, int], float] = {}
        self._best_action: dict[tuple[int, int], torch.Tensor] = {}

    def begin_solve(self, slots: list[int], replans: list[int]) -> None:
        self._slots = slots
        self._replans = replans

    def reset(self) -> None:
        super().reset()
        self._batch_cursor = 0

    def start_batch(self) -> None:
        super().start_batch()
        end = min(self._batch_cursor + self.batch_size, len(self._slots))
        self._batch_slots = self._slots[self._batch_cursor:end]
        self._batch_replans = self._replans[self._batch_cursor:end]
        self._batch_cursor = end

    def compute(self, **state: Any) -> None:
        candidates = state["candidates"]
        costs = state["costs"]
        elites = state["topk_candidates"]
        topk_vals = state["topk_vals"]
        prev_mean = state["prev_mean"]
        prev_var = state["prev_var"]
        mean = state["mean"]
        var = state["var"]
        step = int(state["step"])

        for row, (slot, replan) in enumerate(
            zip(self._batch_slots, self._batch_replans)
        ):
            row_candidates = candidates[row]
            row_elites = elites[row]
            context = (slot, replan)
            iteration_best_index = int(torch.argmin(costs[row]).item())
            iteration_best_cost = float(costs[row, iteration_best_index].item())
            if (
                context not in self._best_cost
                or iteration_best_cost < self._best_cost[context]
            ):
                self._best_cost[context] = iteration_best_cost
                self._best_action[context] = _cpu(
                    row_candidates[iteration_best_index]
                )
            row_summary = {
                "slot": slot,
                "replan": replan,
                "iteration": step,
                "candidate_count": int(row_candidates.shape[0]),
                "cost_min": float(costs[row].min().item()),
                "cost_mean": float(costs[row].mean().item()),
                "elite_cost_mean": float(topk_vals[row].mean().item()),
                "best_cost_so_far": self._best_cost[context],
                "mean_norm": float(mean[row].norm().item()),
                "variance_mean": float(var[row].mean().item()),
                "mean_shift": float((mean[row] - prev_mean[row]).norm().item()),
                "elite_spread": float(
                    row_elites.std(dim=0, unbiased=False).mean().item()
                ),
                "candidate_min": float(row_candidates.min().item()),
                "candidate_max": float(row_candidates.max().item()),
                "candidate_abs_mean": float(row_candidates.abs().mean().item()),
                "candidate_out_of_bounds_fraction": float(
                    (row_candidates.abs() > 1).float().mean().item()
                ),
            }
            self.collector.summaries.append(row_summary)
            if step in self.collector.detailed_steps:
                self.collector.details.append(
                    {
                        "slot": slot,
                        "replan": replan,
                        "iteration": step,
                        "candidates": _cpu(row_candidates),
                        "costs": _cpu(costs[row]),
                        "topk_indices": _cpu(state["topk_inds"][row]),
                        "elites": _cpu(row_elites),
                        "previous_mean": _cpu(prev_mean[row]),
                        "previous_variance": _cpu(prev_var[row]),
                        "updated_mean": _cpu(mean[row]),
                        "updated_variance": _cpu(var[row]),
                        "best_ever": self._best_action[context].clone(),
                    }
                )
        return None


class TracedCEMSolver:
    """Adapter adding trace context around an unchanged installed CEM solver."""

    def __init__(self, solver: Any, collector: CEMTraceCollector):
        if not hasattr(solver, "callbacks"):
            raise TypeError("traced solver requires a callback-capable CEM solver")
        self.solver = solver
        self.collector = collector
        self.callback = CEMTraceCallback(collector, solver.batch_size)
        solver.callbacks.append(self.callback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.solver, name)

    def configure(self, **kwargs: Any) -> None:
        self.solver.configure(**kwargs)

    @property
    def action_dim(self) -> int:
        return self.solver.action_dim

    @property
    def n_envs(self) -> int:
        return self.solver.n_envs

    @property
    def horizon(self) -> int:
        return self.solver.horizon

    def __call__(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        return self.solve(info_dict, init_action=init_action)

    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        clean_info = dict(info_dict)
        raw_slots = clean_info.pop(SLOT_ID_KEY, None)
        if raw_slots is None:
            raise KeyError(f"traced CEM input is missing {SLOT_ID_KEY!r}")
        if torch.is_tensor(raw_slots):
            raw_slots = raw_slots.detach().cpu().numpy()
        slots, replans = self.collector.next_context(
            np.asarray(raw_slots).reshape(-1).tolist()
        )
        prefixes = {
            slot: self.collector.executed_prefix(slot) for slot in slots
        }
        planning_info = {}
        for key, value in clean_info.items():
            if torch.is_tensor(value):
                planning_info[key] = _cpu(value)
            elif isinstance(value, np.ndarray):
                planning_info[key] = value.copy()
        self.callback.begin_solve(slots, replans)
        outputs = self.solver.solve(clean_info, init_action=init_action)
        for row, (slot, replan) in enumerate(zip(slots, replans)):
            row_info = {}
            for key, value in planning_info.items():
                row_info[key] = value[row : row + 1]
            self.collector.plans.append(
                {
                    "slot": slot,
                    "replan": replan,
                    "normalized_plan": _cpu(outputs["actions"][row]),
                    "executed_prefix_before": prefixes[slot],
                    "planning_info": row_info,
                }
            )
        return outputs


class TracedPolicy(swm.policy.BasePolicy):
    """World Policy adapter injecting slots and recording executed actions."""

    def __init__(self, policy: Any, collector: CEMTraceCollector):
        super().__init__()
        self.policy = policy
        self.collector = collector

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def set_env(self, env: Any) -> None:
        self.env = env
        self.policy.set_env(env)

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
        traced_info = dict(info_dict)
        traced_info[SLOT_ID_KEY] = np.arange(self.env.num_envs, dtype=np.int64)
        raw = np.asarray(self.policy.get_action(traced_info, **kwargs))
        terminated = np.asarray(
            info_dict.get("terminated", np.zeros(self.env.num_envs)),
            dtype=bool,
        ).reshape(self.env.num_envs, -1).any(axis=1)
        normalized = raw.copy()
        scaler = getattr(self.policy, "process", {}).get("action")
        finite = (
            np.isfinite(raw.reshape(self.env.num_envs, -1)).all(axis=1)
            & ~terminated
        )
        if scaler is not None and finite.any():
            normalized[finite] = scaler.transform(raw[finite])
        for slot in np.flatnonzero(finite):
            self.collector.executed_raw[int(slot)].append(raw[slot].copy())
            self.collector.executed_normalized[int(slot)].append(
                normalized[slot].copy()
            )
        return raw


@dataclass(frozen=True)
class GroundingPanel:
    actions: torch.Tensor
    sources: tuple[str, ...]
    candidate_indices: tuple[int | None, ...]


def build_shared_initial_panel(
    expert: torch.Tensor,
    actor: torch.Tensor,
    *,
    random_candidates: int,
    actor_neighbors: int,
    actor_noise_std: float,
    generator: torch.Generator,
) -> GroundingPanel:
    """Build the task-level initial panel shared by every scoring model."""
    expert = torch.as_tensor(expert).detach().cpu().float()
    actor = torch.as_tensor(actor).detach().cpu().float()
    if expert.ndim != 2 or actor.shape != expert.shape:
        raise ValueError("expert and actor must share shape [H,A]")
    if random_candidates < 0 or actor_neighbors < 0:
        raise ValueError("candidate counts must be non-negative")
    if actor_noise_std <= 0:
        raise ValueError("actor_noise_std must be positive")
    actions = [
        expert,
        torch.zeros_like(expert),
        expert.flip(0),
        expert.roll(shifts=1, dims=0),
        actor,
    ]
    sources = [
        "expert",
        "zero",
        "time_reverse",
        "time_roll",
        "actor",
    ]
    if random_candidates:
        random = torch.randn(
            int(random_candidates),
            *expert.shape,
            generator=generator,
            dtype=expert.dtype,
        )
        actions.extend(random)
        sources.extend(
            f"random_{index}" for index in range(int(random_candidates))
        )
    if actor_neighbors:
        noise = torch.randn(
            int(actor_neighbors),
            *expert.shape,
            generator=generator,
            dtype=expert.dtype,
        )
        actions.extend(actor.unsqueeze(0) + float(actor_noise_std) * noise)
        sources.extend(
            f"actor_neighbor_{index}"
            for index in range(int(actor_neighbors))
        )
    return GroundingPanel(
        actions=torch.stack(actions),
        sources=tuple(sources),
        candidate_indices=tuple(None for _ in actions),
    )


def build_prefix_replay_actions(
    candidates: torch.Tensor,
    *,
    scaler: Any,
    action_block: int,
    prefix_raw: np.ndarray | None = None,
) -> np.ndarray:
    """Convert normalized plans to raw actions and prepend an exact replay prefix."""
    candidates = candidates.detach().cpu().numpy()
    if candidates.ndim != 3:
        raise ValueError("candidates must have shape [N,H,A]")
    count, horizon, action_dim = candidates.shape
    base_dim = len(scaler.mean_)
    expected_dim = int(action_block) * base_dim
    if action_dim != expected_dim:
        raise ValueError(
            f"candidate action_dim={action_dim}, expected {expected_dim}"
        )
    raw = scaler.inverse_transform(
        candidates.reshape(count * horizon * int(action_block), base_dim)
    ).reshape(count, horizon * int(action_block), base_dim)
    if prefix_raw is None:
        return raw.astype(np.float32, copy=False)
    prefix = np.asarray(prefix_raw, dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != base_dim:
        raise ValueError(
            f"prefix_raw must have shape [T,{base_dim}], got {prefix.shape}"
        )
    repeated = np.repeat(prefix[None], count, axis=0)
    return np.concatenate((repeated, raw), axis=1).astype(
        np.float32, copy=False
    )


class FixedRawSequencePolicy(swm.policy.BasePolicy):
    """Execute one precomputed raw action sequence per simulator environment."""

    def __init__(self, actions: np.ndarray):
        super().__init__()
        self.set_actions(actions)

    def set_actions(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError("fixed actions must have shape [N,T,A]")
        self.actions = actions
        self.step = 0

    def set_env(self, env: Any) -> None:
        self.env = env

    def get_action(self, info: dict[str, Any]) -> np.ndarray:
        index = min(self.step, self.actions.shape[1] - 1)
        self.step += 1
        return self.actions[:, index]


def normalize_expert_plan(
    chunk: dict[str, Any],
    *,
    scaler: Any,
    horizon: int,
    action_block: int,
) -> torch.Tensor:
    """Convert one raw expert chunk to the planner's normalized block layout."""
    raw = np.asarray(chunk["action"], dtype=np.float32)
    plan_steps = int(horizon) * int(action_block)
    raw = raw[:plan_steps]
    if raw.shape[0] != plan_steps:
        raise ValueError(
            f"expert chunk has {raw.shape[0]} actions, expected {plan_steps}"
        )
    normalized = scaler.transform(np.nan_to_num(raw, nan=0.0))
    return torch.from_numpy(normalized.reshape(int(horizon), -1)).float()


def prepare_image_info(raw_info, transform, device):
    """Apply the canonical world-policy image preprocessing to simulator info."""
    preprocessor = swm.policy.BasePolicy()
    preprocessor.process = {}
    preprocessor.transform = transform
    prepared = preprocessor._prepare_info(raw_info)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in prepared.items()
    }


def compute_physical_terminal_cost(task: str, info: dict[str, Any]) -> np.ndarray:
    """Return the task's terminal physical-state distance to its goal."""
    if task == "pusht":
        current_key, goal_key = "state", "goal_state"
    elif task == "reacher":
        current_key, goal_key = "qpos", "goal_qpos"
    else:
        raise ValueError(f"unsupported diagnostic task {task!r}")
    current = np.asarray(info[current_key]).reshape(len(info[current_key]), -1)
    goal = np.asarray(info[goal_key]).reshape(len(info[goal_key]), -1)
    return np.linalg.norm(current - goal, axis=1)


def select_grounding_panel(
    *,
    candidates: torch.Tensor,
    costs: torch.Tensor,
    topk_indices: torch.Tensor,
    previous_mean: torch.Tensor,
    best_ever: torch.Tensor,
    anchors: dict[str, torch.Tensor],
    non_elite_count: int = 10,
    seed: int = 42,
) -> GroundingPanel:
    """Select the fixed CEM grounding panel and deduplicate identical actions."""
    candidates = candidates.detach().cpu()
    costs = costs.detach().cpu().reshape(-1)
    elite_indices = [int(index) for index in topk_indices.detach().cpu().reshape(-1)]
    elite_set = set(elite_indices)
    non_elite = [index for index in range(len(candidates)) if index not in elite_set]
    count = min(max(int(non_elite_count), 0), len(non_elite))
    if count:
        chosen = np.random.default_rng(int(seed)).choice(
            non_elite, size=count, replace=False
        ).tolist()
    else:
        chosen = []
    best_index = int(torch.argmin(costs).item())
    elite_mean = candidates[elite_indices].mean(dim=0)

    entries: list[tuple[str, int | None, torch.Tensor]] = []
    entries.extend(
        (f"elite_{rank}", index, candidates[index])
        for rank, index in enumerate(elite_indices)
    )
    entries.extend(
        (f"non_elite_{rank}", index, candidates[index])
        for rank, index in enumerate(chosen)
    )
    entries.extend(
        [
            ("previous_mean", None, previous_mean.detach().cpu()),
            ("elite_mean", None, elite_mean),
            ("iteration_best", best_index, candidates[best_index]),
            ("best_ever", None, best_ever.detach().cpu()),
        ]
    )
    entries.extend(
        (f"anchor_{name}", None, value.detach().cpu())
        for name, value in sorted(anchors.items())
    )

    unique: list[torch.Tensor] = []
    sources: list[str] = []
    indices: list[int | None] = []
    seen: set[tuple[str, tuple[int, ...], bytes]] = set()
    for source, index, action in entries:
        action = action.contiguous()
        key = (str(action.dtype), tuple(action.shape), action.numpy().tobytes())
        if key in seen:
            continue
        seen.add(key)
        unique.append(action.clone())
        sources.append(source)
        indices.append(index)
    return GroundingPanel(
        actions=torch.stack(unique),
        sources=tuple(sources),
        candidate_indices=tuple(indices),
    )


def should_expand_grounding(
    *,
    panel_successes: Iterable[bool],
    final_selection_success: bool,
    error_decomposition_clear: bool,
) -> str | None:
    """Return the reason that this panel needs all 300 candidates grounded."""
    successes = [bool(value) for value in panel_successes]
    if not any(successes):
        return "no_success_in_panel"
    if not final_selection_success:
        return None
    if not error_decomposition_clear:
        return "ambiguous_error_decomposition"
    return None


def select_diagnostic_slots(
    success_vectors: dict[str, Iterable[bool]],
) -> dict[str, list[int]]:
    """Apply the fixed failure/control/rescue slot selection protocol."""
    required = {"e0", "e1_384", "e3_384"}
    missing = required - set(success_vectors)
    if missing:
        raise KeyError(f"missing success vectors: {sorted(missing)}")
    vectors = {
        key: np.asarray(list(value), dtype=bool)
        for key, value in success_vectors.items()
    }
    lengths = {len(value) for value in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("success vectors must have equal length")
    failure_union = np.flatnonzero(
        ~vectors["e1_384"] | ~vectors["e3_384"]
    ).tolist()
    all_success = np.logical_and.reduce(list(vectors.values()))
    controls = np.flatnonzero(all_success)[:2].tolist()
    e0_rescues = np.flatnonzero(
        ~vectors["e0"] & vectors["e1_384"] & vectors["e3_384"]
    )[:2].tolist()
    return {
        "failure_union": failure_union,
        "controls": controls,
        "e0_rescues": e0_rescues,
        "slots": sorted(set(failure_union + controls + e0_rescues)),
    }


def _strictly_decreasing(values: Any) -> bool:
    if values is None or len(values) != 3:
        return False
    return bool(values[0] > values[1] > values[2])


def _strictly_increasing(values: Any) -> bool:
    if values is None or len(values) != 3:
        return False
    return bool(values[0] < values[1] < values[2])


def classify_failure(evidence: dict[str, Any]) -> list[str]:
    """Apply the predeclared multi-label attribution rules to grounded evidence."""
    if evidence.get("final_success"):
        return []
    labels: list[str] = []
    successes = [bool(value) for value in evidence.get("candidate_successes", [])]
    if evidence.get("all_candidates_grounded") and successes and not any(successes):
        labels.append("coverage_failure")
    final_selection_success = evidence.get(
        "final_selection_success",
        evidence.get("final_success"),
    )
    if any(successes) and final_selection_success is False:
        labels.append("ranking_failure")

    predicted_latent = evidence.get("predicted_true_latent_spearman")
    latent_physical = evidence.get("true_latent_physical_spearman")
    if predicted_latent is not None and latent_physical is not None:
        if predicted_latent < 0 and latent_physical > 0:
            labels.append("dynamics_error")
        elif predicted_latent > 0 and latent_physical < 0:
            labels.append("latent_metric_error")

    predicted_curve = evidence.get("predicted_elite_cost_0_5_29")
    physical_curve = evidence.get("physical_cost_0_5_29")
    earlier_best = evidence.get("earlier_best_physical_cost")
    final_mean = evidence.get("final_mean_physical_cost")
    if (
        _strictly_decreasing(predicted_curve)
        and _strictly_increasing(physical_curve)
        and earlier_best is not None
        and final_mean is not None
        and earlier_best < final_mean
    ):
        labels.append("cem_exploitation")

    distances = evidence.get("physical_distance_start_after_first_final")
    if (
        distances is not None
        and len(distances) == 3
        and distances[1] < distances[0]
        and distances[2] > distances[1]
    ):
        labels.append("replan_regression")

    failed_oob = evidence.get("failed_action_out_of_bounds_fraction")
    reference_oob = evidence.get("reference_action_out_of_bounds_fraction")
    failed_norm = evidence.get("failed_action_norm")
    reference_norm = evidence.get("reference_action_norm")
    if (
        failed_oob is not None
        and reference_oob is not None
        and failed_norm is not None
        and reference_norm is not None
        and (
            failed_oob > reference_oob + 0.05
            or failed_norm > 1.5 * max(reference_norm, 1e-12)
        )
    ):
        labels.append("action_ood")

    if len(labels) != 1:
        labels.append("mixed_or_undetermined")
    return labels


def _validate_trace_finite(collector: CEMTraceCollector) -> None:
    for row in collector.summaries:
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"non-finite trace summary {key}={value}")
    for detail in collector.details:
        for key, value in detail.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(
                    f"non-finite trace tensor at slot={detail['slot']} "
                    f"replan={detail['replan']} iteration={detail['iteration']} "
                    f"field={key}"
                )


def write_trace_artifacts(
    output_dir: str | Path,
    *,
    run_label: str,
    collector: CEMTraceCollector,
    actual_successes: Iterable[bool],
    expected_successes: Iterable[bool],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Validate and persist one exact-replay trace using a stable output schema."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slots_dir = output_dir / "slots"
    slots_dir.mkdir(parents=True, exist_ok=True)
    _validate_trace_finite(collector)
    actual = [bool(value) for value in actual_successes]
    expected = [bool(value) for value in expected_successes]
    if len(actual) != len(expected):
        raise ValueError("actual and expected success vectors have unequal length")
    mismatch_slots = [
        slot
        for slot, (left, right) in enumerate(zip(actual, expected))
        if left != right
    ]
    status = "reproducible" if not mismatch_slots else "non_reproducible"

    tensor_payload = {
        "schema_version": 1,
        "details": collector.details,
        "plans": collector.plans,
        "executed_raw": {
            slot: np.stack(values) if values else np.empty((0,), dtype=np.float32)
            for slot, values in collector.executed_raw.items()
        },
        "executed_normalized": {
            slot: np.stack(values) if values else np.empty((0,), dtype=np.float32)
            for slot, values in collector.executed_normalized.items()
        },
    }
    torch.save(tensor_payload, output_dir / "trace.pt")

    summary_by_context: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in collector.summaries:
        summary_by_context[(int(row["slot"]), int(row["replan"]))].append(row)
    plan_by_context = {
        (int(plan["slot"]), int(plan["replan"])): plan
        for plan in collector.plans
    }
    for slot in range(len(actual)):
        contexts = sorted(
            (
                (replan, rows)
                for (row_slot, replan), rows in summary_by_context.items()
                if row_slot == slot
            ),
            key=lambda item: item[0],
        )
        replans = []
        for replan, rows in contexts:
            rows = sorted(rows, key=lambda row: int(row["iteration"]))
            plan = plan_by_context.get((slot, replan))
            replans.append(
                {
                    "replan": replan,
                    "iteration_count": len(rows),
                    "iterations": [int(row["iteration"]) for row in rows],
                    "candidate_counts": sorted(
                        {int(row["candidate_count"]) for row in rows}
                    ),
                    "final_summary": rows[-1],
                    "plan_saved": plan is not None,
                    "executed_prefix_before_count": (
                        int(len(plan["executed_prefix_before"]))
                        if plan is not None
                        else None
                    ),
                }
            )
        slot_payload = {
            "slot": slot,
            "expected_success": expected[slot],
            "actual_success": actual[slot],
            "reproduced": actual[slot] == expected[slot],
            "replans": replans,
            "executed_raw_action_count": len(
                collector.executed_raw.get(slot, [])
            ),
            "executed_normalized_action_count": len(
                collector.executed_normalized.get(slot, [])
            ),
            "early_termination_reason": (
                "environment_terminated_before_second_replan"
                if len(replans) < 2
                else None
            ),
            "attribution_labels": (
                []
                if status == "reproducible"
                else ["non_reproducible"]
            ),
        }
        (slots_dir / f"slot_{slot:02d}.json").write_text(
            json.dumps(
                slot_payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    result = {
        "schema_version": 1,
        "run_label": str(run_label),
        "status": status,
        "success_vector": actual,
        "reference_success_vector": expected,
        "mismatch_slots": mismatch_slots,
        "metadata": metadata,
        "trace": {
            "summary_rows": len(collector.summaries),
            "detail_rows": len(collector.details),
            "plan_rows": len(collector.plans),
            "slot_files": len(actual),
            "tensor_file": "trace.pt",
        },
    }
    (output_dir / "trace_summary.json").write_text(
        json.dumps(
            collector.summaries,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "reproduction.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "CEMTraceCollector",
    "CEMTraceCallback",
    "DiagnosticManifest",
    "DiagnosticRun",
    "FixedRawSequencePolicy",
    "GroundingPanel",
    "SLOT_ID_KEY",
    "TracedCEMSolver",
    "TracedPolicy",
    "build_prefix_replay_actions",
    "build_shared_initial_panel",
    "classify_failure",
    "compute_physical_terminal_cost",
    "load_diagnostic_manifest",
    "normalize_expert_plan",
    "prepare_image_info",
    "select_diagnostic_slots",
    "select_grounding_panel",
    "should_expand_grounding",
    "write_trace_artifacts",
]
