"""Recoverable, simulator-grounded Fast-LeWAM epoch-pair diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import spearmanr
import torch


SLOT_CATEGORIES = (
    "regression",
    "improvement_control",
    "stable_success",
    "stable_failure",
)
PAIR_METRIC_KEYS = (
    "predicted_physical_spearman",
    "predicted_true_terminal_latent_spearman",
    "true_terminal_latent_physical_spearman",
    "physical_topk_recall",
    "successful_candidate_recall_at_k",
    "top1_success",
    "oracle_success",
    "candidate_success_rate",
    "physical_best_candidate_regret",
    "predicted_terminal_latent_mse",
)


@dataclass(frozen=True)
class PairEpoch:
    label: str
    epoch: int
    checkpoint: Path
    reference_result: Path


@dataclass(frozen=True)
class EpochPair:
    label: str
    task: str
    section: str
    preferred_gpu: int
    run_config: Path
    epochs: dict[str, PairEpoch]
    slot_categories: dict[str, tuple[int, ...]]

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                slot
                for category in SLOT_CATEGORIES
                for slot in self.slot_categories[category]
            )
        )

    def category_for(self, slot: int) -> str:
        matches = [
            category
            for category, slots in self.slot_categories.items()
            if int(slot) in slots
        ]
        if len(matches) != 1:
            raise KeyError(f"slot {slot} belongs to {len(matches)} categories")
        return matches[0]


@dataclass(frozen=True)
class EpochPairManifest:
    path: Path
    root: Path
    output_root: Path
    protocol: dict[str, Any]
    pairs: dict[str, EpochPair]


@dataclass(frozen=True)
class CandidatePanel:
    actions: torch.Tensor
    sources: tuple[str, ...]
    source_aliases: tuple[tuple[str, ...], ...]
    sha256: str


def _reference_successes(path: Path) -> tuple[dict[str, Any], list[bool]]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    return reference, [bool(row["success"]) for row in reference["episodes"]]


def _observed_category(e8_success: bool, e10_success: bool) -> str:
    if e8_success and not e10_success:
        return "regression"
    if not e8_success and e10_success:
        return "improvement_control"
    if e8_success:
        return "stable_success"
    return "stable_failure"


def _validate_reference_protocol(
    *, label: str, reference: dict[str, Any], protocol: dict[str, Any]
) -> None:
    parameters = reference["parameters"]
    checks = {
        "seed": parameters["seed"],
        "num_eval": parameters["num_eval"],
        "goal_offset_steps": parameters["goal_offset_steps"],
        "eval_budget": parameters["eval_budget"],
        "horizon": parameters["plan_config"]["horizon"],
        "receding_horizon": parameters["plan_config"]["receding_horizon"],
        "action_block": parameters["plan_config"]["action_block"],
        "num_samples": parameters["solver"]["num_samples"],
        "n_steps": parameters["solver"]["n_steps"],
        "topk": parameters["solver"]["topk"],
    }
    mismatches = {
        key: (int(actual), int(protocol[key]))
        for key, actual in checks.items()
        if int(actual) != int(protocol[key])
    }
    if mismatches:
        raise ValueError(f"{label} reference protocol mismatch: {mismatches}")
    if reference.get("status") != "ok":
        raise ValueError(f"{label} reference status is not ok")
    if len(reference["episodes"]) != int(protocol["num_eval"]):
        raise ValueError(f"{label} reference episode count mismatch")


def load_epoch_pair_manifest(path: str | Path) -> EpochPairManifest:
    """Validate artifact identities, shared cohorts, and fixed slot classes."""
    path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if int(raw.get("version", 0)) != 1:
        raise ValueError("epoch-pair manifest version must be 1")
    root = (path.parent / str(raw["repository_root"])).resolve()
    output_root = (root / str(raw["output_root"])).resolve()
    protocol = dict(raw["protocol"])
    pairs: dict[str, EpochPair] = {}
    for pair_label, pair_raw in raw["pairs"].items():
        run_config = (root / str(pair_raw["run_config"])).resolve()
        if not run_config.is_file():
            raise FileNotFoundError(run_config)
        epochs: dict[str, PairEpoch] = {}
        references: dict[str, dict[str, Any]] = {}
        successes: dict[str, list[bool]] = {}
        for epoch_label in ("e8", "e10"):
            value = pair_raw["epochs"][epoch_label]
            epoch = PairEpoch(
                label=epoch_label,
                epoch=int(value["epoch"]),
                checkpoint=(root / str(value["checkpoint"])).resolve(),
                reference_result=(
                    root / str(value["reference_result"])
                ).resolve(),
            )
            for artifact in (epoch.checkpoint, epoch.reference_result):
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
            reference, success = _reference_successes(epoch.reference_result)
            _validate_reference_protocol(
                label=f"{pair_label}/{epoch_label}",
                reference=reference,
                protocol=protocol,
            )
            epochs[epoch_label] = epoch
            references[epoch_label] = reference
            successes[epoch_label] = success
        if epochs["e8"].epoch != 8 or epochs["e10"].epoch != 10:
            raise ValueError(f"{pair_label} must compare epochs 8 and 10")
        for cohort_key in ("episode_ids", "start_steps", "start_rows"):
            left = references["e8"]["parameters"][cohort_key]
            right = references["e10"]["parameters"][cohort_key]
            if left != right:
                raise ValueError(
                    f"{pair_label} e8/e10 cohort differs at {cohort_key}"
                )
        slots_raw = dict(pair_raw["slots"])
        if set(slots_raw) != set(SLOT_CATEGORIES):
            raise ValueError(
                f"{pair_label} slots must define {list(SLOT_CATEGORIES)}"
            )
        slot_categories = {
            category: tuple(int(slot) for slot in slots_raw[category])
            for category in SLOT_CATEGORIES
        }
        flat_slots = [
            slot for category in SLOT_CATEGORIES for slot in slot_categories[category]
        ]
        if len(flat_slots) != len(set(flat_slots)):
            raise ValueError(f"{pair_label} slot categories overlap")
        for category, slots in slot_categories.items():
            for slot in slots:
                if slot < 0 or slot >= int(protocol["num_eval"]):
                    raise ValueError(f"{pair_label} slot {slot} is out of range")
                observed = _observed_category(
                    successes["e8"][slot], successes["e10"][slot]
                )
                if observed != category:
                    raise ValueError(
                        f"{pair_label} slot {slot} declared {category}, "
                        f"observed {observed}"
                    )
        pairs[str(pair_label)] = EpochPair(
            label=str(pair_label),
            task=str(pair_raw["task"]),
            section=str(pair_raw["section"]),
            preferred_gpu=int(pair_raw["preferred_gpu"]),
            run_config=run_config,
            epochs=epochs,
            slot_categories=slot_categories,
        )
    if not pairs:
        raise ValueError("epoch-pair manifest contains no pairs")
    return EpochPairManifest(
        path=path,
        root=root,
        output_root=output_root,
        protocol=protocol,
        pairs=pairs,
    )


def validate_diagnostic_device(device: str) -> tuple[int, ...]:
    """Enforce explicit visibility and prohibit physical GPUs 4-7."""
    device = str(device)
    if device == "cpu":
        return ()
    if device != "cuda:0":
        raise ValueError(
            "GPU diagnostics use internal device cuda:0; choose the physical "
            "GPU through CUDA_VISIBLE_DEVICES"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise EnvironmentError(
            "CUDA_VISIBLE_DEVICES must explicitly select only GPU0-3"
        )
    try:
        selected = tuple(int(token.strip()) for token in visible.split(","))
    except ValueError as exc:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be a comma-separated list of GPU IDs"
        ) from exc
    if not selected or any(gpu not in {0, 1, 2, 3} for gpu in selected):
        raise ValueError(
            f"prohibited CUDA_VISIBLE_DEVICES={visible!r}; only GPU0-3 are allowed"
        )
    return selected


def validate_pair_device(pair: EpochPair, device: str) -> tuple[int, ...]:
    """Pin reproducible execution to the physical GPU declared by the pair."""
    selected = validate_diagnostic_device(device)
    if str(device) != "cpu" and selected != (int(pair.preferred_gpu),):
        raise ValueError(
            f"{pair.label} must use preferred physical GPU{pair.preferred_gpu}; "
            f"got CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    return selected


def _action_key(action: torch.Tensor) -> tuple[str, tuple[int, ...], bytes]:
    action = action.detach().cpu().contiguous()
    return str(action.dtype), tuple(action.shape), action.numpy().tobytes()


def candidate_actions_sha256(actions: torch.Tensor) -> str:
    """Hash action dtype, shape, order, and exact contiguous bytes."""
    actions = actions.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(actions.dtype).encode("utf-8"))
    digest.update(json.dumps(list(actions.shape)).encode("ascii"))
    digest.update(actions.numpy().tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _sha256_file_cached(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _sha256_file_cached(str(path), stat.st_size, stat.st_mtime_ns)


def write_atomic_json(path: str | Path, payload: Any) -> None:
    """Write strict JSON atomically in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_atomic_text(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_atomic_torch(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_identity(
    actual: dict[str, Any], expected: dict[str, Any], *, source: str | Path
) -> None:
    if actual != expected:
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(
            f"stale cache identity at {source}: differing keys={differing}"
        )


def require_matching_cache_identity(
    path: str | Path, expected: dict[str, Any]
) -> bool:
    """Return False for a miss and refuse reuse for any stale JSON cache."""
    path = Path(path)
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_identity(payload.get("identity", {}), expected, source=path)
    return True


def build_pair_identity(
    manifest: EpochPairManifest,
    pair: EpochPair,
    *,
    phase: str,
    epoch_label: str | None = None,
    prefix_sha256: str | None = None,
    candidate_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve every immutable input that controls a cached phase artifact."""
    identity: dict[str, Any] = {
        "schema_version": 1,
        "phase": str(phase),
        "pair": pair.label,
        "manifest_sha256": sha256_file(manifest.path),
        "config_sha256": sha256_file(pair.run_config),
        "protocol": dict(manifest.protocol),
        "epochs": {
            label: {
                "epoch": epoch.epoch,
                "checkpoint_sha256": sha256_file(epoch.checkpoint),
                "reference_result_sha256": sha256_file(
                    epoch.reference_result
                ),
            }
            for label, epoch in pair.epochs.items()
        },
    }
    if epoch_label is not None:
        identity["epoch_label"] = str(epoch_label)
    if prefix_sha256 is not None:
        identity["prefix_sha256"] = str(prefix_sha256)
    if candidate_sha256 is not None:
        identity["candidate_sha256"] = str(candidate_sha256)
    return identity


def build_ground_identity(
    manifest: EpochPairManifest,
    pair: EpochPair,
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind a pair-level ground result to every traced state and candidate."""
    if set(traces) != {"e8", "e10"}:
        raise ValueError("ground identity requires e8 and e10 traces")
    prefix_records = []
    candidate_records = []
    state_records = []
    for epoch_label in ("e8", "e10"):
        trace = traces[epoch_label]
        for plan in trace.get("plans", []):
            context = [epoch_label, int(plan["slot"]), int(plan["replan"])]
            prefix_records.append(
                context
                + [tensor_bytes_sha256(plan["executed_prefix_before"])]
            )
            state_records.append(
                context + [planning_state_sha256(plan["planning_info"])]
            )
        for detail in trace.get("details", []):
            candidate_records.append(
                [
                    epoch_label,
                    int(detail["slot"]),
                    int(detail["replan"]),
                    int(detail["iteration"]),
                    *[
                        tensor_bytes_sha256(detail[key])
                        for key in (
                            "candidates",
                            "costs",
                            "topk_indices",
                            "previous_mean",
                            "best_ever",
                        )
                    ],
                ]
            )

    def digest(records: list[list[Any]]) -> str:
        payload = json.dumps(
            sorted(records), separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    identity = build_pair_identity(
        manifest,
        pair,
        phase="ground",
        prefix_sha256=digest(prefix_records),
        candidate_sha256=digest(candidate_records),
    )
    identity["planning_state_sha256"] = digest(state_records)
    return identity


def build_summary_identity(
    manifest: EpochPairManifest,
    pair: EpochPair,
    ground_identity: dict[str, Any],
) -> dict[str, Any]:
    """Bind a summary to the exact prefix, state, and candidate ground inputs."""
    identity = build_pair_identity(manifest, pair, phase="summarize")
    for key in (
        "prefix_sha256",
        "candidate_sha256",
        "planning_state_sha256",
    ):
        identity[key] = ground_identity[key]
    return identity


def _validate_detail_labels(details: dict[str, dict[str, Any]]) -> None:
    labels = set(details)
    if not labels or not labels <= {"e8", "e10"}:
        raise ValueError("details must contain one or both of e8 and e10")


def build_cross_epoch_panel(
    details: dict[str, dict[str, Any]],
    *,
    expert: torch.Tensor,
    non_elite_count: int = 10,
    seed: int = 42,
) -> CandidatePanel:
    """Union available epochs' selected CEM candidates behind one interface."""
    _validate_detail_labels(details)
    if int(non_elite_count) < 0:
        raise ValueError("non_elite_count must be non-negative")
    entries: list[tuple[str, torch.Tensor]] = []
    for position, epoch_label in enumerate(
        label for label in ("e8", "e10") if label in details
    ):
        detail = details[epoch_label]
        candidates = torch.as_tensor(detail["candidates"]).detach().cpu()
        costs = torch.as_tensor(detail["costs"]).detach().cpu().reshape(-1)
        elite_indices = [
            int(index)
            for index in torch.as_tensor(detail["topk_indices"]).reshape(-1)
        ]
        if len(candidates) != len(costs):
            raise ValueError(f"{epoch_label} candidate/cost count mismatch")
        if not elite_indices or any(
            index < 0 or index >= len(candidates) for index in elite_indices
        ):
            raise ValueError(f"{epoch_label} has invalid elite indices")
        entries.extend(
            (f"{epoch_label}:elite_{rank}", candidates[index])
            for rank, index in enumerate(elite_indices)
        )
        elite_set = set(elite_indices)
        non_elites = [
            index for index in range(len(candidates)) if index not in elite_set
        ]
        count = min(int(non_elite_count), len(non_elites))
        if count:
            generator = np.random.default_rng(
                int(seed) + 104729 * position
            )
            chosen = generator.choice(
                non_elites, size=count, replace=False
            ).tolist()
            entries.extend(
                (f"{epoch_label}:non_elite_{rank}", candidates[index])
                for rank, index in enumerate(chosen)
            )
        best_index = int(torch.argmin(costs).item())
        entries.extend(
            (
                (f"{epoch_label}:previous_mean", detail["previous_mean"]),
                (
                    f"{epoch_label}:elite_mean",
                    candidates[elite_indices].mean(dim=0),
                ),
                (f"{epoch_label}:iteration_best", candidates[best_index]),
                (
                    f"{epoch_label}:best_ever",
                    detail.get("best_ever", candidates[best_index]),
                ),
            )
        )
    entries.append(("anchor:expert", expert))
    unique_actions: list[torch.Tensor] = []
    unique_sources: list[str] = []
    source_aliases: list[list[str]] = []
    seen: dict[tuple[str, tuple[int, ...], bytes], int] = {}
    for source, raw_action in entries:
        action = torch.as_tensor(raw_action).detach().cpu().contiguous()
        key = _action_key(action)
        if key in seen:
            source_aliases[seen[key]].append(source)
            continue
        seen[key] = len(unique_actions)
        unique_actions.append(action.clone())
        unique_sources.append(source)
        source_aliases.append([source])
    if not unique_actions:
        raise ValueError("candidate panel is empty")
    actions = torch.stack(unique_actions)
    return CandidatePanel(
        actions=actions,
        sources=tuple(unique_sources),
        source_aliases=tuple(tuple(values) for values in source_aliases),
        sha256=candidate_actions_sha256(actions),
    )


def build_full_cross_epoch_panel(
    details: dict[str, dict[str, Any]], *, expert: torch.Tensor
) -> CandidatePanel:
    """Union available epochs' complete candidates for adaptive grounding."""
    _validate_detail_labels(details)
    entries = [
        (f"{epoch_label}:candidate_{index}", candidate)
        for epoch_label in (
            label for label in ("e8", "e10") if label in details
        )
        for index, candidate in enumerate(details[epoch_label]["candidates"])
    ]
    entries.append(("anchor:expert", expert))
    actions: list[torch.Tensor] = []
    sources: list[str] = []
    source_aliases: list[list[str]] = []
    seen: dict[tuple[str, tuple[int, ...], bytes], int] = {}
    for source, raw_action in entries:
        action = torch.as_tensor(raw_action).detach().cpu().contiguous()
        key = _action_key(action)
        if key in seen:
            source_aliases[seen[key]].append(source)
            continue
        seen[key] = len(actions)
        actions.append(action.clone())
        sources.append(source)
        source_aliases.append([source])
    stacked = torch.stack(actions)
    return CandidatePanel(
        actions=stacked,
        sources=tuple(sources),
        source_aliases=tuple(tuple(values) for values in source_aliases),
        sha256=candidate_actions_sha256(stacked),
    )


def tensor_bytes_sha256(value: Any) -> str:
    array = (
        value.detach().cpu().contiguous().numpy()
        if torch.is_tensor(value)
        else np.ascontiguousarray(value)
    )
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def planning_state_sha256(planning_info: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(planning_info):
        if key == "render_time" or key.endswith("_render_time"):
            continue
        value = planning_info[key]
        if not (torch.is_tensor(value) or isinstance(value, np.ndarray)):
            continue
        digest.update(str(key).encode("utf-8"))
        digest.update(tensor_bytes_sha256(value).encode("ascii"))
    return digest.hexdigest()


def cem_candidate_indices(panel: CandidatePanel) -> tuple[int, ...]:
    """Return proposal indices, excluding a standalone expert anchor."""
    return tuple(
        index
        for index, aliases in enumerate(panel.source_aliases)
        if any(not source.startswith("anchor:") for source in aliases)
    )


def compute_pair_physical_terminal_cost(
    task: str, info: dict[str, Any]
) -> np.ndarray:
    """Return the task-aligned physical terminal distance for paired ranking."""
    if task == "pusht":
        current_key, goal_key = "state", "goal_state"
    elif task == "reacher":
        current_key, goal_key = "qpos", "goal_qpos"
    elif task == "cube":
        current_key = "privileged_block_0_pos"
        goal_key = "goal_privileged_block_0_pos"
    else:
        raise ValueError(f"unsupported diagnostic task {task!r}")
    current = np.asarray(info[current_key]).reshape(len(info[current_key]), -1)
    goal = np.asarray(info[goal_key]).reshape(len(info[goal_key]), -1)
    return np.linalg.norm(current - goal, axis=1)


def cross_score_costs(
    models: dict[str, Any],
    planning_info: dict[str, Any],
    actions: torch.Tensor,
    *,
    scorer: Any,
) -> dict[str, np.ndarray]:
    """Score one immutable action panel with every epoch model adapter."""
    panel_hash = candidate_actions_sha256(actions)
    results: dict[str, np.ndarray] = {}
    for label, model in models.items():
        values = np.asarray(
            scorer(model, planning_info, actions), dtype=np.float64
        ).reshape(-1)
        if len(values) != len(actions) or not np.isfinite(values).all():
            raise ValueError(f"{label} scorer returned invalid costs")
        if candidate_actions_sha256(actions) != panel_hash:
            raise RuntimeError("scorer mutated the shared candidate panel")
        results[str(label)] = values
    return results


def write_pair_trace_artifacts(
    output_dir: str | Path,
    *,
    collector: Any,
    selected_slots: Any,
    actual_successes: Any,
    expected_successes: Any,
    identity: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist only fixed-slot details after a full evaluation."""
    output_dir = Path(output_dir)
    selected = tuple(sorted({int(slot) for slot in selected_slots}))
    selected_set = set(selected)
    actual = [bool(value) for value in actual_successes]
    expected = [bool(value) for value in expected_successes]
    if len(actual) != len(expected):
        raise ValueError("actual and expected success vectors have unequal length")
    mismatch_slots = [
        slot
        for slot, (left, right) in enumerate(zip(actual, expected))
        if left != right
    ]
    summaries = [
        row for row in collector.summaries if int(row["slot"]) in selected_set
    ]
    details = [
        row for row in collector.details if int(row["slot"]) in selected_set
    ]
    plans = [
        row for row in collector.plans if int(row["slot"]) in selected_set
    ]
    expected_summary_count = len(plans) * int(protocol["n_steps"])
    expected_detail_count = len(plans) * len(protocol["detailed_iterations"])
    if len(summaries) != expected_summary_count:
        raise RuntimeError(
            f"incomplete selected trace summaries: {len(summaries)} "
            f"!= {expected_summary_count}"
        )
    if len(details) != expected_detail_count:
        raise RuntimeError(
            f"incomplete selected trace details: {len(details)} "
            f"!= {expected_detail_count}"
        )
    for row in summaries:
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"non-finite trace summary {key}={value}")
    for detail in details:
        for key, value in detail.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(
                    f"non-finite trace tensor slot={detail['slot']} "
                    f"replan={detail['replan']} iteration={detail['iteration']} "
                    f"field={key}"
                )
    tensor_payload = {
        "schema_version": 1,
        "identity": identity,
        "selected_slots": selected,
        "summaries": summaries,
        "details": details,
        "plans": plans,
        "executed_raw": {
            slot: np.stack(collector.executed_raw.get(slot, []))
            if collector.executed_raw.get(slot)
            else np.empty((0,), dtype=np.float32)
            for slot in selected
        },
        "executed_normalized": {
            slot: np.stack(collector.executed_normalized.get(slot, []))
            if collector.executed_normalized.get(slot)
            else np.empty((0,), dtype=np.float32)
            for slot in selected
        },
    }
    write_atomic_torch(output_dir / "trace.pt", tensor_payload)
    write_atomic_json(output_dir / "trace_summary.json", summaries)
    plans_by_context = {
        (int(plan["slot"]), int(plan["replan"])): plan for plan in plans
    }
    details_by_context: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        details_by_context[
            (int(detail["slot"]), int(detail["replan"]))
        ].append(detail)
    for slot in selected:
        contexts = []
        for context, context_details in sorted(details_by_context.items()):
            if context[0] != slot:
                continue
            plan = plans_by_context[context]
            context_details.sort(key=lambda row: int(row["iteration"]))
            contexts.append(
                {
                    "replan": context[1],
                    "iterations": [
                        int(row["iteration"]) for row in context_details
                    ],
                    "planning_state_sha256": planning_state_sha256(
                        plan["planning_info"]
                    ),
                    "prefix_sha256": tensor_bytes_sha256(
                        np.asarray(plan["executed_prefix_before"])
                    ),
                    "candidate_sha256": {
                        str(int(row["iteration"])): candidate_actions_sha256(
                            row["candidates"]
                        )
                        for row in context_details
                    },
                }
            )
        write_atomic_json(
            output_dir / "slots" / f"slot_{slot:02d}.json",
            {
                "slot": slot,
                "expected_success": expected[slot],
                "actual_success": actual[slot],
                "reproduced": actual[slot] == expected[slot],
                "replans": contexts,
                "early_termination_reason": (
                    "environment_terminated_before_second_replan"
                    if len(contexts) < 2
                    else None
                ),
            },
        )
    result = {
        "schema_version": 1,
        "status": "reproducible" if not mismatch_slots else "non_reproducible",
        "identity": identity,
        "selected_slots": list(selected),
        "success_vector": actual,
        "reference_success_vector": expected,
        "mismatch_slots": mismatch_slots,
        "trace": {
            "summary_rows": len(summaries),
            "detail_rows": len(details),
            "plan_rows": len(plans),
            "tensor_file": "trace.pt",
        },
    }
    write_atomic_json(output_dir / "reproduction.json", result)
    return result


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def compute_ranking_metrics(
    *,
    predicted_cost: Any,
    true_terminal_latent_cost: Any,
    physical_cost: Any,
    successes: Any,
    predicted_terminal_latents: torch.Tensor,
    true_terminal_latents: torch.Tensor,
    topk: int = 30,
) -> dict[str, Any]:
    """Compute the complete per-model metric bundle for one shared panel."""
    predicted = np.asarray(predicted_cost, dtype=np.float64).reshape(-1)
    latent = np.asarray(
        true_terminal_latent_cost, dtype=np.float64
    ).reshape(-1)
    physical = np.asarray(physical_cost, dtype=np.float64).reshape(-1)
    success = np.asarray(successes, dtype=bool).reshape(-1)
    lengths = {len(predicted), len(latent), len(physical), len(success)}
    if len(lengths) != 1 or not predicted.size:
        raise ValueError("panel metrics require equally sized non-empty arrays")
    if not all(np.isfinite(values).all() for values in (predicted, latent, physical)):
        raise ValueError("panel costs must be finite")
    predicted_terminal_latents = torch.as_tensor(
        predicted_terminal_latents
    ).detach().cpu()
    true_terminal_latents = torch.as_tensor(
        true_terminal_latents
    ).detach().cpu()
    if predicted_terminal_latents.shape != true_terminal_latents.shape:
        raise ValueError("predicted and true terminal latent shapes differ")
    if len(predicted_terminal_latents) != len(predicted):
        raise ValueError("terminal latent count differs from candidate count")
    mse = (
        (predicted_terminal_latents - true_terminal_latents)
        .float()
        .square()
        .mean()
        .item()
    )
    if not np.isfinite(mse):
        raise ValueError("predicted terminal latent MSE is non-finite")
    k = min(max(int(topk), 1), len(predicted))
    predicted_order = np.argsort(predicted, kind="stable")
    physical_order = np.argsort(physical, kind="stable")
    predicted_top = set(predicted_order[:k].tolist())
    physical_top = set(physical_order[:k].tolist())
    top1 = int(predicted_order[0])
    success_count = int(success.sum())
    return {
        "candidate_count": int(len(predicted)),
        "topk": k,
        "predicted_physical_spearman": _finite_spearman(
            predicted, physical
        ),
        "predicted_true_terminal_latent_spearman": _finite_spearman(
            predicted, latent
        ),
        "true_terminal_latent_physical_spearman": _finite_spearman(
            latent, physical
        ),
        "physical_topk_recall": float(
            len(predicted_top & physical_top) / k
        ),
        "successful_candidate_recall_at_k": (
            float(success[list(predicted_order[:k])].sum() / success_count)
            if success_count
            else 0.0
        ),
        "top1_success": bool(success[top1]),
        "oracle_success": bool(success_count),
        "candidate_success_rate": float(success.mean()),
        "physical_best_candidate_regret": float(
            physical[top1] - physical.min()
        ),
        "predicted_terminal_latent_mse": float(mse),
        "predicted_top1_index": top1,
        "physical_best_index": int(physical_order[0]),
    }


def _strictly_decreasing(values: Any) -> bool:
    return bool(values is not None and len(values) == 3 and values[0] > values[1] > values[2])


def _strictly_increasing(values: Any) -> bool:
    return bool(values is not None and len(values) == 3 and values[0] < values[1] < values[2])


def classify_epoch_pair_failure(evidence: dict[str, Any]) -> list[str]:
    """Apply the predeclared conservative attribution rules."""
    labels: list[str] = []
    successes = [bool(value) for value in evidence.get("candidate_successes", [])]
    if evidence.get("all_candidates_grounded") and successes and not any(successes):
        labels.append("coverage_failure")
    if successes and any(successes) and evidence.get("selected_success") is False:
        labels.append("ranking_failure")

    predicted_latent = evidence.get(
        "predicted_true_terminal_latent_spearman"
    )
    latent_physical = evidence.get(
        "true_terminal_latent_physical_spearman"
    )
    if predicted_latent is not None and latent_physical is not None:
        if predicted_latent < 0 < latent_physical:
            labels.append("dynamics_error")
        elif latent_physical < 0 < predicted_latent:
            labels.append("latent_metric_error")

    predicted_curve = evidence.get("predicted_elite_cost_0_5_29")
    physical_curve = evidence.get("physical_elite_cost_0_5_29")
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
    if len(labels) != 1:
        labels.append("mixed_or_undetermined")
    return labels


def _mean_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _summarize_slot_group(slots: list[dict[str, Any]]) -> dict[str, Any]:
    by_epoch: dict[str, dict[str, list[float]]] = {
        epoch: defaultdict(list) for epoch in ("e8", "e10")
    }
    paired_deltas: dict[str, list[float]] = defaultdict(list)
    attribution_counts = {epoch: Counter() for epoch in ("e8", "e10")}
    panel_count = 0
    for slot in slots:
        for epoch in ("e8", "e10"):
            attribution_counts[epoch].update(
                slot.get("attributions", {}).get(epoch, [])
            )
        for panel in slot.get("panels", []):
            models = panel.get("models", {})
            if not all(epoch in models for epoch in ("e8", "e10")):
                continue
            panel_count += 1
            for key in PAIR_METRIC_KEYS:
                values = []
                for epoch in ("e8", "e10"):
                    value = models[epoch].get("metrics", {}).get(key)
                    if value is not None:
                        numeric = float(value)
                        if np.isfinite(numeric):
                            by_epoch[epoch][key].append(numeric)
                            values.append(numeric)
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                if all(value is not None for value in values):
                    paired_deltas[key].append(values[1] - values[0])
    return {
        "slot_count": len(slots),
        "slots": [int(slot["slot"]) for slot in slots],
        "panel_count": panel_count,
        "metrics": {
            epoch: {
                key: _mean_or_none(by_epoch[epoch].get(key, []))
                for key in PAIR_METRIC_KEYS
            }
            for epoch in ("e8", "e10")
        },
        "e10_minus_e8": {
            key: _mean_or_none(paired_deltas.get(key, []))
            for key in PAIR_METRIC_KEYS
        },
        "attribution_counts": {
            epoch: dict(sorted(attribution_counts[epoch].items()))
            for epoch in ("e8", "e10")
        },
    }


def _decision_from_attributions(summary: dict[str, Any]) -> str:
    counts = Counter()
    for epoch_counts in summary["overall"]["attribution_counts"].values():
        counts.update(epoch_counts)
    coverage = counts["coverage_failure"]
    ranking_model = sum(
        counts[label]
        for label in (
            "ranking_failure",
            "dynamics_error",
            "latent_metric_error",
        )
    )
    search = counts["cem_exploitation"] + counts["replan_regression"]
    if coverage > max(ranking_model, search):
        return "change_proposal_or_actor_warm_start"
    if ranking_model > max(coverage, search):
        return "run_section_3_4_serial_one_step_control"
    delta = summary["overall"]["e10_minus_e8"]
    ranking_not_worse = all(
        delta.get(key) is None or delta[key] >= 0
        for key in (
            "predicted_physical_spearman",
            "physical_topk_recall",
        )
    )
    if search and ranking_not_worse and search >= max(coverage, ranking_model):
        return "change_cem_best_ever_early_stop_or_replan"
    return "expand_grounded_panel_before_new_training"


def grounded_pair_acceptance_errors(
    pair: EpochPair,
    grounded: dict[str, Any],
    *,
    detailed_iterations: Any = (0, 5, 29),
) -> list[str]:
    """Return every reason a grounded pair cannot claim formal completion."""
    errors: list[str] = []
    iterations = {int(value) for value in detailed_iterations}
    slots = list(grounded.get("slots", []))
    slot_ids = [int(slot.get("slot", -1)) for slot in slots]
    if len(slot_ids) != len(set(slot_ids)) or set(slot_ids) != set(pair.slots):
        errors.append("diagnostic slot set is incomplete or duplicated")
    for slot in slots:
        slot_id = int(slot.get("slot", -1))
        if slot_id not in pair.slots:
            continue
        if slot.get("category") != pair.category_for(slot_id):
            errors.append(f"slot {slot_id} category mismatch")
        final_success = slot.get("final_success", {})
        terminal = set(slot.get("terminal_before_replan_1", []))
        if not terminal <= {"e8", "e10"}:
            errors.append(f"slot {slot_id} has invalid terminal epoch labels")
        for epoch_label in terminal:
            if final_success.get(epoch_label) is not True:
                errors.append(
                    f"slot {slot_id}/{epoch_label} missing replan 1 without success"
                )
        expected_contexts = {
            ("shared", 0, iteration) for iteration in iterations
        }
        expected_contexts.update(
            (epoch_label, 1, iteration)
            for epoch_label in ("e8", "e10")
            if epoch_label not in terminal
            for iteration in iterations
        )
        panels = list(slot.get("panels", []))
        actual_contexts = [
            (
                str(panel.get("state_epoch")),
                int(panel.get("replan", -1)),
                int(panel.get("iteration", -1)),
            )
            for panel in panels
        ]
        if len(actual_contexts) != len(set(actual_contexts)):
            errors.append(f"slot {slot_id} has duplicate grounded panels")
        if set(actual_contexts) != expected_contexts:
            errors.append(f"slot {slot_id} grounded panel set is incomplete")
        for panel in panels:
            context = (
                f"slot {slot_id}/replan {panel.get('replan')}/"
                f"state {panel.get('state_epoch')}/iteration "
                f"{panel.get('iteration')}"
            )
            for key in (
                "planning_state_sha256",
                "prefix_sha256",
                "sampled_candidate_sha256",
                "candidate_sha256",
            ):
                value = panel.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    errors.append(f"{context} has invalid {key}")
            candidate_count = int(panel.get("candidate_count", 0))
            if (
                candidate_count <= 0
                or len(panel.get("candidate_successes", []))
                != candidate_count
            ):
                errors.append(f"{context} has inconsistent candidates")
            if not panel.get("cem_candidate_successes"):
                errors.append(f"{context} has no CEM candidates")
            models = panel.get("models", {})
            if set(models) != {"e8", "e10"}:
                errors.append(f"{context} is not cross-scored by both models")
                continue
            for epoch_label in ("e8", "e10"):
                metrics = models[epoch_label].get("metrics", {})
                for key in PAIR_METRIC_KEYS:
                    value = metrics.get(key)
                    if value is None:
                        errors.append(
                            f"{context}/"
                            f"{epoch_label} missing finite {key}"
                        )
                        continue
                    try:
                        finite = np.isfinite(float(value))
                    except (TypeError, ValueError):
                        finite = False
                    if not finite:
                        errors.append(f"{context}/{epoch_label} non-finite {key}")
    return errors

def summarize_grounded_pair(
    pair: EpochPair, grounded: dict[str, Any]
) -> dict[str, Any]:
    """Summarize paired e10-e8 deltas without cross-task physical averaging."""
    if grounded.get("status") not in {"ok", "incomplete"}:
        raise ValueError("grounded pair status must be ok or incomplete")
    slots = list(grounded.get("slots", []))
    acceptance_errors = list(grounded.get("acceptance_errors", []))
    acceptance_errors.extend(grounded_pair_acceptance_errors(pair, grounded))
    acceptance_errors = list(dict.fromkeys(acceptance_errors))
    by_category = {
        category: [
            slot for slot in slots if slot.get("category") == category
        ]
        for category in SLOT_CATEGORIES
    }
    summary = {
        "status": (
            "ok"
            if grounded.get("status") == "ok" and not acceptance_errors
            else "incomplete"
        ),
        "acceptance_errors": acceptance_errors,
        "pair": pair.label,
        "task": pair.task,
        "section": pair.section,
        "categories": {
            category: _summarize_slot_group(rows)
            for category, rows in by_category.items()
        },
        "overall": _summarize_slot_group(slots),
    }
    summary["decision"] = _decision_from_attributions(summary)
    return summary


__all__ = [
    "CandidatePanel",
    "EpochPair",
    "EpochPairManifest",
    "PairEpoch",
    "SLOT_CATEGORIES",
    "PAIR_METRIC_KEYS",
    "build_cross_epoch_panel",
    "build_full_cross_epoch_panel",
    "build_ground_identity",
    "build_pair_identity",
    "build_summary_identity",
    "candidate_actions_sha256",
    "cem_candidate_indices",
    "classify_epoch_pair_failure",
    "compute_ranking_metrics",
    "compute_pair_physical_terminal_cost",
    "cross_score_costs",
    "grounded_pair_acceptance_errors",
    "load_epoch_pair_manifest",
    "planning_state_sha256",
    "require_identity",
    "require_matching_cache_identity",
    "sha256_file",
    "summarize_grounded_pair",
    "tensor_bytes_sha256",
    "validate_diagnostic_device",
    "validate_pair_device",
    "write_atomic_json",
    "write_atomic_text",
    "write_atomic_torch",
    "write_pair_trace_artifacts",
]
