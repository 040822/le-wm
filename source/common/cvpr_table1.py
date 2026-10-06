"""Frozen Table 1 condition registry and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SEEDS = (42, 100, 2026, 3407, 1234, 4444)
TASKS = ("tworoom", "pusht", "reacher", "cube")
BOUND_MODES = ("main", "crop")
_BASE_METHODS = (
    ("P0", "P0", "none", None),
    ("P1", "P1", "none", None),
    ("P2", "P2", "none", None),
    ("P3", "P3", "none", None),
)
_GUIDED_METHODS = (
    ("P0-GF", "P0", "guided_flow", "gf"),
    ("P0-PO", "P0", "post_opt", "po"),
    ("P2-GF", "P2", "guided_flow", "gf"),
    ("P2-PO", "P2", "post_opt", "po"),
    ("P3-GF", "P3", "guided_flow", "gf"),
    ("P3-PO", "P3", "post_opt", "po"),
    ("P3-PO-refine", "P3", "post_opt_refine", "po"),
)
_BUDGETS = {"L": 2, "H": 5}


def cowm_methods() -> list[dict[str, Any]]:
    """Return the 18 frozen CoWM protocols, each expanded by action handling."""
    methods = []
    for name, mode, guidance, _kind in _BASE_METHODS:
        methods.append(
            {
                "id": name,
                "label": name,
                "mode": mode,
                "guidance_mode": guidance,
                "budget": None,
                "action_flow_steps": 2 if mode != "P1" else None,
            }
        )
    for name, mode, guidance, kind in _GUIDED_METHODS:
        for budget, updates in _BUDGETS.items():
            inner_steps = (
                (1 if budget == "L" else 5)
                if kind == "gf"
                else updates
            )
            methods.append(
                {
                    "id": f"{name}-{budget}",
                    "label": f"{name}-{budget}",
                    "mode": mode,
                    "guidance_mode": guidance,
                    "budget": budget,
                    "guidance_inner_steps": inner_steps,
                    "guidance_last_steps": 2 if kind == "gf" else None,
                    "guidance_inner_steps_per_euler_step": (
                        1 if budget == "L" else 5
                    ) if kind == "gf" else None,
                    "action_flow_steps": 2,
                }
            )
    if len(methods) != 18 or len({item["id"] for item in methods}) != 18:
        raise RuntimeError("the frozen CoWM method registry must contain 18 unique methods")
    return methods


def all_methods() -> list[dict[str, Any]]:
    """Return all methods/configurations, including the second-stage baseline."""
    rows: list[dict[str, Any]] = []
    for bound_mode in BOUND_MODES:
        for method in cowm_methods():
            rows.append(
                {
                    **method,
                    "method_id": f"cowm_{method['id'].lower().replace('-', '_')}__{bound_mode}",
                    "family": "cowm",
                    "action_bound_mode": _bound_mode(method["mode"], bound_mode),
                    "cem_protocol": _cem_protocol(method["mode"], bound_mode),
                    "stage": 1,
                }
            )
    rows.extend(
        [
            {
                "method_id": "lewm__main",
                "id": "LeWM",
                "label": "LeWM",
                "family": "lewm",
                "action_bound_mode": "none",
                "cem_protocol": "legacy",
                "stage": 1,
            },
            {
                "method_id": "leflow__main",
                "id": "LeFlow",
                "label": "LeFlow",
                "family": "leflow",
                "action_bound_mode": "none",
                "cem_protocol": "not_applicable",
                "stage": 1,
            },
            {
                "method_id": "subjepa__main",
                "id": "Sub-JEPA",
                "label": "Sub-JEPA",
                "family": "subjepa",
                "action_bound_mode": "none",
                "cem_protocol": "legacy",
                "stage": 2,
            },
        ]
    )
    return rows


def _bound_mode(mode: str, variant: str) -> str:
    if variant == "main":
        return "none"
    if mode in {"P1", "P2"}:
        return "candidate_clip"
    return "clip"


def _cem_protocol(mode: str, variant: str) -> str:
    if mode not in {"P1", "P2"}:
        return "not_applicable"
    return "legacy" if variant == "main" else "cem-clip"


def evaluation_cells() -> list[dict[str, Any]]:
    """Enumerate independent evaluations; aliases are represented separately."""
    cells = []
    for method in all_methods():
        for task in TASKS:
            if method["family"] == "subjepa":
                for seed in SEEDS:
                    cells.append(_cell(method, task, seed))
                continue
            if method["family"] == "cowm" and method["id"].startswith("P3"):
                pass
            for seed in SEEDS:
                cells.append(_cell(method, task, seed))

    # Selection† is a Reacher-only 10/10 variant. Other tasks are strict aliases
    # of the corresponding P3 condition and never add evaluation work.
    for bound in BOUND_MODES:
        base_method = next(
            row for row in all_methods()
            if row["family"] == "cowm" and row["method_id"] == f"cowm_p3__{bound}"
        )
        dagger = {
            **base_method,
            "method_id": f"cowm_selection_dagger__{bound}",
            "id": "CoWM-Selection†",
            "label": "CoWM-Selection†",
            "execute_steps": 10,
            "score_horizon_blocks": 2,
            "short_horizon": True,
        }
        for seed in SEEDS:
            cells.append(_cell(dagger, "reacher", seed))
    expected = {"core": 912, "dagger": 12, "subjepa": 24}
    counts = {
        "core": sum(
            cell["stage"] == 1
            and cell["family"] != "subjepa"
            and not cell.get("short_horizon", False)
            for cell in cells
        ),
        "dagger": sum(cell.get("short_horizon", False) for cell in cells),
        "subjepa": sum(cell["family"] == "subjepa" for cell in cells),
    }
    if counts != expected:
        raise RuntimeError(f"frozen Table 1 matrix mismatch: expected {expected}, got {counts}")
    return cells


def timing_conditions() -> list[dict[str, Any]]:
    """Return the complete batch=1 timing matrix, including Sub-JEPA and dagger."""
    conditions = [
        _cell(method, task, 42)
        for task in TASKS
        for method in all_methods()
    ]
    for bound in BOUND_MODES:
        base_method = next(
            row
            for row in all_methods()
            if row["family"] == "cowm" and row["method_id"] == f"cowm_p3__{bound}"
        )
        dagger = {
            **base_method,
            "method_id": f"cowm_selection_dagger__{bound}",
            "id": "CoWM-Selection†",
            "label": "CoWM-Selection†",
            "execute_steps": 10,
            "score_horizon_blocks": 2,
            "short_horizon": True,
        }
        conditions.append(_cell(dagger, "reacher", 42))
    if len(conditions) != 158:
        raise RuntimeError(f"expected 158 timing conditions; got {len(conditions)}")
    return conditions


def _cell(method: Mapping[str, Any], task: str, seed: int) -> dict[str, Any]:
    record = dict(method)
    record.update(task=task, evaluation_seed=int(seed))
    record["environment_seed"] = int(seed) + 10000
    record["policy_seed"] = int(seed) + 20000
    record["cell_id"] = f"{record['method_id']}__{task}__seed_{seed}"
    return record


def identity_hash(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CEMArchiveCallback:
    """Capture the final CEM candidate scores and optimizer trace per env slot."""

    output_key = "cvpr_table1_archive"

    def __init__(self, *, iterations: int, topk: int):
        self.iterations = int(iterations)
        self.topk = int(topk)
        self.events: list[dict[str, Any]] = []
        self._current = None

    def reset(self):
        # Keep prior policy-call events; CEM invokes reset at every solve.
        self._current = None

    @property
    def history(self):
        return self.events

    def end_solve(self):
        return None

    def start_batch(self):
        self._current = {
            "iteration_elite_mean_costs": [],
            "optimizer_iterations": self.iterations,
            "topk": self.topk,
        }
        self.events.append(self._current)

    def __call__(
        self,
        *,
        step,
        candidates,
        costs,
        topk_vals,
        topk_inds,
        topk_candidates,
        mean,
        var,
        prev_mean,
        prev_var,
    ):
        del topk_candidates, prev_mean, prev_var
        if self._current is None:
            return
        self._current["iteration_elite_mean_costs"].append(
            topk_vals.detach().mean(dim=1).cpu().tolist()
        )
        if int(step) != self.iterations - 1:
            return
        best = costs.argmin(dim=1)
        self._current.update(
            {
                "candidate_count": int(candidates.shape[1]),
                "final_candidate_costs": costs.detach().cpu().tolist(),
                "selected_candidate_indices": best.detach().cpu().tolist(),
                "final_elite_indices": topk_inds.detach().cpu().tolist(),
                "selected_action_sequences": mean.detach().cpu().tolist(),
                "final_variance": var.detach().cpu().tolist(),
            }
        )


def attach_cem_archive_callback(policy: Any, *, iterations: int = 30, topk: int = 30):
    """Attach score capture to a standard or projected CEM policy solver."""
    solver = getattr(policy, "solver", None)
    seen = set()
    while solver is not None and id(solver) not in seen:
        seen.add(id(solver))
        if hasattr(solver, "callbacks") and hasattr(solver, "_solver"):
            inner = solver._solver
            if hasattr(inner, "callbacks") and hasattr(inner, "model"):
                solver = inner
                break
        if hasattr(solver, "callbacks") and hasattr(solver, "model"):
            break
        solver = getattr(solver, "_solver", None)
    if solver is None or not hasattr(solver, "callbacks"):
        return None
    callback = CEMArchiveCallback(iterations=iterations, topk=topk)
    solver.callbacks.append(callback)
    policy.cvpr_cem_capture = callback
    return callback


class TimingCEMSolver:
    """Same CEM updates as the installed solver without diagnostic transfers."""

    def __init__(self, solver: Any, *, projection_mode: str | None = None, bounds=None):
        self._solver = solver
        self.projection_mode = projection_mode
        self.action_bounds = bounds
        self.last_selected_indices = []

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def configure(self, *, action_space, n_envs, config):
        self._solver.configure(
            action_space=action_space, n_envs=n_envs, config=config
        )

    def set_action_bounds(self, bounds):
        self.action_bounds = bounds

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        import torch

        with torch.inference_mode():
            return self._solve_inference(info_dict, init_action)

    def _solve_inference(self, info_dict, init_action=None):
        import torch
        from stable_worldmodel.solver.cem import prepare_init_action

        from .round4_action_bounds import project_normalized_actions

        base = self._solver
        total_envs = len(next(iter(info_dict.values())))
        prepared = prepare_init_action(
            base.model,
            info_dict,
            init_action,
            base.horizon,
            n_envs=total_envs,
            action_dim=base.action_dim,
        )
        if self.projection_mode is not None:
            if self.action_bounds is None:
                raise RuntimeError("timing CEM projection requires action bounds")
            prepared = project_normalized_actions(
                prepared, self.action_bounds, mode=self.projection_mode
            )
        mean, variance = base.init_action_distrib(total_envs, prepared)
        mean = mean.to(base.device)
        variance = variance.to(base.device)
        self.last_selected_indices = []
        with torch.inference_mode():
            for start in range(0, total_envs, base.batch_size):
                end = min(start + base.batch_size, total_envs)
                batch_size = end - start
                batch_mean = mean[start:end]
                batch_variance = variance[start:end]
                expanded_infos = {}
                for key, value in info_dict.items():
                    value_batch = value[start:end]
                    if torch.is_tensor(value_batch):
                        dtype = base.dtype if value_batch.is_floating_point() else None
                        value_batch = value_batch.to(
                            device=base.device, dtype=dtype
                        ).unsqueeze(1).expand(
                            batch_size,
                            base.num_samples,
                            *value_batch.shape[1:],
                        )
                    elif hasattr(value_batch, "ndim"):
                        import numpy as np

                        if isinstance(value_batch, np.ndarray):
                            value_batch = np.repeat(
                                value_batch[:, None, ...], base.num_samples, axis=1
                            )
                    expanded_infos[key] = value_batch

                selected_indices = None
                for _step in range(base.n_steps):
                    candidates = torch.randn(
                        batch_size,
                        base.num_samples,
                        base.horizon,
                        base.action_dim,
                        generator=base.torch_gen,
                        device=base.device,
                        dtype=base.dtype,
                    )
                    candidates = (
                        candidates * batch_variance.unsqueeze(1)
                        + batch_mean.unsqueeze(1)
                    )
                    candidates[:, 0] = batch_mean
                    if self.projection_mode is not None:
                        candidates = project_normalized_actions(
                            candidates,
                            self.action_bounds,
                            mode=self.projection_mode,
                        )
                        candidates[:, 0] = project_normalized_actions(
                            batch_mean,
                            self.action_bounds,
                            mode=self.projection_mode,
                        )
                    costs = base.model.get_cost(expanded_infos, candidates)
                    _, topk_indices = torch.topk(
                        costs, k=base.topk, dim=1, largest=False
                    )
                    row_indices = torch.arange(
                        batch_size, device=base.device
                    ).unsqueeze(1).expand(-1, base.topk)
                    elite_candidates = candidates[row_indices, topk_indices]
                    batch_mean = elite_candidates.mean(dim=1)
                    batch_variance = elite_candidates.std(dim=1)
                    selected_indices = costs.argmin(dim=1)
                mean[start:end] = batch_mean
                variance[start:end] = batch_variance
                self.last_selected_indices.append(selected_indices.detach())
        return {"actions": mean.detach().cpu()}


def install_timing_cem(policy: Any):
    """Replace a CEM chain with its timing-only implementation and suppress diagnostics."""
    current = getattr(policy, "solver", None)
    if current is None:
        return None
    projection_mode = None
    bounds = getattr(policy, "action_bounds", None)
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if projection_mode is None:
            projection_mode = getattr(current, "projection_mode", None)
        if bounds is None:
            bounds = getattr(current, "action_bounds", None)
        if hasattr(current, "callbacks") and hasattr(current, "num_samples"):
            base = current
            break
        current = getattr(current, "_solver", None)
    else:
        return None
    timing_solver = TimingCEMSolver(
        base,
        projection_mode=projection_mode,
        bounds=bounds,
    )
    policy.solver = timing_solver
    model_view = getattr(base, "model", None)
    if hasattr(model_view, "timing_mode"):
        model_view.timing_mode = True
    nested_model = getattr(model_view, "model", None)
    if hasattr(nested_model, "timing_mode"):
        nested_model.timing_mode = True
    if hasattr(policy, "timing_mode"):
        policy.timing_mode = True
    return timing_solver


def matrix_counts() -> dict[str, int]:
    cells = evaluation_cells()
    return {
        "independent_cells": len(cells),
        "stage1_cells": sum(cell["stage"] == 1 for cell in cells),
        "stage2_cells": sum(cell["stage"] == 2 for cell in cells),
        "episodes": len(cells) * 50,
        "coWM_methods_per_bound_mode": len(cowm_methods()),
    }


def prioritized_cells(*, task: str, phase: int | None = None) -> list[dict[str, Any]]:
    """Order work with Table 1 main rows first, followed by remaining tests."""
    priority = {
        "lewm__main": 0,
        "leflow__main": 1,
        "cowm_p3__main": 2,
        "cowm_p0_po_l__main": 3,
        "cowm_selection_dagger__main": 4,
        "cowm_p3__crop": 10,
        "cowm_selection_dagger__crop": 11,
        "subjepa__main": 100,
    }

    def key(cell):
        return (
            priority.get(cell["method_id"], 50),
            cell["method_id"],
            SEEDS.index(int(cell["evaluation_seed"])),
        )

    return sorted(
        [cell for cell in evaluation_cells() if cell["task"] == task and (phase is None or cell["stage"] == phase)],
        key=key,
    )


__all__ = [
    "BOUND_MODES",
    "CEMArchiveCallback",
    "SEEDS",
    "TASKS",
    "all_methods",
    "attach_cem_archive_callback",
    "cowm_methods",
    "evaluation_cells",
    "identity_hash",
    "install_timing_cem",
    "matrix_counts",
    "prioritized_cells",
    "timing_conditions",
]
