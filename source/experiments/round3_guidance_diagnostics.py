"""CPU-testable manifests and gradient diagnostics for Round 3 E1/E2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


GUIDANCE_METHODS = (
    "a",
    "a_plus_b",
    "post_opt",
    "guided_flow",
    "b_cem",
    "a_cem",
)


def make_signed_guidance_candidates(
    actor_action: torch.Tensor,
    cost_gradient: torch.Tensor,
    *,
    rms: float,
    seed: int = 42,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Build E1's actor/negative/positive/random paired panel.

    The cost gradient is normalized independently for this state.  The
    negative direction is the expected descent direction; the positive and
    random directions have the same RMS magnitude for a fair diagnostic.
    """
    actor = torch.as_tensor(actor_action).detach().float()
    gradient = torch.as_tensor(cost_gradient).detach().to(actor).float()
    if actor.ndim != 2 or gradient.shape != actor.shape:
        raise ValueError("actor_action and cost_gradient must both have shape [H,A]")
    if not torch.isfinite(actor).all() or not torch.isfinite(gradient).all():
        raise ValueError("actor_action and cost_gradient must be finite")
    magnitude = float(rms)
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("rms must be finite and positive")
    grad_rms = gradient.square().mean().sqrt()
    if float(grad_rms) <= torch.finfo(gradient.dtype).eps:
        raise ValueError("cost_gradient is zero and cannot define signed probes")
    generator = torch.Generator(device=actor.device).manual_seed(int(seed))
    random = torch.randn(actor.shape, device=actor.device, generator=generator)
    random = random / random.square().mean().sqrt().clamp_min(torch.finfo(random.dtype).eps)
    unit = gradient / grad_rms
    offsets = torch.stack((-unit, unit, random), dim=0) * magnitude
    panel = torch.cat((actor.unsqueeze(0), actor.unsqueeze(0) + offsets), dim=0)
    return panel, (
        "actor",
        "negative_latent_gradient",
        "positive_latent_gradient",
        "random_direction",
    )


@dataclass(frozen=True)
class GuidanceDiagnosticConfig:
    """Frozen E1/E2 knobs recorded before any simulator rollout."""

    seed: int = 42
    e1_states: int = 64
    e1_rms: tuple[float, ...] = (0.01, 0.03, 0.10)
    e1_action_blocks: int = 5
    e2_dev_episodes: int = 50
    flow_steps: int = 10
    guidance_last_steps: int = 5
    guidance_inner_steps: int = 5
    guidance_step_size: float = 0.01
    guidance_max_rms_offset: float = 0.20
    cem_samples: int = 300
    cem_topk: int = 30
    cem_iterations: int = 30
    cem_var_scale: float = 1.0

    def __post_init__(self) -> None:
        if int(self.e1_states) != 64 or int(self.e2_dev_episodes) != 50:
            raise ValueError("Round3 E1/E2 requires 64 states and a 50-episode dev cohort")
        if int(self.flow_steps) != 10 or int(self.guidance_last_steps) != 5:
            raise ValueError("Round3 E1/E2 flow settings are fixed at 10/5 steps")
        if int(self.guidance_inner_steps) != 5:
            raise ValueError("Round3 guidance uses exactly five inner updates")
        if int(self.e1_action_blocks) != 5:
            raise ValueError("Round3 E1 uses five complete action blocks")
        if tuple(float(value) for value in self.e1_rms) != (0.01, 0.03, 0.10):
            raise ValueError("Round3 E1 RMS settings are fixed at .01/.03/.10")


def e2_method_specs(config: GuidanceDiagnosticConfig | None = None) -> dict[str, dict[str, Any]]:
    """Describe the six E2 comparisons without selecting a winner."""
    config = config or GuidanceDiagnosticConfig()
    return {
        "a": {
            "stage": "stage_a",
            "actor_warm_start": False,
            "guidance_mode": "none",
            "solver_samples": None,
            "description": "single raw Stage-A candidate",
        },
        "a_plus_b": {
            "stage": "stage_b",
            "actor_warm_start": False,
            "guidance_mode": "none",
            "solver_samples": 64,
            "solver_topk": min(30, 64),
            "description": "64-candidate Stage-B selection",
        },
        "post_opt": {
            "stage": "stage_a",
            "actor_warm_start": False,
            "guidance_mode": "post_opt",
            "solver_samples": None,
            "description": "single Stage-A candidate plus five post-opt updates",
        },
        "guided_flow": {
            "stage": "stage_a",
            "actor_warm_start": False,
            "guidance_mode": "guided_flow",
            "solver_samples": None,
            "description": "single Stage-A candidate with late flow guidance",
        },
        "b_cem": {
            "stage": "stage_b",
            "actor_warm_start": False,
            "guidance_mode": "none",
            "solver_samples": int(config.cem_samples),
            "solver_topk": int(config.cem_topk),
            "description": "Stage-B CEM with the canonical 300/30 budget",
        },
        "a_cem": {
            "stage": "stage_b",
            "actor_warm_start": True,
            "guidance_mode": "none",
            "solver_samples": int(config.cem_samples),
            "solver_topk": int(config.cem_topk),
            "description": "Stage-A actor warm-start followed by canonical CEM",
        },
    }


def make_guidance_manifest(
    *,
    task: str,
    checkpoint: str | Path,
    output: str | Path,
    config: GuidanceDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Write an auditable E1/E2 manifest for one task."""
    if str(task) not in {"reacher", "pusht"}:
        raise ValueError("Round3 guidance diagnostics only support reacher and pusht")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = config or GuidanceDiagnosticConfig()
    payload = {
        "schema_version": 1,
        "phase": "round3_guidance_diagnostics",
        "task": str(task),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "config": asdict(config),
        "e1": {
            "states": int(config.e1_states),
            "methods": ("actor", "negative_latent_gradient", "positive_latent_gradient", "random_direction"),
            "exclude_from_training_and_evaluation": True,
        },
        "e2": e2_method_specs(config),
        "decision_policy": "report-only; user decides whether guidance enters fixed replay",
    }
    payload["content_sha256"] = _canonical_hash(payload)
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def finite_difference_guidance_check(
    model: Any,
    z0: torch.Tensor,
    goal_latent: torch.Tensor,
    actions: torch.Tensor,
    *,
    epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Check autograd direction against a central finite-difference probe."""
    if float(epsilon) <= 0.0 or not np.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    value = torch.as_tensor(actions).detach().clone().requires_grad_(True)
    cost = model._latent_cost_from_clean_actions(z0, goal_latent, value, None)
    gradient = torch.autograd.grad(cost.sum(), value)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("guidance gradient is non-finite")
    direction = gradient.detach()
    norm = direction.square().mean().sqrt().clamp_min(torch.finfo(direction.dtype).eps)
    direction = direction / norm
    plus = actions.detach() + float(epsilon) * direction
    minus = actions.detach() - float(epsilon) * direction
    with torch.no_grad():
        plus_cost = model._latent_cost_from_clean_actions(z0, goal_latent, plus, None).sum()
        minus_cost = model._latent_cost_from_clean_actions(z0, goal_latent, minus, None).sum()
    finite_directional_derivative = float((plus_cost - minus_cost).cpu() / (2.0 * float(epsilon)))
    autograd_directional_derivative = float((gradient * direction).sum().detach().cpu())
    # Moving in -gradient should lower cost; moving in +gradient should raise it.
    return {
        "epsilon": float(epsilon),
        "cost": float(cost.sum().detach().cpu()),
        "plus_cost": float(plus_cost.cpu()),
        "minus_cost": float(minus_cost.cpu()),
        "autograd_directional_derivative": autograd_directional_derivative,
        "finite_difference_directional_derivative": finite_directional_derivative,
        "sign_match": bool(
            np.sign(autograd_directional_derivative)
            == np.sign(finite_directional_derivative)
        ),
        "descent_probe_lower_cost": bool(minus_cost < plus_cost),
    }


def summarize_e1_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate E1 callback rows while retaining paired, report-only metrics."""
    if not rows:
        raise ValueError("E1 rows cannot be empty")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    summary = {}
    for method, values in grouped.items():
        improvements = [float(item["physical_cost_delta"]) < 0.0 for item in values]
        summary[method] = {
            "count": len(values),
            "paired_improvement_rate": float(np.mean(improvements)),
            "mean_physical_cost_delta": float(np.mean([item["physical_cost_delta"] for item in values])),
            "mean_latent_cost_delta": float(np.mean([item["latent_cost_delta"] for item in values])),
        }
    return {"status": "report_only", "methods": summary}


def run_e2_report(
    *,
    task: str,
    checkpoint: str | Path,
    output_dir: str | Path,
    evaluator: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    config: GuidanceDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Run injected E2 method evaluations and persist a no-gating report."""
    config = config or GuidanceDiagnosticConfig()
    specs = e2_method_specs(config)
    results = {
        method: dict(evaluator(method, spec))
        for method, spec in specs.items()
    }
    payload = {
        "schema_version": 1,
        "phase": "round3_e2_guidance",
        "task": str(task),
        "checkpoint": str(Path(checkpoint).resolve()),
        "config": asdict(config),
        "methods": results,
        "decision": "report_only_user_selects_collector",
    }
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("e2_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "GUIDANCE_METHODS",
    "GuidanceDiagnosticConfig",
    "e2_method_specs",
    "finite_difference_guidance_check",
    "make_signed_guidance_candidates",
    "make_guidance_manifest",
    "run_e2_report",
    "summarize_e1_rows",
]
