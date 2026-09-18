"""CPU verifier for the four E5 physical-time/type Phase 3 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.round3_phase1 import (
    CohortManifest,
    canonical_result_path,
    canonical_trace_path,
    sha256_file,
)


def _resolved(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _config_value(config: Any, name: str) -> Any:
    value = OmegaConf.select(config, name)
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def verify(config_path: Path, *, output: Path, strict: bool) -> dict[str, Any]:
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    task_results: dict[str, Any] = {}
    all_complete = True
    for task, entry in spec["tasks"].items():
        reasons: list[str] = []
        cohort_path = _resolved(entry["dev_cohort"])
        try:
            cohort = CohortManifest.load(cohort_path)
            if cohort.computed_sha256 != entry["dev_cohort_sha256"]:
                reasons.append("dev cohort hash differs from Phase 3 registration")
            if cohort.cohort_kind != "dev" or cohort.protocol_variant != "round3_revised":
                reasons.append("dev cohort kind/protocol is not frozen round3_revised")
        except Exception as exc:  # noqa: BLE001 - report missing/invalid artifacts
            cohort = None
            reasons.append(f"dev cohort unavailable: {type(exc).__name__}: {exc}")

        checkpoint_path = _resolved(entry["checkpoint"])
        checkpoint_hash = None
        if not checkpoint_path.is_file():
            reasons.append("epoch-10 checkpoint is missing")
        else:
            checkpoint_hash = sha256_file(checkpoint_path)

        run_dir = _resolved(entry["run_dir"])
        config_candidates = (run_dir / "config.yaml", run_dir / ".hydra" / "config.yaml")
        run_config = next((path for path in config_candidates if path.is_file()), None)
        observed: dict[str, Any] = {}
        if run_config is None:
            reasons.append("training config is missing")
        else:
            try:
                train_config = OmegaConf.load(run_config)
                for name in (
                    "seed",
                    "latent_head_dim",
                    "train_mode",
                    "stage_a_goal_injection",
                    "detach_clean_action",
                    "latent_action_mix_epochs",
                    "stage_b_timestep_mode",
                    "stage_b_dynamics",
                    "stage_b_attention_mode",
                    "serial_activation_checkpointing",
                    "inference_steps",
                    "token_encoding",
                ):
                    observed[name] = _config_value(train_config, name)
                expected = {
                    "seed": 3072,
                    "latent_head_dim": 192,
                    "train_mode": "stage_ab",
                    "stage_a_goal_injection": "token",
                    "detach_clean_action": False,
                    "stage_b_timestep_mode": "legacy",
                    "stage_b_dynamics": "parallel_prefix",
                    "stage_b_attention_mode": "strict_causal",
                    "serial_activation_checkpointing": False,
                    "inference_steps": 10,
                    "token_encoding": "physical_time_type",
                }
                for name, value in expected.items():
                    if observed.get(name) != value:
                        reasons.append(
                            f"config mismatch {name}: expected {value!r}, got {observed.get(name)!r}"
                        )
                if observed.get("latent_action_mix_epochs") is None or float(
                    observed["latent_action_mix_epochs"]
                ) != 10.0:
                    reasons.append(
                        "config does not preserve historical E5 latent_action_mix_epochs=10"
                    )
            except Exception as exc:  # noqa: BLE001 - report malformed config
                reasons.append(f"training config invalid: {type(exc).__name__}: {exc}")

        evaluation_records: dict[str, Any] = {}
        evaluation_spec = spec.get("evaluation", {})
        if bool(evaluation_spec.get("require_results", False)) and cohort is not None:
            result_root = _resolved(evaluation_spec["result_root"])
            for stage in evaluation_spec.get(
                "stages", [evaluation_spec.get("stage", "stage_b")]
            ):
                result_path = canonical_result_path(
                    result_root,
                    task=task,
                    method=str(evaluation_spec.get("method", "e5_fast")),
                    protocol_variant="round3_revised",
                    stage=str(stage),
                    cohort_kind="dev",
                    cohort_sha256=cohort.computed_sha256,
                )
                trace_path = canonical_trace_path(
                    result_root,
                    task=task,
                    method=str(evaluation_spec.get("method", "e5_fast")),
                    protocol_variant="round3_revised",
                    stage=str(stage),
                    cohort_kind="dev",
                    cohort_sha256=cohort.computed_sha256,
                )
                result_payload = None
                result_reasons: list[str] = []
                if not result_path.is_file():
                    result_reasons.append("result.json is missing")
                else:
                    try:
                        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                        if result_payload.get("status") != "ok":
                            result_reasons.append("result status is not ok")
                        if len(result_payload.get("episodes", [])) != 50:
                            result_reasons.append("result does not contain 50 dev episodes")
                        if result_payload.get("cohort_sha256") != cohort.computed_sha256:
                            result_reasons.append("result cohort hash differs from registered dev cohort")
                        result_checkpoint = result_payload.get("checkpoint")
                        if result_checkpoint is None or Path(result_checkpoint).resolve() != checkpoint_path.resolve():
                            result_reasons.append("result checkpoint path differs from registered epoch-10 checkpoint")
                    except Exception as exc:  # noqa: BLE001 - report malformed results
                        result_reasons.append(f"result.json invalid: {type(exc).__name__}: {exc}")
                if not trace_path.is_file() or trace_path.stat().st_size == 0:
                    result_reasons.append("episodes.jsonl trace is missing or empty")
                if result_reasons:
                    reasons.extend(f"{stage} evaluation: {reason}" for reason in result_reasons)
                evaluation_records[str(stage)] = {
                    "result": str(result_path),
                    "trace": str(trace_path),
                    "status": "ok" if not result_reasons else "pending",
                    "success_rate": None if result_payload is None else result_payload.get("success_rate"),
                    "reasons": result_reasons,
                }

        status = "ok" if not reasons else "pending"
        if status != "ok":
            all_complete = False
        task_results[task] = {
            "status": status,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "train_config": str(run_config) if run_config else None,
            "observed_config": observed,
            "dev_cohort": str(cohort_path),
            "dev_cohort_sha256": None if cohort is None else cohort.computed_sha256,
            "evaluations": evaluation_records,
            "reasons": reasons,
        }

    report = {
        "schema_version": 1,
        "phase": "round3_phase3",
        "protocol": "round3_revised",
        "representation": spec["representation"],
        "status": "ok" if all_complete else "pending_gpu_training",
        "tasks": task_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    if strict and not all_complete:
        raise SystemExit(1)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/round3/phase3_e5_physical_time_type.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/round3/phase3/phase3_e5_physical_time_type_registry.json",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    report = verify(_resolved(args.config), output=_resolved(args.output), strict=args.strict)
    print(json.dumps({"status": report["status"], "output": str(_resolved(args.output))}, sort_keys=True))


if __name__ == "__main__":
    main()
