#!/usr/bin/env python3
"""Compare normalized action-bound projections on the fixed Reacher dev cohort.

The script reuses the frozen Round 4 evaluator for new P1/P3 variants and
loads existing P0/P1/P2/P3 step-16 results as baselines.  It deliberately
does not retrain a checkpoint or run the final cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_phase1 import CohortManifest, paired_comparison, sha256_file
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility


CONFIG_PATH = ROOT / "config" / "round4" / "reacher_action_bounds_compare.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "round4" / "reacher_action_bounds_compare_seed3072_dev"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _validate_gpu_argument(value: str | None) -> str:
    if value is None:
        raise ValueError("CUDA comparison requires --gpu with one physical GPU0-3")
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) not in range(4):
        raise ValueError(
            f"GPU visibility must select exactly one physical GPU0-3; got {value!r}"
        )
    return values[0]


def _configure_device(device: str, gpu: str | None) -> None:
    if str(device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = _validate_gpu_argument(gpu)
        validate_gpu_visibility(device)


def _evaluation_config(manifest: CohortManifest, device: str):
    cfg = compose_eval_config(
        "reacher",
        overrides=[
            f"eval.num_eval={len(manifest.entries)}",
            "output.save_video=false",
            f"solver.device={device}",
        ],
    )
    cfg.solver.device = device
    return cfg


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _validate_manifest(manifest: CohortManifest) -> None:
    if manifest.task != "reacher" or manifest.cohort_kind != "dev":
        raise ValueError("the action-bound comparison requires the Reacher dev cohort")
    if len(manifest.entries) != 50:
        raise ValueError(f"expected 50 dev entries, got {len(manifest.entries)}")
    if manifest.protocol_variant != "round3_revised":
        raise ValueError(
            "the action-bound comparison requires the round3_revised cohort"
        )
    if int(manifest.seed) != 42 or int(manifest.goal_offset_steps) != 25:
        raise ValueError("the action-bound comparison requires seed42 and goal offset25")


def _validate_result(
    payload: Mapping[str, Any],
    *,
    name: str,
    spec: Mapping[str, Any],
    checkpoint: Path,
    manifest: CohortManifest,
) -> None:
    if payload.get("task") != "reacher":
        raise ValueError(f"{name} is not a Reacher result")
    if payload.get("cohort_kind") != "dev":
        raise ValueError(f"{name} is not a dev result")
    if payload.get("round4_mode", payload.get("stage")) != spec["mode"]:
        raise ValueError(f"{name} has an unexpected mode")
    parameters = payload.get("parameters", {})
    if parameters.get("cohort_id") != manifest.cohort_id:
        raise ValueError(f"{name} uses a different cohort id")
    if parameters.get("cohort_sha256") != manifest.computed_sha256:
        raise ValueError(f"{name} uses a different cohort checksum")
    recorded_checkpoint = payload.get("checkpoint")
    if recorded_checkpoint is not None and Path(recorded_checkpoint).resolve() != checkpoint:
        raise ValueError(f"{name} uses a different checkpoint")
    if int(parameters.get("seed", -1)) != 42:
        raise ValueError(f"{name} uses a different evaluation seed")
    if int(parameters.get("action_block", -1)) != 5:
        raise ValueError(f"{name} uses a different action block")
    if int(parameters.get("horizon", -1)) != 5:
        raise ValueError(f"{name} uses a different planning horizon")
    planning = payload.get("round4_planning", {})
    if spec["mode"] in {"P0", "P3"} and int(
        planning.get("action_flow_steps", -1)
    ) != 16:
        raise ValueError(f"{name} is not an Euler-16 result")
    if spec["mode"] == "P2" and int(
        planning.get("actor_flow_steps", planning.get("action_flow_steps", -1))
    ) != 16:
        raise ValueError(f"{name} is not a P2 Euler-16 result")
    if spec["mode"] == "P1":
        for field, expected in (
            ("candidate_count", 300),
            ("cem_iterations", 30),
            ("cem_topk", 30),
        ):
            if int(planning.get(field, -1)) != expected:
                raise ValueError(f"{name} does not use the frozen P1 CEM settings")


def _physical_action_stats(episodes: list[Mapping[str, Any]]) -> dict[str, float | int]:
    total = 0
    violations = 0
    max_abs = 0.0
    for episode in episodes:
        actions = [step.get("action") for step in episode.get("steps", [])]
        actions = [item for item in actions if item is not None]
        if not actions:
            continue
        array = np.asarray(actions, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if not np.all(np.isfinite(array)):
            raise ValueError("trace contains a non-finite physical action")
        mask = (array < -1.0 - 1e-6) | (array > 1.0 + 1e-6)
        total += int(mask.size)
        violations += int(mask.sum())
        max_abs = max(max_abs, float(np.max(np.abs(array))))
    return {
        "physical_action_elements": total,
        "physical_action_violation_elements": violations,
        "physical_action_violation_fraction": (
            float(violations / total) if total else 0.0
        ),
        "physical_action_max_abs": max_abs,
    }


def _projection_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    planning = payload.get("round4_planning", {})
    events = planning.get("action_bound_projection", [])
    if isinstance(events, Mapping):
        events = [events]
    if not events:
        return {}
    warm_events = [
        item["warm_start"]
        for item in events
        if isinstance(item.get("warm_start"), Mapping)
    ]
    projection_events = warm_events or list(events)
    before = [item["before"] for item in projection_events if item.get("before")]
    after = [item["after"] for item in projection_events if item.get("after")]

    def mean(field: str, rows: list[Mapping[str, Any]]) -> float | None:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return float(np.mean(values)) if values else None

    return {
        "projection_events": len(events),
        "projection_changed_fraction": mean(
            "changed_fraction", projection_events
        ),
        "projection_mean_abs_delta": mean(
            "mean_abs_delta", projection_events
        ),
        "projection_max_abs_delta": max(
            [float(item.get("max_abs_delta", 0.0)) for item in projection_events],
            default=0.0,
        ),
        "legacy_unit_threshold_before_fraction": mean(
            "legacy_unit_threshold_fraction", before
        ),
        "true_normalized_bound_before_fraction": mean(
            "true_normalized_bound_violation_fraction", before
        ),
        "true_normalized_bound_after_fraction": mean(
            "true_normalized_bound_violation_fraction", after
        ),
        "raw_candidate_violation_fraction": mean(
            "raw_candidate_violation_fraction", list(events)
        ),
        "projected_candidate_violation_fraction": mean(
            "projected_candidate_violation_fraction", list(events)
        ),
        "projection_input_violation_fraction": (
            mean("raw_candidate_violation_fraction", list(events))
            if any(
                item.get("raw_candidate_violation_fraction") is not None
                for item in events
            )
            else mean(
                "true_normalized_bound_violation_fraction", before
            )
        ),
        "candidate_changed_fraction": mean(
            "candidate_changed_fraction", list(events)
        ),
    }


def _result_row(name: str, spec: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    planning = payload.get("round4_planning", {})
    physical = _physical_action_stats(list(payload.get("episodes", [])))
    row = {
        "variant": name,
        "method": spec["mode"],
        "action_bound_mode": spec["action_bound_mode"],
        "success_rate": float(payload.get("success_rate", 0.0)),
        "success_rate_percent": 100.0 * float(payload.get("success_rate", 0.0)),
        "evaluation_seconds": payload.get("evaluation_seconds"),
        "planning_median_seconds": planning.get("planning_median_seconds"),
        "planning_p95_seconds": planning.get("planning_p95_seconds"),
        "forward_count": planning.get("forward_count"),
        "peak_memory_bytes": planning.get("peak_memory_bytes"),
        **physical,
        **_projection_stats(payload),
    }
    for field in (
        "true_normalized_bound_before_fraction",
        "true_normalized_bound_after_fraction",
        "legacy_unit_threshold_before_fraction",
        "projection_changed_fraction",
        "projection_mean_abs_delta",
        "projection_max_abs_delta",
        "projection_input_violation_fraction",
        "candidate_changed_fraction",
        "raw_candidate_violation_fraction",
        "projected_candidate_violation_fraction",
    ):
        row.setdefault(field, None)
    return row


def _render_report(
    *,
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    paired: Mapping[str, Any],
    bounds: Mapping[str, Any] | None,
    output_root: Path,
) -> str:
    lines = [
        "# Reacher 动作边界快速对照报告",
        "",
        "## 实验说明",
        "",
        "本报告使用 R4-AB epoch10、训练 seed3072、`round3_revised` Reacher dev cohort 的 50 个 episode。没有重新训练，也没有评测 final cohort。现有 z-score action normalizer 保持不变；新增操作只是在归一化动作空间内进行边界投影。",
        "",
        "`abs(action_norm)>1` 仅作为历史 proxy 记录。真实归一化边界由 Reacher 物理 action space 和现有 `StandardScaler` 计算，不能把 proxy 直接称为环境越界。",
        "",
        "## 归一化边界",
        "",
    ]
    if bounds:
        lines.extend(
            [
                f"- physical low/high: `{bounds.get('physical_low')}` / `{bounds.get('physical_high')}`",
                f"- normalized low/high: `{bounds.get('normalized_low')}` / `{bounds.get('normalized_high')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Dev 结果",
            "",
            "| variant | 方法 | 边界处理 | 成功率 | 物理动作越界率 | 投影输入归一化真实越界率 | 原始候选归一化越界率 | 候选修改率 | planning median(s) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {method} | {action_bound_mode} | {success_rate_percent:.1f}% | {physical_action_violation_fraction:.4f} | {projection_input_violation_fraction} | {raw_candidate_violation_fraction} | {candidate_changed_fraction} | {planning_median_seconds} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- P0、P1、P2、P3 均以各自的 `*_none` 作为 paired baseline；P1 的 candidate-clip 只约束 CEM candidate，P3 的 clip/global-scale 约束 A 生成的 64 条候选后再交给 B verifier。",
            "- CEM candidate 约束同时改变候选、B 评分输入和 elite 更新轨迹；P3 的候选约束则改变 B 评分输入和最终 argmin。因此成功率变化不能简单归因于执行阶段避免越界。",
            "- 归一化真实边界约为 `[-1.73, 1.73]`，而不是 `[-1, 1]`；本报告同时记录真实物理动作越界和归一化边界越界。",
            "",
            "## Episode-level paired comparison",
            "",
        ]
    )
    if paired:
        lines.extend(
            [
                "| comparison | improved | regressed | net success delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, result in paired.items():
            net_delta = result.get(
                "net_success_delta",
                int(result.get("improved", 0)) - int(result.get("regressed", 0)),
            )
            lines.append(
                f"| {name} | {result.get('improved')} | {result.get('regressed')} | {net_delta} |"
            )
    else:
        lines.append("没有可用的 paired comparison。")
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- P0 比较直接动作路径的 `clip` 和 `global_scale`；P1 比较随机初始化 CEM 的 candidate-clip；P2 保留 warm-start、CEM candidate 和 global-scale 三类处理；P3 比较 A best-of-64 候选的 `clip` 和 `global_scale`。",
            "- 若真实物理越界下降且成功率提高，动作边界处理可作为后续规划候选，但不覆盖默认 evaluator。",
            "- 若真实越界接近零而成功率不变，则此前主要信号来自 `abs>1` proxy，不能据此断言环境动作越界是根因。",
            "",
            f"输出目录：`{output_root}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_or_run_variant(
    *,
    name: str,
    spec: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    cfg: Any,
    device: str,
    output_root: Path,
) -> dict[str, Any]:
    reuse = spec.get("reuse")
    if reuse:
        path = _resolve(reuse)
        payload = _load_json(path)
        _validate_result(
            payload,
            name=name,
            spec=spec,
            checkpoint=checkpoint,
            manifest=manifest,
        )
        return payload

    target = output_root / name
    result_path = target / "result.json"
    if result_path.exists():
        payload = _load_json(result_path)
        _validate_result(
            payload,
            name=name,
            spec=spec,
            checkpoint=checkpoint,
            manifest=manifest,
        )
        return payload

    policy_or_model, resolved_checkpoint = load_policy_or_model(str(checkpoint))
    identity = EvaluationIdentity(
        entrypoint="round4_reacher_action_bounds",
        policy_kind="round4_shared_dit_action_bounds",
        checkpoint=str(resolved_checkpoint or checkpoint),
        epoch=10,
        stage=spec["mode"],
    )
    payload = run_round4_evaluation(
        cfg,
        task="reacher",
        policy_or_model=policy_or_model,
        mode=spec["mode"],
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=device,
        trace=True,
        candidate_count=64,
        flow_steps=16,
        action_flow_steps=16,
        solver_batch_size=1,
        actor_warm_start_scale=1.0,
        action_flow_integrator="euler",
        action_bound_mode=spec["action_bound_mode"],
    )
    _validate_result(
        payload,
        name=name,
        spec=spec,
        checkpoint=checkpoint,
        manifest=manifest,
    )
    return payload


def _write_tables(
    output_root: Path,
    rows: list[Mapping[str, Any]],
    paired: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    _write_json(output_root / "summary.json", {**metadata, "rows": rows, "paired": paired})
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field)) for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", help="one physical GPU0-3; required for CUDA")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    output_root = _resolve(args.output_root)
    manifest = CohortManifest.load(_resolve(config["cohort"]["source"]))
    _validate_manifest(manifest)
    checkpoint = _resolve(config["training"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint = checkpoint.resolve()
    variants = config["variants"]
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "variants": variants}, indent=2))
        return

    if not args.analyze_only:
        _configure_device(args.device, args.gpu)
        cfg = _evaluation_config(manifest, args.device)
    else:
        cfg = None

    payloads: dict[str, dict[str, Any]] = {}
    for name, spec in variants.items():
        if args.analyze_only and not spec.get("reuse"):
            path = output_root / name / "result.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing result for analyze-only variant {name}: {path}"
                )
            payload = _load_json(path)
            _validate_result(
                payload,
                name=name,
                spec=spec,
                checkpoint=checkpoint,
                manifest=manifest,
            )
        else:
            payload = _load_or_run_variant(
                name=name,
                spec=spec,
                manifest=manifest,
                checkpoint=checkpoint,
                cfg=cfg,
                device=args.device,
                output_root=output_root,
            )
        payloads[name] = payload

    rows = [
        _result_row(name, variants[name], payloads[name])
        for name in variants
    ]
    paired: dict[str, Any] = {}
    baseline_by_method = {
        "P0": "P0_none",
        "P1": "P1_none",
        "P2": "P2_none",
        "P3": "P3_none",
    }
    for name, spec in variants.items():
        baseline_name = baseline_by_method[spec["mode"]]
        if name == baseline_name:
            continue
        comparison = paired_comparison(
            payloads[baseline_name].get("episodes", []),
            payloads[name].get("episodes", []),
        )
        comparison["net_success_delta"] = int(
            comparison["improved"] - comparison["regressed"]
        )
        paired[f"{name}_vs_{baseline_name}"] = comparison

    bounds = None
    for payload in payloads.values():
        candidate_bounds = payload.get("round4_planning", {}).get("action_bounds")
        if candidate_bounds:
            bounds = candidate_bounds
            break
    metadata = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "protocol": config["protocol"],
        "code_commit": _git_commit(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "cohort_kind": manifest.cohort_kind,
        "bounds": bounds,
        "action_bound_semantics": "physical_env_bounds_mapped_through_existing_standard_scaler",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_tables(output_root, rows, paired, metadata)
    (output_root / "report.md").write_text(
        _render_report(
            config=config,
            rows=rows,
            paired=paired,
            bounds=bounds,
            output_root=output_root,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary": str(output_root / "summary.json"),
                "report": str(output_root / "report.md"),
                "variants": len(payloads),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
