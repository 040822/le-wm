"""Create the fixed-slot §3.4 parallel-vs-serial diagnostic manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "seed": 42,
    "num_eval": 50,
    "goal_offset_steps": 25,
    "eval_budget": 50,
    "horizon": 5,
    "receding_horizon": 5,
    "action_block": 5,
    "num_samples": 300,
    "n_steps": 30,
    "topk": 30,
    "detailed_iterations": [0, 5, 29],
    "fixed_non_elites": 10,
}
RUNS = {
    "reacher_s34": {
        "task": "reacher",
        "preferred_gpu": 3,
        "parallel": "outputs/fast_lewam/reacher/0809_e1_stage_b_only_seed3072_tmux",
        "parallel_reference": (
            "outputs/fast_lewam/reacher/0813_s34_parallel_replay_reference"
            "/eval/epoch_10/stage_b/result.json"
        ),
        "serial": "outputs/fast_lewam/reacher/0813_s34_e1_serial_one_step_192",
    },
    "pusht_s34": {
        "task": "pusht",
        "preferred_gpu": 3,
        "parallel": "outputs/fast_lewam/pusht/0802_e1_stage_b_only",
        "parallel_reference": (
            "outputs/fast_lewam/pusht/0814_s34_parallel_replay_reference"
            "/eval/epoch_10/stage_b/result.json"
        ),
        "serial": "outputs/fast_lewam/pusht/0813_s34_e1_serial_one_step_192",
    },
}


def _read_result(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise ValueError(f"reference result is not ok: {path}")
    if len(result.get("episodes", [])) != 50:
        raise ValueError(f"reference result does not contain 50 episodes: {path}")
    return result


def _slot_categories(parallel: dict, serial: dict) -> dict[str, list[int]]:
    categories = {
        "regression": [],
        "improvement_control": [],
        "stable_success": [],
        "stable_failure": [],
    }
    for key in ("episode_ids", "start_steps", "start_rows"):
        if parallel["parameters"][key] != serial["parameters"][key]:
            raise ValueError(f"parallel/serial cohorts differ at {key}")
    for slot, (parallel_row, serial_row) in enumerate(
        zip(parallel["episodes"], serial["episodes"], strict=True)
    ):
        before = bool(parallel_row["success"])
        after = bool(serial_row["success"])
        if before and not after:
            category = "regression"
        elif not before and after:
            category = "improvement_control"
        elif before:
            category = "stable_success"
        else:
            category = "stable_failure"
        categories[category].append(slot)
    return categories


def build_manifest(root: Path = ROOT) -> dict:
    pairs = {}
    for label, run in RUNS.items():
        parallel_root = root / run["parallel"]
        serial_root = root / run["serial"]
        parallel_result_path = root / run.get(
            "parallel_reference",
            str(Path(run["parallel"]) / "eval/epoch_10/stage_b/result.json"),
        )
        serial_result_path = serial_root / "eval/epoch_10/stage_b/result.json"
        parallel = _read_result(parallel_result_path)
        serial = _read_result(serial_result_path)
        pairs[label] = {
            "task": run["task"],
            "section": "3.4",
            "preferred_gpu": run["preferred_gpu"],
            "run_config": str(parallel_root.relative_to(root) / "config.yaml"),
            "epochs": {
                "e8": {
                    "epoch": 10,
                    "run_config": str(
                        parallel_root.relative_to(root) / "config.yaml"
                    ),
                    "checkpoint": str(
                        parallel_root.relative_to(root)
                        / "checkpoints/fast_lewam_weights_epoch_10.pt"
                    ),
                    "reference_result": str(parallel_result_path.relative_to(root)),
                },
                "e10": {
                    "epoch": 10,
                    "run_config": str(serial_root.relative_to(root) / "config.yaml"),
                    "checkpoint": str(
                        serial_root.relative_to(root)
                        / "checkpoints/fast_lewam_weights_epoch_10.pt"
                    ),
                    "reference_result": str(serial_result_path.relative_to(root)),
                },
            },
            "slots": _slot_categories(parallel, serial),
        }
    return {
        "version": 1,
        "repository_root": str(root.resolve()),
        "output_root": "outputs/diagnostics/fast_lewam_serial_one_step/0813",
        "protocol": dict(PROTOCOL),
        "pairs": pairs,
    }


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        yaml.safe_dump(payload, stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config/diagnostics/fast_lewam_serial_one_step.yaml",
    )
    args = parser.parse_args()
    payload = build_manifest(ROOT)
    write_manifest(args.output.resolve(), payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
