"""Validate all Round 3 Phase 2 task runs and curve aggregates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/round3/phase2"
TASKS = ("reacher", "pusht", "tworoom")
TRAINING_ROOTS = {
    ("reacher", "e0"): "reacher_e0_three_arms_100step_threads4_retry1",
    ("reacher", "e5"): "reacher_e5_legacy_three_arms_100step_threads4_retry1",
    ("pusht", "e0"): "pusht_e0_three_arms_100step_threads4_retry1",
    ("pusht", "e5"): "pusht_e5_legacy_three_arms_100step_threads4_retry1",
    ("tworoom", "e0"): "tworoom_e0_three_arms_100step_threads4_retry1",
    ("tworoom", "e5"): "tworoom_e5_legacy_three_arms_100step_threads4_retry1",
}


def main() -> int:
    for (task, experiment), root_name in TRAINING_ROOTS.items():
        payload = json.loads((OUTPUT / root_name / "run_state.json").read_text())
        assert payload["status"] == "ok"
        assert payload["current_environment_steps"] == 20_000
        print(f"training {task}/{experiment}: ok")

    aggregate_count = 0
    for task in TASKS:
        task_hashes: set[str] = set()
        for experiment in ("e0", "e5"):
            for arm in ("offline_continue", "online_adapt"):
                series = OUTPUT / f"{task}_curve_200ep_1k" / experiment / arm
                curve = json.loads((series / "curve.json").read_text())
                assert curve["status"] == "ok"
                assert len(curve["results"]) == 21
                assert [row["environment_steps"] for row in curve["results"]] == list(
                    range(0, 20_001, 1_000)
                )
                for row in curve["results"]:
                    result = json.loads(
                        (
                            series
                            / f"envsteps_{row['environment_steps']:06d}"
                            / "result.json"
                        ).read_text()
                    )
                    assert result["status"] == "ok"
                    assert result["parameters"]["num_eval"] == 200
                    assert len(result["episodes"]) == 200
                    task_hashes.add(result["cohort_sha256"])
                    aggregate_count += 1
                endpoint = curve["results"][-1]["success_rate_percent"]
                print(f"curve {task}/{experiment}/{arm}: ok endpoint={endpoint:.1f}%")
        assert len(task_hashes) == 1, (task, task_hashes)
        print(f"cohort {task}: {next(iter(task_hashes))}")

    print(f"VALIDATION_OK training_runs=6 curves=12 aggregate_results={aggregate_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
