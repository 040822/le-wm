"""Plot Round 3 Phase 2 success-rate curves for one or more tasks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TASK_LABELS = {
    "cube": "Cube",
    "pusht": "Push-T",
    "reacher": "Reacher",
    "tworoom": "TwoRoom",
}
EXPERIMENT_LABELS = {"e0": "E0 LeWM", "e5": "Legacy E5 Fast-LeWAM"}
ARMS = ("offline_continue", "online_adapt")
ARM_COLORS = {"offline_continue": "#356AE6", "online_adapt": "#D1495B"}
ARM_LABELS = {"offline_continue": "Offline continue", "online_adapt": "Online adapt"}
STEPS = tuple(range(0, 20_001, 1_000))


def load_curve(task: str, experiment: str, arm: str) -> tuple[list[float], list[float]]:
    path = ROOT / "outputs/round3/phase2" / f"{task}_curve_200ep_1k" / experiment / arm / "curve.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(STEPS):
        raise ValueError(f"expected {len(STEPS)} curve points in {path}, got {len(rows)}")
    actual_steps = tuple(int(row["environment_steps"]) for row in rows)
    if actual_steps != STEPS:
        raise ValueError(f"unexpected steps in {path}: {actual_steps}")
    statuses = {row["status"] for row in rows}
    if statuses != {"ok"}:
        raise ValueError(f"curve is not complete in {path}: statuses={statuses}")
    steps = [step / 1000.0 for step in actual_steps]
    rates = [float(row["success_rate_percent"]) for row in rows]
    return steps, rates


def _task_ylim(task_rates: list[float]) -> tuple[float, float]:
    low = max(0.0, 5.0 * math.floor(min(task_rates) / 5.0) - 5.0)
    high = min(100.0, 5.0 * math.ceil(max(task_rates) / 5.0) + 5.0)
    if high - low < 20.0:
        midpoint = (high + low) / 2.0
        low = max(0.0, midpoint - 10.0)
        high = min(100.0, midpoint + 10.0)
    return low, high


def plot_tasks(tasks: list[str], output_stem: Path) -> None:
    for task in tasks:
        if task not in TASK_LABELS:
            raise ValueError(f"unsupported task: {task}")

    curve_data: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}
    for task in tasks:
        for experiment in ("e0", "e5"):
            for arm in ARMS:
                curve_data[(task, experiment, arm)] = load_curve(task, experiment, arm)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )
    figure, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(11.5, 3.6 * len(tasks)),
        squeeze=False,
        sharex=True,
    )
    for row_index, task in enumerate(tasks):
        all_task_rates: list[float] = []
        for experiment in ("e0", "e5"):
            for arm in ARMS:
                all_task_rates.extend(curve_data[(task, experiment, arm)][1])
        y_min, y_max = _task_ylim(all_task_rates)
        for column_index, experiment in enumerate(("e0", "e5")):
            axis = axes[row_index][column_index]
            for arm in ARMS:
                steps, rates = curve_data[(task, experiment, arm)]
                axis.plot(
                    steps,
                    rates,
                    marker="o",
                    markersize=3.0,
                    linewidth=1.8,
                    color=ARM_COLORS[arm],
                    label=ARM_LABELS[arm],
                )
            axis.set_title(f"{TASK_LABELS[task]} · {EXPERIMENT_LABELS[experiment]}")
            axis.set_xlabel("Training steps (k env steps)")
            axis.set_xticks([0, 5, 10, 15, 20])
            axis.set_xlim(-0.5, 20.5)
            axis.set_ylim(y_min, y_max)
            axis.set_yticks(list(range(int(y_min), int(y_max) + 1, 5)))
            axis.grid(True, alpha=0.28, linewidth=0.8)
            axis.axvline(5, color="#777777", linestyle="--", linewidth=0.8, alpha=0.55)
            axis.axvline(20, color="#777777", linestyle=":", linewidth=0.8, alpha=0.55)
            axis.legend(loc="best", frameon=True)
        axes[row_index][0].set_ylabel("Success rate (%)")

    figure.suptitle(
        "Round 3 Phase 2: 200-Episode Final-Cohort Success Curves",
        fontsize=16,
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Each point uses the frozen final cohort; 200 episodes evaluated as 4×50 batches.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    print(output_stem.with_suffix(".png"))
    print(output_stem.with_suffix(".svg"))
    print(output_stem.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["cube"])
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()
    if args.output_stem is None:
        if len(args.tasks) == 1:
            output_stem = ROOT / "outputs/round3/phase2" / f"{args.tasks[0]}_curve_200ep_1k" / "round3_phase2_success_curves"
        else:
            output_stem = ROOT / "outputs/round3/phase2/round3_phase2_remaining_tasks_success_curves"
    else:
        output_stem = Path(args.output_stem)
    plot_tasks(list(args.tasks), output_stem)


if __name__ == "__main__":
    main()
