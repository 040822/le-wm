"""Phase 3 task adapters and the frozen legacy cohort builder.

The benchmark data uses the same flat HDF5 layout as the existing LeWM
experiments, but the installed stable-worldmodel release predates two of the
goal setters/termination predicates used by the DeWM evaluation protocol.
This module keeps those differences explicit and local to Phase 3 jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .round3_phase1 import CohortEntry, CohortManifest
from .round3_protocol import PHASE3_TASKS


ROOT = Path(__file__).resolve().parents[2]
PHASE3_DATASETS: dict[str, Path] = {
    "scene": ROOT / "data" / "datasets" / "ogbench" / "scene.h5",
    "finger": ROOT / "data" / "datasets" / "dmcontrol" / "finger_turn_hard.h5",
    "humanoid": ROOT / "data" / "datasets" / "dmcontrol" / "humanoid_walk.h5",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def build_phase3_legacy_manifest(
    dataset: Any,
    *,
    task: str,
    seed: int = 42,
    goal_offset_steps: int = 25,
    num_eval: int = 50,
) -> CohortManifest:
    """Reproduce the upstream legacy sampling arithmetic for any Phase 3 task."""

    task = str(task).lower()
    if task not in PHASE3_TASKS:
        raise ValueError(f"unsupported Phase 3 task: {task!r}")
    columns = set(str(item) for item in dataset.column_names)
    episode_column = "episode_idx" if "episode_idx" in columns else "ep_idx"
    episodes = np.asarray(dataset.get_col_data(episode_column))
    steps = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    if episodes.ndim != 1 or steps.ndim != 1 or len(episodes) != len(steps):
        raise ValueError("Phase 3 dataset episode_idx/step_idx columns are invalid")

    unique_episodes, inverse = np.unique(episodes, return_inverse=True)
    max_steps = np.full(len(unique_episodes), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(max_steps, inverse, steps)
    per_row_limit = max_steps[inverse] - int(goal_offset_steps)
    valid_indices = np.flatnonzero(steps <= per_row_limit).astype(np.int64)
    candidate_count = len(valid_indices) - 1  # upstream global-last-row quirk
    if candidate_count < int(num_eval):
        raise ValueError(
            f"Phase 3 legacy cohort needs {num_eval} starts, only "
            f"{max(candidate_count, 0)} candidates are available"
        )
    positions = np.random.default_rng(int(seed)).choice(
        candidate_count, size=int(num_eval), replace=False
    )


def _phase3_initial_success_flags(
    dataset: Any,
    *,
    task: str,
    start_rows: np.ndarray,
    goal_rows: np.ndarray,
) -> np.ndarray:
    """Evaluate the Phase 3 runtime success predicate on reset states.

    Targets are taken from the same future goal rows that the evaluator uses.
    The returned values therefore answer the eligibility question before a
    policy takes its first action.
    """

    task = str(task).lower()
    start_rows = np.asarray(start_rows, dtype=np.int64)
    goal_rows = np.asarray(goal_rows, dtype=np.int64)
    if start_rows.ndim != 1 or goal_rows.shape != start_rows.shape:
        raise ValueError("Phase 3 initial-state rows must be matching one-dimensional arrays")

    if task == "scene":
        block = np.asarray(dataset.get_col_data("privileged_block_0_pos"))
        button_0 = np.asarray(dataset.get_col_data("privileged_button_0_state"))
        button_1 = np.asarray(dataset.get_col_data("privileged_button_1_state"))
        drawer = np.asarray(dataset.get_col_data("privileged_drawer_pos")).reshape(-1)
        window = np.asarray(dataset.get_col_data("privileged_window_pos")).reshape(-1)
        return (
            (np.linalg.norm(block[start_rows] - block[goal_rows], axis=-1) <= 0.04)
            & (button_0[start_rows] == button_0[goal_rows])
            & (button_1[start_rows] == button_1[goal_rows])
            & (np.abs(drawer[start_rows] - drawer[goal_rows]) <= 0.04)
            & (np.abs(window[start_rows] - window[goal_rows]) <= 0.04)
        )

    if task == "finger":
        tip = np.asarray(dataset.get_col_data("tip_position"))
        target = np.asarray(dataset.get_col_data("target_position"))
        # stable-worldmodel's turn_hard wrapper uses dm_control's hard target
        # radius (0.03 m); get_reward() returns one exactly when the tip lies
        # inside this target.  The target is intentionally read from the
        # future goal row, matching set_target_position() during evaluation.
        return np.linalg.norm(tip[start_rows] - target[goal_rows], axis=-1) <= 0.03

    if task == "humanoid":
        head_height = np.asarray(dataset.get_col_data("head_height")).reshape(-1)
        torso_upright = np.asarray(dataset.get_col_data("torso_upright")).reshape(-1)
        planar_speed = np.asarray(dataset.get_col_data("speed")).reshape(-1)
        return (
            (head_height[start_rows] >= 1.4)
            & (torso_upright[start_rows] >= 0.9)
            & (planar_speed[start_rows] >= 1.0)
        )

    raise ValueError(f"unsupported Phase 3 task: {task!r}")


def build_phase3_initial_failure_manifest(
    dataset: Any,
    *,
    task: str,
    reference_manifest: CohortManifest,
    seed: int = 42,
    selection_seed: int = 43,
    goal_offset_steps: int = 25,
    num_eval: int = 50,
) -> CohortManifest:
    """Freeze a disjoint cohort whose start states all fail the runtime predicate.

    Candidate start-goal pairs follow the Phase 3 legacy row-validity rule and
    goal offset.  The 50 episodes already used by the legacy cohort are
    excluded, eligible candidates are grouped by episode, and one start is
    sampled uniformly within each of 50 selected episodes.
    """

    task = str(task).lower()
    if task not in PHASE3_TASKS:
        raise ValueError(f"unsupported Phase 3 task: {task!r}")
    if reference_manifest.task != task:
        raise ValueError("reference Phase 3 cohort task does not match requested task")
    if reference_manifest.protocol_variant != "legacy":
        raise ValueError("initial-failure cohort requires a legacy reference cohort")
    if reference_manifest.goal_offset_steps != int(goal_offset_steps):
        raise ValueError("reference cohort uses a different goal offset")

    columns = set(str(item) for item in dataset.column_names)
    episode_column = "episode_idx" if "episode_idx" in columns else "ep_idx"
    episodes = np.asarray(dataset.get_col_data(episode_column))
    steps = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    if episodes.ndim != 1 or steps.ndim != 1 or len(episodes) != len(steps):
        raise ValueError("Phase 3 dataset episode_idx/step_idx columns are invalid")

    unique_episodes, inverse = np.unique(episodes, return_inverse=True)
    max_steps = np.full(len(unique_episodes), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(max_steps, inverse, steps)
    per_row_limit = max_steps[inverse] - int(goal_offset_steps)
    valid_indices = np.flatnonzero(steps <= per_row_limit).astype(np.int64)
    candidate_rows = valid_indices[: max(len(valid_indices) - 1, 0)]

    guessed_goal_rows = candidate_rows + int(goal_offset_steps)
    goal_rows_match = (
        (guessed_goal_rows < len(episodes))
        & (episodes[guessed_goal_rows] == episodes[candidate_rows])
        & (steps[guessed_goal_rows] == steps[candidate_rows] + int(goal_offset_steps))
    ) if len(candidate_rows) else np.zeros(0, dtype=bool)
    if not np.all(goal_rows_match):
        lookup = {
            (int(episode), int(step)): int(index)
            for index, (episode, step) in enumerate(zip(episodes, steps))
        }
        mapped = np.asarray(
            [
                lookup.get((int(episodes[row]), int(steps[row]) + int(goal_offset_steps)), -1)
                for row in candidate_rows
            ],
            dtype=np.int64,
        )
        has_goal = mapped >= 0
        candidate_rows = candidate_rows[has_goal]
        candidate_goal_rows = mapped[has_goal]
    else:
        candidate_goal_rows = guessed_goal_rows

    if len(candidate_rows) == 0:
        raise ValueError("no valid Phase 3 start-goal candidates are available")

    reference_rows = np.asarray(
        [entry.row_index for entry in reference_manifest.entries], dtype=np.int64
    )
    reference_goals = np.asarray(
        [entry.goal_row_index for entry in reference_manifest.entries], dtype=np.int64
    )
    reference_initial_successes = _phase3_initial_success_flags(
        dataset,
        task=task,
        start_rows=reference_rows,
        goal_rows=reference_goals,
    )
    reference_episode_ids = np.asarray(
        [_jsonable(entry.episode_id) for entry in reference_manifest.entries],
        dtype=episodes.dtype,
    )
    fresh_mask = ~np.isin(episodes[candidate_rows], reference_episode_ids)
    fresh_rows = candidate_rows[fresh_mask]
    fresh_goal_rows = candidate_goal_rows[fresh_mask]
    fresh_initial_successes = _phase3_initial_success_flags(
        dataset,
        task=task,
        start_rows=fresh_rows,
        goal_rows=fresh_goal_rows,
    )
    eligible_rows = fresh_rows[~fresh_initial_successes]
    eligible_goal_rows = fresh_goal_rows[~fresh_initial_successes]
    if len(eligible_rows) == 0:
        raise ValueError("no initially unsuccessful Phase 3 candidates remain")

    eligible_episodes = episodes[eligible_rows]
    episode_order = np.argsort(eligible_episodes, kind="stable")
    sorted_rows = eligible_rows[episode_order]
    sorted_goal_rows = eligible_goal_rows[episode_order]
    sorted_episodes = eligible_episodes[episode_order]
    group_starts = np.r_[
        0,
        np.flatnonzero(sorted_episodes[1:] != sorted_episodes[:-1]) + 1,
    ].astype(np.int64)
    group_ends = np.r_[group_starts[1:], len(sorted_rows)].astype(np.int64)
    if len(group_starts) < int(num_eval):
        raise ValueError(
            f"Phase 3 initial-failure cohort needs {num_eval} distinct episodes, "
            f"only {len(group_starts)} are eligible"
        )

    rng = np.random.default_rng(int(selection_seed))
    selected_groups = rng.choice(len(group_starts), size=int(num_eval), replace=False)
    entries = []
    selected_episode_ids = []
    for group in selected_groups:
        start = int(group_starts[int(group)])
        end = int(group_ends[int(group)])
        slot = int(rng.integers(start, end))
        row = int(sorted_rows[slot])
        goal_row = int(sorted_goal_rows[slot])
        episode_id = _jsonable(episodes[row])
        selected_episode_ids.append(episode_id)
        entries.append(
            CohortEntry(
                row_index=row,
                episode_id=episode_id,
                start_step=int(steps[row]),
                goal_row_index=goal_row,
                goal_step=int(steps[row]) + int(goal_offset_steps),
                start_distance=None,
                initially_successful=False,
                start_state=None,
                goal_state=None,
            )
        )

    diagnostics = {
        "candidate_count_legacy_validity_rule": int(len(candidate_rows)),
        "reference_cohort_id": reference_manifest.cohort_id,
        "reference_cohort_sha256": reference_manifest.computed_sha256,
        "reference_episode_count_excluded": int(len(reference_episode_ids)),
        "reference_initial_success_count": int(reference_initial_successes.sum()),
        "reference_initial_failure_count": int(len(reference_initial_successes) - reference_initial_successes.sum()),
        "fresh_candidate_count": int(len(fresh_rows)),
        "fresh_initial_success_count": int(fresh_initial_successes.sum()),
        "fresh_initial_failure_candidate_count": int(len(eligible_rows)),
        "fresh_initial_failure_episode_count": int(len(group_starts)),
        "selected_count": int(len(entries)),
        "selected_initial_success_count": int(sum(entry.initially_successful is True for entry in entries)),
        "unique_episode_sampling": True,
        "environment_success_source": "runtime_predicate_reconstructed_from_start_and_goal_rows",
    }
    return CohortManifest(
        task=task,
        cohort_id=f"{task}_initial_failure_{int(num_eval)}_phase3_v1",
        cohort_kind="dev",
        protocol_variant="sampling_revised",
        seed=int(seed),
        goal_offset_steps=int(goal_offset_steps),
        entries=tuple(entries),
        episode_split={"dev": tuple(selected_episode_ids)},
        sampling_rule={
            "implementation": "Phase 3 initial-state filter over legacy-valid dataset start-goal rows",
            "global_last_row_excluded": True,
            "episode_level_deduplication": True,
            "initial_success_exclusion": True,
            "initial_success_predicate": {
                "scene": "cube position L2 <= 0.04 m AND both button states match AND drawer/window position error <= 0.04 m",
                "finger": "||start tip_position - future goal target_position||_2 <= 0.03 m (turn_hard runtime reward >= 1)",
                "humanoid": "head_height >= 1.4 AND torso_upright >= 0.9 AND planar speed >= 1.0",
            }[task],
            "reference_episodes_excluded": True,
            "selection_seed": int(selection_seed),
            "sampling_unit": "episode, with one uniformly selected eligible start per episode",
        },
        diagnostics=diagnostics,
    )
    rows = np.sort(valid_indices[positions])

    lookup = {
        (int(episode), int(step)): int(index)
        for index, (episode, step) in enumerate(zip(episodes, steps))
    }
    entries = []
    selected_episodes = []
    for row in rows:
        episode_id = _jsonable(episodes[int(row)])
        start_step = int(steps[int(row)])
        goal_step = start_step + int(goal_offset_steps)
        goal_row = lookup.get((int(episodes[int(row)]), goal_step))
        if goal_row is None:
            raise ValueError(
                f"missing goal row for episode={episode_id}, step={goal_step}"
            )
        selected_episodes.append(episode_id)
        entries.append(
            CohortEntry(
                row_index=int(row),
                episode_id=episode_id,
                start_step=start_step,
                goal_row_index=int(goal_row),
                goal_step=goal_step,
                # Phase 3 success is evaluated by the environment.  These
                # fields intentionally remain empty rather than deriving DMC
                # success from the NaN dataset success column.
                start_distance=None,
                initially_successful=None,
                start_state=None,
                goal_state=None,
            )
        )

    diagnostics = {
        "candidate_count": int(candidate_count),
        "valid_row_count": int(len(valid_indices)),
        "global_last_row_excluded": int(valid_indices[-1]),
        "selected_initial_success_not_filtered": True,
        "environment_success_source": "runtime_termination",
    }
    return CohortManifest(
        task=task,
        cohort_id=f"{task}_legacy_{int(num_eval)}_phase3_v1",
        cohort_kind="dev" if int(num_eval) == 50 else "custom",
        protocol_variant="legacy",
        seed=int(seed),
        goal_offset_steps=int(goal_offset_steps),
        entries=tuple(entries),
        episode_split={"selected": tuple(selected_episodes)},
        sampling_rule={
            "implementation": "Phase 3 compatibility copy of source.common.eval.select_eval_cohort",
            "global_last_row_excluded": True,
            "episode_level_deduplication": False,
            "initial_success_exclusion": False,
        },
        diagnostics=diagnostics,
    )


def _scalar(value: Any) -> Any:
    value = np.asarray(value)
    if value.ndim == 0:
        value = value.item()
    elif value.size == 1:
        value = value.reshape(-1)[0].item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def set_phase3_scene_goal(
    self,
    target_block_pos,
    target_block_quat,
    target_button_0_state,
    target_button_1_state,
    target_drawer_pos,
    target_window_pos,
) -> None:
    """Restore Scene termination targets from the future physical goal row."""

    block_pos = np.asarray(target_block_pos, dtype=np.float64).reshape(-1)
    block_quat = np.asarray(target_block_quat, dtype=np.float64).reshape(-1)
    if block_pos.shape != (3,):
        raise ValueError("Scene goal block position must contain three values")
    if block_quat.shape != (4,):
        raise ValueError("Scene goal block quaternion must contain four values")

    self.set_cube_target_pos(0, block_pos, block_quat)
    self.set_target_button_state(0, int(_scalar(target_button_0_state)))
    self.set_target_button_state(1, int(_scalar(target_button_1_state)))
    self.set_target_drawer_pos(float(_scalar(target_drawer_pos)))
    self.set_target_window_pos(float(_scalar(target_window_pos)))


def patch_phase3_environments() -> None:
    """Install missing DeWM-compatible setters and termination predicates."""

    from stable_worldmodel.envs.dmcontrol.finger import FingerDMControlWrapper
    from stable_worldmodel.envs.dmcontrol.humanoid import HumanoidDMControlWrapper
    from stable_worldmodel.envs.ogbench.scene_env import SceneEnv
    from stable_worldmodel.world import world as world_module

    # The installed stable-worldmodel extractor ignores scalar string columns.
    # Scene stores ``privileged_target_task`` as one string per episode, so
    # restore that field into the goal state before applying the Phase 3 goal
    # callable.  Numeric columns continue through the upstream extractor.
    if not getattr(world_module, "_phase3_string_goal_patched", False):
        original_extract_init_goal = world_module._extract_init_goal

        def extract_init_goal_with_scene_task(
            dataset, episodes_idx, start_steps, goal_offset
        ):
            init_state, goal_state, dataset_videos = original_extract_init_goal(
                dataset, episodes_idx, start_steps, goal_offset
            )
            if (
                "privileged_target_task" in dataset.column_names
                and "goal_privileged_target_task" not in goal_state
            ):
                chunks = dataset.load_chunk(
                    np.asarray(episodes_idx),
                    np.asarray(start_steps),
                    np.asarray(start_steps) + int(goal_offset) + 1,
                )
                values = []
                for chunk in chunks:
                    value = chunk["privileged_target_task"]
                    if isinstance(value, (bytes, bytearray)):
                        value = value.decode("utf-8")
                    elif not isinstance(value, str):
                        value = np.asarray(value).reshape(-1)[0].item()
                    values.append(value)
                task_values = np.asarray(values, dtype=object)
                init_state["privileged_target_task"] = task_values
                goal_state["goal_privileged_target_task"] = task_values.copy()
            return init_state, goal_state, dataset_videos

        world_module._extract_init_goal = extract_init_goal_with_scene_task
        world_module._phase3_string_goal_patched = True

    if not getattr(SceneEnv, "_phase3_patched", False):
        original_set_state = SceneEnv.set_state

        def set_state(self, qpos, qvel, button_states=None, **kwargs):
            if button_states is not None:
                states = np.asarray(button_states).reshape(-1)
                if states.size != 2:
                    raise ValueError("Scene button_states must contain two values")
                kwargs = dict(kwargs)
                kwargs.update(
                    button_state_0=int(states[0]), button_state_1=int(states[1])
                )
            return original_set_state(self, qpos, qvel, **kwargs)

        SceneEnv.set_state = set_state
        SceneEnv.set_phase3_scene_goal = set_phase3_scene_goal
        SceneEnv._phase3_patched = True

    if not getattr(FingerDMControlWrapper, "_phase3_patched", False):
        def set_target_position(self, target_position):
            target = np.asarray(target_position, dtype=np.float64).reshape(-1)
            if target.shape != (2,):
                raise ValueError("Finger target_position must have shape (2,)")
            spinner = self.env.physics.named.data.sensordata["spinner"][[0, 2]]
            absolute = spinner + target
            self.env.physics.named.model.site_pos["target", ["x", "z"]] = absolute
            self.env.physics.forward()

        def is_terminated(self, step):
            del step
            return bool((self.env.task.get_reward(self.env.physics) or 0.0) >= 1.0)

        FingerDMControlWrapper.set_target_position = set_target_position
        FingerDMControlWrapper._is_terminated = is_terminated
        FingerDMControlWrapper._phase3_patched = True

    if not getattr(HumanoidDMControlWrapper, "_phase3_patched", False):
        def is_terminated(self, step):
            del step
            physics = self.env.physics
            head_height = float(physics.head_height())
            torso_upright = float(physics.torso_upright())
            planar_speed = float(np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]]))
            return bool(
                head_height >= 1.4
                and torso_upright >= 0.9
                and planar_speed >= 1.0
            )

        HumanoidDMControlWrapper._is_terminated = is_terminated
        HumanoidDMControlWrapper._phase3_patched = True


def scene_target_group(dataset: Any, manifest: CohortManifest) -> list[str]:
    """Return the target_task label for each frozen cohort slot."""

    values = np.asarray(dataset.get_col_data("privileged_target_task"))
    result = []
    for entry in manifest.entries:
        value = values[int(entry.goal_row_index or entry.row_index)]
        result.append(str(_scalar(value)))
    return result


__all__ = [
    "PHASE3_DATASETS",
    "PHASE3_TASKS",
    "build_phase3_legacy_manifest",
    "patch_phase3_environments",
    "scene_target_group",
]
