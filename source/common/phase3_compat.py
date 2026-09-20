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

        def set_phase3_scene_goal(
            self,
            target_task,
            target_block_pos,
            target_button,
            target_button_state,
            target_drawer_pos,
            target_window_pos,
            goal_button_states=None,
        ):
            task_name = str(_scalar(target_task))
            self.set_cube_target_pos(0, np.asarray(target_block_pos).reshape(-1)[:3])
            if goal_button_states is not None:
                states = np.asarray(goal_button_states).reshape(-1).astype(np.int64)
                if states.size != 2:
                    raise ValueError("Scene goal button_states must contain two values")
            else:
                states = np.zeros(2, dtype=np.int64)
            if goal_button_states is None and task_name == "button":
                button_id = int(_scalar(target_button))
                if button_id not in (0, 1):
                    raise ValueError(f"Scene target button must be 0 or 1, got {button_id}")
                states[button_id] = int(_scalar(target_button_state))
            self.set_target_button_state(0, int(states[0]))
            self.set_target_button_state(1, int(states[1]))
            self.set_target_drawer_pos(float(np.asarray(target_drawer_pos).reshape(-1)[0]))
            self.set_target_window_pos(float(np.asarray(target_window_pos).reshape(-1)[0]))

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
