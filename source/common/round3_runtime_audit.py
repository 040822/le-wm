"""Runtime goal-refresh checks for the four Phase 1 environments.

These checks are CPU-side and intentionally separate from model evaluation.
They bind the generic ten-case adapter to the installed environment objects so
the audit can distinguish a static source inspection from an actual goal-field
refresh check.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .round3_phase1 import make_goal_refresh_cases, run_goal_refresh_checks


def _normalise(task: str) -> str:
    return task.lower().replace("push-t", "pusht").replace("two_room", "tworoom")


def _make_runtime_adapter(task: str, seed: int):
    key = _normalise(task)
    if key == "cube":
        from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

        env = CubeEnv(env_type="single", ob_type="states")
        env.reset(seed=int(seed))
        target_id = env._cube_target_mocap_ids[0]

        def current():
            return np.asarray(env._data.joint("object_joint_0").qpos[:3], dtype=np.float64).copy()

        def set_goal(value):
            env.set_target_pos(0, value)

        def read_goal():
            return np.asarray(env._data.mocap_pos[target_id], dtype=np.float64).copy()

        def read_success():
            return bool(env._compute_successes()[0])

    elif key == "reacher":
        from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

        env = ReacherDMControlWrapper(task="qpos_match", seed=int(seed), render_mode=None)
        env.reset()

        def current():
            return np.asarray(env.env.physics.data.qpos, dtype=np.float64).copy()

        def set_goal(value):
            env.set_target_qpos(value)

        def read_goal():
            return np.asarray(env.env.task.target_qpos, dtype=np.float64).copy()

        def read_success():
            return env.env.task.get_termination(env.env.physics) is not None

    elif key == "pusht":
        from stable_worldmodel.envs.pusht.env import PushT

        env = PushT(render_mode="rgb_array")
        env.reset(seed=int(seed))

        def current():
            return np.asarray(env._get_obs(), dtype=np.float64).copy()

        def set_goal(value):
            env._set_goal_state(np.asarray(value, dtype=np.float64).copy())

        def read_goal():
            return np.asarray(env.goal_state, dtype=np.float64).copy()

        def read_success():
            return bool(env.eval_state(env.goal_state, current())[0])

    elif key == "tworoom":
        from stable_worldmodel.envs.two_room.env import TwoRoomEnv

        import torch

        env = TwoRoomEnv(render_mode="rgb_array")
        env._cache_params()
        env.agent_position = torch.tensor([60.0, 112.0], dtype=torch.float32)
        env.target_position = torch.tensor([164.0, 112.0], dtype=torch.float32)

        def current():
            return np.asarray(env.agent_position.detach().cpu().numpy(), dtype=np.float64).copy()

        def set_goal(value):
            env._set_goal_state(np.asarray(value, dtype=np.float32))

        def read_goal():
            return np.asarray(env.target_position.detach().cpu().numpy(), dtype=np.float64).copy()

        def read_success():
            return bool(np.linalg.norm(current() - read_goal()) < 16.0)

    else:
        raise ValueError(f"unsupported task {task!r}")

    return env, current, set_goal, read_goal, read_success


def run_runtime_goal_refresh_audit(
    task: str, *, count: int = 10, seed: int = 42
) -> dict[str, Any]:
    """Run ten deterministic old/new goal checks against installed runtime code."""
    if int(count) != 10:
        raise ValueError("Phase 1 runtime audit requires exactly ten checks")
    env = None
    try:
        env, current, set_goal, read_goal, read_success = _make_runtime_adapter(task, seed)
        baseline = current()
        key = _normalise(task)
        deltas = {
            "cube": np.asarray([0.02, 0.0, 0.0]),
            "reacher": np.asarray([0.02, -0.02]),
            "pusht": np.asarray([5.0, 0.0, 5.0, 0.0, 0.05, 0.0, 0.0]),
            "tworoom": np.asarray([5.0, 0.0]),
        }
        delta = deltas[key]
        cases = tuple(
            {
                "case": index,
                "old_goal": (baseline + (index + 1) * delta).tolist(),
                "new_goal": baseline.tolist(),
                "current": baseline.tolist(),
            }
            for index in range(10)
        )
        result = run_goal_refresh_checks(
            task,
            cases,
            set_goal=set_goal,
            read_goal=read_goal,
            read_success=read_success,
        )
        result["adapter"] = "installed_stable_worldmodel_runtime"
        result["seed"] = int(seed)
        return result
    except Exception as exc:  # noqa: BLE001 - audit must record an unaccepted runtime
        return {
            "status": "unaccepted",
            "checks_expected": 10,
            "checks_run": 0,
            "adapter": "installed_stable_worldmodel_runtime",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "seed": int(seed),
        }
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


__all__ = ["run_runtime_goal_refresh_audit"]
