"""Runtime goal-refresh checks for the four Phase 1 environments.

These checks are CPU-side and intentionally separate from model evaluation.
They bind the generic ten-case adapter to the installed environment objects so
the audit can distinguish a static source inspection from an actual goal-field
refresh check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .round3_phase1 import (
    _jsonable,
    make_goal_refresh_cases,
    run_goal_refresh_checks,
    sha256_json,
)
from .round3_protocol import (
    ROUND3_PROTOCOL,
    TASK_PREDICATES,
    evaluate_success,
    physical_distance_components,
)


def _normalise(task: str) -> str:
    return task.lower().replace("push-t", "pusht").replace("two_room", "tworoom")


def _reset_env(env: Any, seed: int) -> tuple[Any, Mapping[str, Any]]:
    try:
        result = env.reset(seed=int(seed))
    except TypeError:
        result = env.reset()
    if isinstance(result, tuple) and len(result) >= 2:
        info = result[1] if isinstance(result[1], Mapping) else {}
        return result[0], info
    return result, {}


def _reset_runtime_env(task: str, env: Any, seed: int) -> tuple[Any, Mapping[str, Any]]:
    """Reset an installed task while binding every task-owned RNG to ``seed``.

    The installed Cube wrapper consumes ``seed`` in its override and omits it
    when delegating to the Gym parent.  Its parent therefore keeps the prior
    ``np_random`` stream unless the adapter binds it explicitly.  The fixed
    task id below removes the remaining global-task sampling from the audit.
    """
    if _normalise(task) == "cube":
        setattr(env, "np_random", np.random.default_rng(int(seed)))
    return _reset_env(env, seed)


def _make_runtime_adapter(task: str, seed: int):
    key = _normalise(task)
    if key == "cube":
        from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

        env = CubeEnv(
            env_type="single",
            ob_type="states",
            terminate_at_goal=False,
            reward_task_id=2,
            permute_blocks=False,
        )
        _reset_runtime_env(key, env, seed)
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
        _reset_env(env, seed)

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

        env = PushT(render_mode="rgb_array", relative=True)
        _reset_env(env, seed)

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

        env = TwoRoomEnv(render_mode="rgb_array")
        _reset_env(env, seed)

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


def _state_key(value: Any) -> str:
    return sha256_json(_jsonable(value))


def _two_room_old_goal(state: np.ndarray, index: int) -> np.ndarray:
    """Choose a legal, deterministic old target opposite the current side."""
    low = float(14)
    high = float(224 - 14 - 1)
    state = _finite_vector(state, name="TwoRoom runtime state")
    if np.any(state < low) or np.any(state > high):
        raise ValueError("TwoRoom runtime state is outside the legal position bounds")
    midpoint = (low + high) / 2.0
    old = np.array(state, copy=True)
    old[0] = high if state[0] <= midpoint else low
    span = high - low
    old[1] = low + ((state[1] - low + 17.0 * (int(index) + 1)) % span)
    if float(np.linalg.norm(old - state)) < 16.0:
        old[1] = high if state[1] <= midpoint else low
    if float(np.linalg.norm(old - state)) < 16.0:
        raise ValueError("TwoRoom old goal is not sufficiently separated")
    return old


def run_runtime_goal_refresh_audit(
    task: str, *, count: int = 10, seed: int = 42
) -> dict[str, Any]:
    """Run ten deterministic, distinct goal-refresh checks in real environments."""
    if int(count) != 10:
        raise ValueError("Phase 1 runtime audit requires exactly ten checks")
    env = None
    key = _normalise(task)
    try:
        env, current, set_goal, read_goal, read_success = _make_runtime_adapter(task, seed)
        deltas = {
            "cube": np.asarray([0.08, 0.0, 0.0]),
            "reacher": np.asarray([0.10, -0.10]),
            "pusht": np.asarray([30.0, 0.0, 30.0, 0.0, 0.50, 0.0, 0.0]),
        }
        case_seeds = tuple(int(seed) + index * 1009 for index in range(10))
        states: list[list[float]] = []
        cases: list[dict[str, Any]] = []
        for index, case_seed in enumerate(case_seeds):
            _reset_runtime_env(key, env, case_seed)
            state = np.asarray(current(), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(state)):
                raise ValueError(f"{key} runtime state {index} is not finite")
            states.append(state.tolist())
            old_goal = (
                _two_room_old_goal(state, index)
                if key == "tworoom"
                else state + (index + 1) * deltas[key]
            )
            cases.append(
                {
                    "case": index,
                    "old_goal": old_goal.tolist(),
                    "new_goal": state.tolist(),
                    "current": state.tolist(),
                }
            )
        state_hashes = [_state_key(state) for state in states]
        replay_hashes = []
        for case_seed in case_seeds:
            _reset_runtime_env(key, env, case_seed)
            replay_state = np.asarray(current(), dtype=np.float64).reshape(-1)
            replay_hashes.append(_state_key(replay_state))
        result = run_goal_refresh_checks(
            task,
            cases,
            set_goal=set_goal,
            read_goal=read_goal,
            read_success=read_success,
            reset_case=lambda index, case: _reset_runtime_env(key, env, case_seeds[index]),
        )
        result["adapter"] = "installed_stable_worldmodel_runtime"
        result["seed"] = int(seed)
        result["case_seeds"] = list(case_seeds)
        result["states"] = states
        result["state_hashes"] = state_hashes
        result["states_unique"] = len(set(state_hashes)) == 10
        result["seed_reproducible"] = replay_hashes == state_hashes
        if not result["states_unique"]:
            result["status"] = "unaccepted"
            result["error"] = "runtime refresh states are not pairwise unique"
        diagnostic = run_independent_cpu_neutral_hold(
            key,
            seed=int(seed),
            protocol=ROUND3_PROTOCOL,
            protocol_variant="runtime_audit",
            stage="runtime_audit",
            cohort_id=f"{key}_runtime_audit",
            cohort_sha256=None,
            policy_identity={
                "entrypoint": "round3_runtime_audit",
                "policy_kind": "independent_cpu_runtime",
                "checkpoint": None,
                "epoch": None,
                "stage": "runtime_audit",
            },
        )
        result["neutral_hold_diagnostic"] = diagnostic
        result["neutral_hold"] = diagnostic
        try:
            validate_neutral_hold_diagnostic(diagnostic, task=key)
        except ValueError as exc:
            result["status"] = "unaccepted"
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001 - audit must record an unaccepted runtime
        return {
            "status": "unaccepted",
            "checks_expected": 10,
            "checks_run": 0,
            "adapter": "installed_stable_worldmodel_runtime",
            "states": [],
            "states_unique": False,
            "seed_reproducible": False,
            "neutral_hold_diagnostic": _unaccepted_neutral_hold(task, seed, exc),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "seed": int(seed),
        }
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


def _mapping_value(value: Any, names: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    return None


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _strict_bool(value: Any, *, name: str) -> bool:
    """Decode only a scalar bool, including a one-element bool ndarray."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray) and value.size == 1 and value.dtype == np.bool_:
        return bool(value.reshape(-1)[0])
    raise ValueError(f"{name} must be an explicit scalar boolean")


def _reset_result(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1] if isinstance(result[1], Mapping) else {}
    return result, {}


def _task_runtime_step(task: str, env: Any, action: Any) -> Any:
    """Dispatch one step through the installed task-specific adapter."""
    key = _normalise(task)
    action_array = np.asarray(action, dtype=np.float64).copy()
    if key == "reacher":
        return _reacher_continuous_step(env, action_array)
    if key == "cube":
        return env.step(action_array)
    if key == "pusht":
        return env.step(action_array)
    if key == "tworoom":
        return env.step(action_array)
    raise ValueError(f"unsupported task {task!r}")
def _call_env_factory(factory: Any, task: str, seed: int) -> Any:
    if factory is None:
        return None
    try:
        return factory(task=task, seed=int(seed))
    except TypeError:
        try:
            return factory(task, int(seed))
        except TypeError:
            return factory()


def _lookup_state(
    observation: Any,
    info: Mapping[str, Any],
    *,
    names: Sequence[str],
    env: Any,
) -> Any:
    value = _mapping_value(info, names)
    if value is None:
        value = _mapping_value(observation, names)
    if value is None:
        for name in names:
            if hasattr(env, name):
                value = getattr(env, name)
                break
    return value


def _generic_neutral_adapter(task: str, env: Any, seed: int) -> dict[str, Any]:
    definition = TASK_PREDICATES[_normalise(task)]

    def reset(value: int) -> tuple[Any, Mapping[str, Any]]:
        return _reset_result(_reset_runtime_env(task, env, value))

    def state(observation: Any, info: Mapping[str, Any]) -> np.ndarray:
        value = _lookup_state(
            observation,
            info,
            names=(definition.current_field, *definition.field_aliases, "state", "observation"),
            env=env,
        )
        return _finite_vector(value, name="runtime state")

    def goal(observation: Any, info: Mapping[str, Any]) -> np.ndarray:
        value = _lookup_state(
            observation,
            info,
            names=(definition.goal_field, *definition.goal_aliases, "goal", "goal_state"),
            env=env,
        )
        return _finite_vector(value, name="runtime goal")

    def set_goal(value: Any) -> None:
        array = _finite_vector(value, name="requested goal")
        for name in ("set_goal", "_set_goal_state", "set_target_qpos", "set_target_pos"):
            setter = getattr(env, name, None)
            if callable(setter):
                if name == "set_target_pos":
                    setter(0, array)
                else:
                    setter(array)
                return
        for name in ("goal_state", "goal", "target_position"):
            if hasattr(env, name):
                setattr(env, name, np.array(array, copy=True))
                return
        raise ValueError("runtime environment exposes no goal setter")

    return {
        "env": env,
        "backend": "injected_environment",
        "reset": reset,
        "state": state,
        "goal": goal,
        "set_goal": set_goal,
        "action_space": getattr(env, "action_space", None),
        "step": lambda action: _task_runtime_step(task, env, action),
        "step_source": f"{_normalise(task)}_task_adapter.step",
        "continuous": False,
        "relative": getattr(env, "relative", None),
    }


def _make_neutral_runtime_adapter(
    task: str,
    seed: int,
    *,
    env_factory: Any = None,
) -> dict[str, Any]:
    if env_factory is not None:
        env = _call_env_factory(env_factory, _normalise(task), seed)
        if env is None:
            raise ValueError("environment factory returned no environment")
        return _generic_neutral_adapter(task, env, seed)

    key = _normalise(task)
    env, current, set_goal, read_goal, _ = _make_runtime_adapter(key, seed)

    def reset(value: int) -> tuple[Any, Mapping[str, Any]]:
        observation, info = _reset_runtime_env(key, env, value)
        if _normalise(task) == "reacher":
            raw_env = getattr(getattr(env, "env", None), "_env", None)
            if raw_env is None:
                raise ValueError("Reacher continuous control environment is unavailable")
            qpos = np.asarray(raw_env.physics.data.qpos, dtype=np.float64).copy()
            qvel = np.zeros_like(raw_env.physics.data.qvel, dtype=np.float64)
            env.set_state(qpos, qvel)
            observation = env._obs_to_array(
                raw_env.task.get_observation(raw_env.physics)
            )
            info = dict(env.info)
        return observation, info

    def state(observation: Any, info: Mapping[str, Any]) -> np.ndarray:
        return _finite_vector(current(), name="runtime state")

    def goal(observation: Any, info: Mapping[str, Any]) -> np.ndarray:
        return _finite_vector(read_goal(), name="runtime goal")

    def set_runtime_goal(value: Any) -> None:
        set_goal(_finite_vector(value, name="requested goal"))

    return {
        "env": env,
        "backend": "stable_worldmodel_cpu_environment",
        "reset": reset,
        "state": state,
        "goal": goal,
        "set_goal": set_runtime_goal,
        "step": lambda action: _task_runtime_step(key, env, action),
        "step_source": (
            "reacher_dm_control_task_adapter.step" if key == "reacher"
            else f"{key}_task_adapter.step"
        ),
        "action_space": getattr(env, "action_space", None),
        "continuous": _normalise(task) == "reacher",
        "relative": getattr(env, "relative", None),
    }


def _action_bounds(action_space: Any, info: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    nested = _mapping_value(info, ("action_bounds", "neutral_action_bounds"))
    if isinstance(nested, Mapping):
        low = nested.get("low")
        high = nested.get("high")
        if low is not None and high is not None:
            return _finite_vector(low, name="action lower bound"), _finite_vector(high, name="action upper bound"), "env_info"
    for low_name, high_name in (
        ("action_low", "action_high"),
        ("action_space_low", "action_space_high"),
    ):
        low = _mapping_value(info, (low_name,))
        high = _mapping_value(info, (high_name,))
        if low is not None and high is not None:
            return _finite_vector(low, name="action lower bound"), _finite_vector(high, name="action upper bound"), "env_info"
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is not None and high is not None:
        return _finite_vector(low, name="action lower bound"), _finite_vector(high, name="action upper bound"), "action_space_bounds"
    return None, None, "unavailable"


def _neutral_action(
    action_space: Any,
    info: Mapping[str, Any],
    *,
    provided: Any = None,
) -> tuple[np.ndarray, str, str]:
    value = _mapping_value(info, ("neutral_action", "action"))
    if provided is not None:
        value = provided
        source = "caller"
        reason = "explicit deterministic neutral action supplied by the CPU audit"
    elif value is not None:
        source = str(_mapping_value(info, ("neutral_action_source",)) or "env_info")
        reason = str(_mapping_value(info, ("neutral_action_reason", "action_reason")) or "neutral action declared by environment info")
    else:
        shape = getattr(action_space, "shape", None)
        if shape is None:
            low = getattr(action_space, "low", None)
            shape = np.asarray(low).shape if low is not None else None
        if shape is None or int(np.prod(shape)) <= 0:
            raise ValueError("neutral action is unavailable")
        value = np.zeros(shape, dtype=np.float64)
        source = "action_space_bounds"
        reason = "deterministic zero action is the CPU neutral action"
    return _finite_vector(value, name="neutral action"), source, reason


def _task_neutral_action(
    task: str,
    action_space: Any,
    info: Mapping[str, Any],
    *,
    provided: Any = None,
) -> tuple[np.ndarray, str, str]:
    key = _normalise(task)
    expected_size = {"cube": 5, "pusht": 2, "reacher": 2, "tworoom": 2}[key]
    has_declared_action = _mapping_value(info, ("neutral_action", "action")) is not None
    if has_declared_action:
        if _mapping_value(info, ("neutral_action_source",)) in (None, ""):
            raise ValueError("neutral action source is unavailable")
        if _mapping_value(info, ("neutral_action_reason", "action_reason")) in (None, ""):
            raise ValueError("neutral action reason is unavailable")
    action, source, reason = _neutral_action(
        action_space,
        info,
        provided=provided,
    )
    if action.size != expected_size or not np.allclose(action, 0.0, rtol=0.0, atol=0.0):
        raise ValueError(f"{key} neutral action must be an exact zero vector of size {expected_size}")
    if not has_declared_action and provided is None:
        source = f"{key}_task_adapter"
        reason = f"{key} task-specific deterministic zero action from action-space bounds"
    if not source or source == "unavailable" or not reason:
        raise ValueError("neutral action source/reason is unavailable")
    return action, source, reason


def _unaccepted_neutral_hold(task: str, seed: int, exc: Exception) -> dict[str, Any]:
    return {
        "status": "unaccepted",
        "task": _normalise(task),
        "seed": int(seed),
        "independent": True,
        "independent_cpu_runtime": True,
        "cpu_only": True,
        "model_used": False,
        "cem_used": False,
        "action_legal": None,
        "action_source": "unavailable",
        "action_reason": str(exc),
        "action_bounds": None,
        "raw_steps": [],
        "steps": [],
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def validate_neutral_hold_diagnostic(
    diagnostic: Mapping[str, Any],
    *,
    task: str | None = None,
    expected_policy_identity: Mapping[str, Any] | None = None,
    expected_protocol: str | None = None,
    expected_protocol_variant: str | None = None,
    expected_stage: str | None = None,
    expected_cohort_id: str | None = None,
    expected_cohort_sha256: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless a complete, independent five-step CPU hold exists."""
    if not isinstance(diagnostic, Mapping):
        raise ValueError("neutral hold diagnostic must be a mapping")
    value = dict(diagnostic)
    if value.get("status") != "accepted":
        raise ValueError("neutral hold diagnostic is not accepted")
    expected_task = _normalise(task) if task is not None else value.get("task")
    if value.get("task") != expected_task:
        raise ValueError("neutral hold task identity mismatch")
    for name in ("independent", "independent_cpu_runtime"):
        if value.get(name) is not True:
            raise ValueError(f"neutral hold requires {name}=true")
    if value.get("cpu_only", True) is not True:
        raise ValueError("neutral hold must be CPU-only")
    for name in ("model_used", "cem_used"):
        if value.get(name) is not False:
            raise ValueError(f"neutral hold requires {name}=false")
    for name in ("protocol", "protocol_variant", "stage", "cohort_id", "cohort_sha256", "seed"):
        if name not in value:
            raise ValueError(f"neutral hold identity field {name} is missing")
    if expected_task is None:
        raise ValueError("neutral hold task is missing")
    if _normalise(expected_task) not in TASK_PREDICATES:
        raise ValueError("neutral hold task is unsupported")
    if expected_protocol is not None and value.get("protocol") != expected_protocol:
        raise ValueError("neutral hold protocol identity mismatch")
    if expected_protocol_variant is not None and value.get("protocol_variant") != expected_protocol_variant:
        raise ValueError("neutral hold protocol variant mismatch")
    if expected_stage is not None and value.get("stage") != expected_stage:
        raise ValueError("neutral hold stage identity mismatch")
    if expected_cohort_id is not None and value.get("cohort_id") != expected_cohort_id:
        raise ValueError("neutral hold cohort identity mismatch")
    if expected_cohort_sha256 is not None and value.get("cohort_sha256") != expected_cohort_sha256:
        raise ValueError("neutral hold cohort hash mismatch")
    if expected_seed is not None and value.get("seed") != int(expected_seed):
        raise ValueError("neutral hold seed mismatch")
    identity = value.get("policy_identity")
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("neutral hold policy identity is missing")
    if expected_policy_identity is not None:
        for key, expected in expected_policy_identity.items():
            if key not in identity or identity[key] != expected:
                raise ValueError(f"neutral hold policy identity mismatch at {key}")
    action = _finite_vector(value.get("neutral_action"), name="neutral action")
    expected_size = {"cube": 5, "pusht": 2, "reacher": 2, "tworoom": 2}[_normalise(expected_task)]
    if action.size != expected_size or not np.allclose(action, 0.0, rtol=0.0, atol=0.0):
        raise ValueError("neutral action does not match the frozen task contract")
    action_legal = value.get("action_legal")
    if action_legal is not True:
        raise ValueError("neutral hold action legality is unknown or false")
    source = str(value.get("action_source") or "")
    reason = str(value.get("action_reason") or "")
    bounds = value.get("action_bounds")
    if source in {"", "unavailable"} or not reason or not isinstance(bounds, Mapping):
        raise ValueError("neutral hold action provenance is incomplete")
    low = _finite_vector(bounds.get("low"), name="action lower bound")
    high = _finite_vector(bounds.get("high"), name="action upper bound")
    if low.shape != high.shape or low.shape != action.shape or np.any(low > high):
        raise ValueError("neutral hold action bounds are invalid")
    if not np.all(action >= low) or not np.all(action <= high):
        raise ValueError("neutral action is outside declared bounds")
    legality = value.get("action_legality")
    if not isinstance(legality, Mapping) or legality.get("legal") is not True:
        raise ValueError("nested neutral action legality is not accepted")
    if legality.get("source") != source or legality.get("reason") != reason:
        raise ValueError("nested action provenance disagrees with top-level provenance")
    if legality.get("bounds_source") not in {"env_info", "action_space_bounds", "action_space"}:
        raise ValueError("nested action bounds provenance is unavailable")
    if not isinstance(legality.get("bounds"), Mapping):
        raise ValueError("nested action bounds are missing")
    raw_steps = value.get("raw_steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != 5:
        raise ValueError("neutral hold must contain exactly five raw steps")
    for expected_step, row in enumerate(raw_steps, start=1):
        if not isinstance(row, Mapping) or row.get("raw_step") != expected_step:
            raise ValueError("neutral hold raw steps must be numbered 1..5")
        row_action = _finite_vector(row.get("action"), name="neutral hold action")
        if row_action.shape != action.shape or not np.array_equal(row_action, action):
            raise ValueError("neutral hold action changed across raw steps")
        if row.get("success") is not True:
            raise ValueError("neutral hold did not succeed on every raw step")
        terminated = _strict_bool(row.get("terminated"), name="terminated")
        truncated = _strict_bool(row.get("truncated"), name="truncated")
        termination_reason = row.get("termination_reason")
        if not isinstance(termination_reason, str) or not termination_reason:
            raise ValueError("neutral hold termination reason is missing")
        if truncated:
            raise ValueError("neutral hold was truncated before five steps")
        if terminated and termination_reason not in {"success", "terminated"}:
            raise ValueError("terminated neutral hold step lacks a compatible reason")
        state = _finite_vector(row.get("state"), name="neutral hold state")
        goal = _finite_vector(row.get("goal"), name="neutral hold goal")
        try:
            predicate_success = bool(evaluate_success(expected_task, state, goal))
        except (TypeError, ValueError) as exc:
            raise ValueError("neutral hold state/goal cannot be evaluated") from exc
        if not predicate_success:
            raise ValueError("neutral hold state is not successful under the frozen predicate")
        components = row.get("distance_components")
        if not isinstance(components, Mapping) or not components:
            raise ValueError("neutral hold distance components are missing")
        for name in ("distance_unit", "distance_aggregation", "distance_source"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"neutral hold {name} is missing")
        if _normalise(expected_task) == "cube" and row.get("env_success") is not True:
            raise ValueError("Cube neutral hold requires explicit environment success")
        if "predicate_success" in row and row["predicate_success"] is not True:
            raise ValueError("neutral hold predicate success disagrees")
        if "env_success" in row and row["env_success"] is not None and row["env_success"] is not True:
            raise ValueError("neutral hold environment success disagrees")
    if value.get("steps") is not None and value.get("steps") != raw_steps:
        raise ValueError("neutral hold steps alias disagrees with raw_steps")
    if value.get("raw_step_count", 5) != 5 or value.get("env_steps", 5) != 5:
        raise ValueError("neutral hold step count is not exactly five")
    return value


def _termination_reason(
    info: Mapping[str, Any],
    *,
    terminated: bool,
    truncated: bool,
    success: bool,
) -> tuple[str, str]:
    """Resolve a strict termination reason and fail closed on contradictions."""
    declared = info.get("termination_reason")
    if declared is None:
        if truncated:
            reason = "time_limit"
            source = "derived_from_truncation"
        elif terminated:
            reason = "success" if success else "unknown"
            source = "derived_from_termination_and_frozen_predicate"
        else:
            reason = "running"
            source = "derived_from_step_flags"
    else:
        if not isinstance(declared, str) or not declared:
            raise ValueError("termination_reason must be a non-empty string")
        reason = declared
        source = "environment_info"
    if reason not in {"running", "success", "terminated", "time_limit"}:
        raise ValueError(f"unknown termination reason {reason!r}")
    if truncated and reason != "time_limit":
        raise ValueError("truncated step lacks time_limit termination reason")
    if terminated and reason not in {"success", "terminated"}:
        raise ValueError("terminated step lacks success termination reason")
    if not terminated and not truncated and reason != "running":
        raise ValueError("non-terminal step has a non-running termination reason")
    if reason == "success" and not success:
        raise ValueError("success termination reason disagrees with frozen predicate")
    return reason, source
def run_independent_cpu_neutral_hold(
    task: str,
    *,
    seed: int = 42,
    protocol: str = ROUND3_PROTOCOL,
    protocol_variant: str | None = None,
    stage: str | None = None,
    cohort_id: str | None = None,
    cohort_sha256: str | None = None,
    policy_identity: Mapping[str, Any] | None = None,
    env_factory: Any = None,
    neutral_action: Any = None,
    goal_state: Any = None,
) -> dict[str, Any]:
    """Run exactly five neutral actions in an independent CPU environment."""
    key = _normalise(task)
    env = None
    try:
        adapter = _make_neutral_runtime_adapter(key, int(seed), env_factory=env_factory)
        env = adapter["env"]
        observation, reset_info = adapter["reset"](int(seed))
        state = adapter["state"](observation, reset_info)
        requested_goal = (
            _finite_vector(goal_state, name="requested goal")
            if goal_state is not None
            else np.array(state, copy=True)
        )
        adapter["set_goal"](requested_goal)
        goal = adapter["goal"](observation, reset_info)
        goal_atol = 1e-5 if key == "tworoom" else 1e-8
        if goal.shape != requested_goal.shape or not np.allclose(goal, requested_goal, rtol=0.0, atol=goal_atol):
            raise ValueError("runtime goal setter did not provide an exact readback")
        action, action_source, action_reason = _task_neutral_action(key, adapter["action_space"], reset_info, provided=neutral_action)
        low, high, bounds_source = _action_bounds(adapter["action_space"], reset_info)
        if low is None or high is None:
            raise ValueError("neutral action bounds are unavailable")
        if low.shape != high.shape or low.shape != action.shape or np.any(low > high):
            raise ValueError("neutral action bounds do not match action shape")
        if not np.all(action >= low) or not np.all(action <= high):
            raise ValueError("neutral action is outside environment bounds")
        if action_source == "unavailable" or not action_reason:
            raise ValueError("neutral action source/reason is unavailable")
        raw_steps = []
        step_source = str(adapter.get("step_source") or "task_specific_adapter.step")
        for raw_step in range(1, 6):
            step_result = adapter["step"](np.array(action, copy=True))
            observation, _reward, terminated, truncated, info_value = _parse_step_result(step_result)
            state = adapter["state"](observation, info_value)
            goal = adapter["goal"](observation, info_value)
            success = bool(evaluate_success(key, state, goal))
            env_success_value = _mapping_value(info_value, ("success", "is_success"))
            if key == "reacher":
                _optional_success(
                    env_success_value,
                    name="env success",
                    ignore_nan=True,
                    ignore_invalid=True,
                )
                env_success = None
            else:
                env_success = (
                    None
                    if env_success_value is None
                    else _optional_success(env_success_value, name="env success")
                )
            if env_success is not None and env_success is not success:
                raise ValueError("environment success disagrees with the frozen predicate")
            termination_reason, termination_reason_source = _termination_reason(
                info_value,
                terminated=terminated,
                truncated=truncated,
                success=success,
            )
            raw_steps.append(
                {
                    "raw_step": raw_step,
                    "raw_env_step": raw_step,
                    "action": _jsonable(action),
                    "success": success,
                    "predicate_success": success,
                    "env_success": env_success,
                    "terminated": terminated,
                    "truncated": truncated,
                    "termination_reason": termination_reason,
                    "termination_reason_source": termination_reason_source,
                    "step_source": step_source,
                    "termination_evidence": str(info_value.get("termination_evidence") or "task_adapter_step_result"),
                    "state": _jsonable(state),
                    "goal": _jsonable(goal),
                    "distance_components": _jsonable(physical_distance_components(key, state, goal)),
                    "distance_unit": TASK_PREDICATES[key].unit,
                    "distance_aggregation": "named task components; success uses frozen predicate",
                    "distance_source": "independent_cpu_runtime_state_goal",
                }
            )
            if truncated:
                raise ValueError("environment truncated during the five-step hold")
        bounds = {"low": _jsonable(low), "high": _jsonable(high)}
        legality = {
            "legal": True,
            "source": action_source,
            "bounds_source": bounds_source,
            "bounds": bounds,
            "reason": action_reason,
        }
        diagnostic = {
            "status": "accepted",
            "schema_version": 1,
            "task": key,
            "policy_identity": dict(policy_identity or {}),
            "protocol": protocol,
            "protocol_variant": protocol_variant,
            "stage": stage,
            "cohort_id": cohort_id,
            "cohort_sha256": cohort_sha256,
            "seed": int(seed),
            "independent": True,
            "independent_cpu_runtime": True,
            "cpu_only": True,
            "model_used": False,
            "cem_used": False,
            "env_backend": adapter["backend"],
            "step_source": step_source,
            "neutral_action": _jsonable(action),
            "action_legal": True,
            "action_source": action_source,
            "action_reason": action_reason,
            "action_bounds": bounds,
            "action_legality": legality,
            "raw_step_count": 5,
            "env_steps": 5,
            "raw_steps": raw_steps,
            "steps": raw_steps,
            "predicate": TASK_PREDICATES[key].as_dict(),
        }
        validate_neutral_hold_diagnostic(
            diagnostic,
            task=key,
            expected_protocol=protocol,
            expected_protocol_variant=protocol_variant,
            expected_stage=stage,
            expected_cohort_id=cohort_id,
            expected_cohort_sha256=cohort_sha256,
            expected_seed=seed,
        )
        return diagnostic
    except Exception as exc:  # noqa: BLE001 - callers must fail closed
        return _unaccepted_neutral_hold(key, int(seed), exc)
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


__all__ = [
    "run_independent_cpu_neutral_hold",
    "run_runtime_goal_refresh_audit",
    "validate_neutral_hold_diagnostic",
]
def _parse_step_result(result: Any) -> tuple[Any, Any, bool, bool, Mapping[str, Any]]:
    if not isinstance(result, (tuple, list)):
        raise ValueError("environment step must return a tuple/list")
    if len(result) == 5:
        observation, reward, terminated_value, truncated_value, info = result
        terminated = _strict_bool(terminated_value, name="terminated")
        truncated = _strict_bool(truncated_value, name="truncated")
    elif len(result) == 4:
        observation, reward, done_value, info = result
        done = _strict_bool(done_value, name="done")
        if not isinstance(info, Mapping) or "TimeLimit.truncated" not in info:
            raise ValueError("legacy four-tuple requires explicit TimeLimit.truncated info")
        truncated = _strict_bool(info["TimeLimit.truncated"], name="TimeLimit.truncated")
        if truncated and not done:
            raise ValueError("legacy done=False cannot be TimeLimit.truncated")
        terminated = bool(done and not truncated)
    else:
        raise ValueError("environment step must return exactly four or five values")
    if not isinstance(info, Mapping):
        raise ValueError("environment step info must be a mapping")
    return observation, reward, terminated, truncated, info


def _optional_success(
    value: Any, *, name: str, ignore_nan: bool = False, ignore_invalid: bool = False
) -> bool | None:
    if value is None:
        return None
    array = np.asarray(value)
    if ignore_nan and array.size and array.dtype.kind in "fc" and np.all(np.isnan(array)):
        return None
    try:
        return _strict_bool(value, name=name)
    except ValueError:
        if ignore_invalid:
            return None
        raise


def _reacher_continuous_step(
    env: Any, action: Any
) -> tuple[Any, Any, bool, bool, Mapping[str, Any]]:
    wrapper = getattr(env, "env", None)
    raw_env = getattr(wrapper, "_env", None)
    if raw_env is None:
        raise ValueError("Reacher continuous control environment is unavailable")
    if hasattr(raw_env, "_reset_next_step"):
        raw_env._reset_next_step = False
    raw_action = (
        wrapper._transform(np.asarray(action, dtype=np.float64))
        if hasattr(wrapper, "_transform")
        else np.asarray(action, dtype=np.float64)
    )
    time_step = raw_env.step(raw_action)
    time_step_last = _strict_bool(time_step.last(), name="dm_control TimeStep.last")
    termination = raw_env.task.get_termination(raw_env.physics)
    terminated = bool(time_step_last and termination is not None)
    truncated = bool(time_step_last and termination is None)
    info = dict(env.info)
    info["termination_value"] = termination
    info["termination_evidence"] = "dm_control.TimeStep.last + qpos_match.get_termination"
    info["termination_reason"] = (
        "success" if terminated else "time_limit" if truncated else "running"
    )
    return (
        env._obs_to_array(time_step.observation),
        time_step.reward or 0.0,
        terminated,
        truncated,
        info,
    )
