"""Simulator adapters for collecting true planner-action trajectories."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
from omegaconf import OmegaConf
import stable_worldmodel as swm
import torch

from eval_fast_lewam import load_model_from_weights
from source.common.eval import (
    compose_eval_config,
    fit_eval_processors,
    get_dataset,
    img_transform,
    select_eval_cohort,
)
from source.diagnostics.stage_b_episode_failures import (
    FixedRawSequencePolicy,
    build_prefix_replay_actions,
    build_shared_initial_panel,
    normalize_expert_plan,
    prepare_image_info,
)
from source.diagnostics.stage_b_epoch_pair_ranking import (
    CandidatePanel,
    compute_pair_physical_terminal_cost,
)
from source.experiments.planner_transition_replay import (
    PlannerTransitionReplay,
    load_replay,
    save_replay,
)


class BlockTrajectoryPolicy(FixedRawSequencePolicy):
    """Execute fixed raw actions and retain observations at Action Block seams."""

    def __init__(self, actions: np.ndarray, *, action_block: int):
        if int(action_block) < 1:
            raise ValueError("action_block must be positive")
        self.action_block = int(action_block)
        self._block_observations: list[torch.Tensor] = []
        self._initial_info: dict[str, torch.Tensor] | None = None
        super().__init__(actions)

    def set_actions(self, actions: np.ndarray) -> None:
        super().set_actions(actions)
        self._block_observations = []
        self._initial_info = None
        if self.actions.shape[1] % self.action_block:
            raise ValueError("raw action length must be divisible by action_block")

    @staticmethod
    def _pixels(info: dict[str, Any]) -> torch.Tensor:
        if "pixels" not in info:
            raise KeyError("simulator info is missing pixels")
        pixels = torch.as_tensor(info["pixels"]).detach().cpu()
        if pixels.ndim == 5 and pixels.shape[1] == 1:
            pixels = pixels[:, 0]
        return pixels.clone()

    def get_action(self, info: dict[str, Any]) -> np.ndarray:
        if self._initial_info is None:
            self._initial_info = {
                key: torch.as_tensor(info[key]).detach().cpu().clone()
                for key in ("pixels", "goal")
                if key in info
            }
        if self.step % self.action_block == 0:
            self._block_observations.append(self._pixels(info))
        return super().get_action(info)

    def initial_info(self) -> dict[str, torch.Tensor]:
        if self._initial_info is None or "goal" not in self._initial_info:
            raise RuntimeError("simulator did not provide initial pixels and goal")
        return dict(self._initial_info)

    def finish(self, terminal_info: dict[str, Any]) -> torch.Tensor:
        terminal = self._pixels(terminal_info)
        expected = self.actions.shape[1] // self.action_block
        if len(self._block_observations) > expected:
            raise RuntimeError(
                f"recorded {len(self._block_observations)} block states, expected {expected}"
            )
        missing = expected - len(self._block_observations)
        sequence = [*self._block_observations]
        sequence.extend(terminal.clone() for _ in range(missing))
        sequence.append(terminal)
        return torch.stack(sequence, dim=1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _grounding_payload_sha256(payload) -> str:
    digest = hashlib.sha256()
    values = (
        payload["physical"],
        payload["success"],
        payload["initial"]["pixels"],
        payload["initial"]["goal"],
    )
    for value in values:
        array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_replay_identity(manifest, pilot, level: int) -> dict[str, Any]:
    """Bind replay reuse to every model, trace, slot, and protocol input."""
    paths = {
        "baseline_checkpoint": pilot.baseline_checkpoint,
        "baseline_run_config": pilot.baseline_run_config,
        "actor_checkpoint": pilot.actor_checkpoint,
        "actor_run_config": pilot.actor_run_config,
        "trace": pilot.trace,
    }
    missing = [str(path) for path in paths.values() if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pilot inputs are missing: {missing}")
    cfg = compose_eval_config(pilot.task, ("output.save_video=false",))
    code_paths = (
        Path(__file__),
        Path(__file__).with_name("planner_transition_replay.py"),
        manifest.repository_root / "scripts" / "run_fast_lewam_pilot_worker.py",
    )
    collection_core = {
        "task": pilot.task,
        "inputs": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in paths.items()
            if path is not None
        },
        "replay_protocol": dict(manifest.replay),
        "evaluation": OmegaConf.to_container(cfg, resolve=True),
        "collection_code": {
            str(path.relative_to(manifest.repository_root)): _file_sha256(path)
            for path in code_paths
        },
    }
    encoded = repr(collection_core).encode("utf-8")
    return {
        "schema_version": 2,
        "manifest_sha256": manifest.sha256,
        "pilot": pilot.name,
        "task": pilot.task,
        "level": int(level),
        "slots": list(pilot.slots[: manifest.levels[int(level)].grounded_slots]),
        **collection_core,
        "collection_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _actor_anchor(actor, planning_info, *, device: str, seed: int) -> torch.Tensor:
    current = actor._last_frame(planning_info["pixels"]).to(device)
    goal = actor._last_frame(planning_info["goal"]).to(device)
    with torch.no_grad():
        z0 = actor.encode_pixels(current)
        goal_latent = actor.encode_pixels(goal)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        return actor.sample_actions(
            z0,
            generator=generator,
            goal_latent=goal_latent,
        )[0].detach().cpu()


@torch.inference_mode()
def _rank_candidate_costs(model, planning_info, actions, device):
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    count = len(actions)
    expanded = {}
    for key, value in planning_info.items():
        if torch.is_tensor(value):
            value = value.to(device)
            expanded[key] = value[:, None].expand(value.shape[0], count, *value.shape[1:])
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None], count, axis=1)
    result = model.get_cost(expanded, actions.to(device).unsqueeze(0))[0].cpu()
    if hasattr(model, "train"):
        model.train(was_training)
    return result


def _slot_candidate_panel(
    *, detail, expert, actor_action, replay_cfg, seed: int,
    ranking_model, planning_info, device: str,
) -> CandidatePanel:
    candidates = torch.as_tensor(detail["candidates"]).detach().cpu().float()
    costs = _rank_candidate_costs(
        ranking_model, planning_info, candidates, device
    )
    order = torch.argsort(costs, stable=True)
    elite_count = min(30, len(order))
    selected = order[:elite_count].tolist()
    remaining = order[elite_count:].tolist()
    generator = torch.Generator().manual_seed(int(seed))
    if remaining:
        permutation = torch.randperm(len(remaining), generator=generator).tolist()
        selected.extend(
            remaining[index]
            for index in permutation[: int(replay_cfg["fixed_non_elites"])]
        )
    cem_actions = candidates[selected]
    cem_panel = CandidatePanel(
        actions=cem_actions,
        sources=tuple(
            "model_elite" if index < elite_count else "model_non_elite"
            for index in range(len(cem_actions))
        ),
        source_aliases=tuple(
            ("model_elite",) if index < elite_count else ("model_non_elite",)
            for index in range(len(cem_actions))
        ),
        sha256="pending",
    )
    shared = build_shared_initial_panel(
        expert,
        actor_action,
        random_candidates=int(replay_cfg["random_candidates"]),
        actor_neighbors=int(replay_cfg["actor_neighbors"]),
        actor_noise_std=float(replay_cfg["actor_noise_std"]),
        generator=torch.Generator().manual_seed(int(seed)),
    )
    actions = torch.cat((cem_panel.actions, shared.actions), dim=0)
    sources = tuple(cem_panel.sources) + tuple(shared.sources)
    aliases = tuple(cem_panel.source_aliases) + tuple(
        (source,) for source in shared.sources
    )
    from source.diagnostics.stage_b_epoch_pair_ranking import candidate_actions_sha256

    expected_count = elite_count + min(
        int(replay_cfg["fixed_non_elites"]), len(remaining)
    ) + len(shared.actions)
    if len(actions) != expected_count:
        raise RuntimeError("candidate panel budget changed during construction")
    return CandidatePanel(
        actions=actions,
        sources=sources,
        source_aliases=aliases,
        sha256=candidate_actions_sha256(actions),
    )


def _encode_pixels(model, pixels, transform, device):
    prepared = prepare_image_info({"pixels": pixels}, transform, device)["pixels"]
    with torch.no_grad():
        encoded = model.encode_pixels(prepared)
    return encoded.detach().cpu().float()


def _collect_slot(
    *,
    cfg,
    dataset,
    scaler,
    model,
    ranking_model,
    actor,
    plan,
    detail,
    episode,
    start,
    task,
    slot,
    replay_cfg,
    device,
):
    episode = int(episode)
    start = int(start)
    horizon = int(cfg.plan_config.horizon)
    action_block = int(cfg.plan_config.action_block)
    chunk = dataset.load_chunk(
        np.asarray([episode]),
        np.asarray([start]),
        np.asarray([start + horizon * action_block]),
    )[0]
    expert = normalize_expert_plan(
        chunk, scaler=scaler, horizon=horizon, action_block=action_block
    )
    actor_action = _actor_anchor(
        actor, plan["planning_info"], device=device, seed=int(replay_cfg["seed"]) + slot
    )
    panel = _slot_candidate_panel(
        detail=detail,
        expert=expert,
        actor_action=actor_action,
        replay_cfg=replay_cfg,
        seed=int(replay_cfg["seed"]) + slot,
        ranking_model=ranking_model,
        planning_info=plan["planning_info"],
        device=device,
    )
    raw = build_prefix_replay_actions(
        panel.actions, scaler=scaler, action_block=action_block, prefix_raw=None
    )
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    all_latents = []
    all_goals = []
    all_physical = []
    all_success = []
    batch_size = int(replay_cfg["simulator_batch_size"])
    for offset in range(0, len(raw), batch_size):
        batch_raw = raw[offset : offset + batch_size]
        policy = BlockTrajectoryPolicy(batch_raw, action_block=action_block)
        world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
        world_cfg["num_envs"] = len(batch_raw)
        world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
        world = swm.World(**world_cfg, image_shape=(224, 224))
        world.set_policy(policy)
        try:
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=[episode] * len(batch_raw),
                start_steps=[start] * len(batch_raw),
                goal_offset=int(cfg.eval.goal_offset_steps),
                eval_budget=int(batch_raw.shape[1]),
                callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
                reset_mode="wait",
            )
            trajectory = policy.finish({"pixels": world.infos["pixels"]})
            all_latents.append(_encode_pixels(model, trajectory, transform, device))
            terminal = prepare_image_info(
                {"pixels": world.infos["goal"]}, transform, device
            )["pixels"]
            with torch.no_grad():
                all_goals.append(
                    model.encode_pixels(model._last_frame(terminal))
                    .detach().cpu().float()
                )
            all_physical.append(compute_pair_physical_terminal_cost(task, world.infos))
            all_success.append(
                np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
            )
        finally:
            world.close()
    latents = torch.cat(all_latents)
    return PlannerTransitionReplay(
        states=latents[:, :-1],
        actions=panel.actions.float(),
        targets=latents[:, 1:],
        sources=tuple(
            "expert" if "expert" in aliases else "planner"
            for aliases in panel.source_aliases
        ),
        groups=(int(slot),) * len(panel.actions),
        identity={"slot": int(slot), "candidate_sha256": panel.sha256},
        goal_latents=torch.cat(all_goals),
        physical_costs=torch.from_numpy(np.concatenate(all_physical)).float(),
        successes=torch.from_numpy(np.concatenate(all_success)),
    )


def concatenate_replays(
    replays: list[PlannerTransitionReplay], *, identity: dict[str, Any]
) -> PlannerTransitionReplay:
    return PlannerTransitionReplay(
        states=torch.cat([value.states for value in replays]),
        actions=torch.cat([value.actions for value in replays]),
        targets=torch.cat([value.targets for value in replays]),
        sources=tuple(source for value in replays for source in value.sources),
        groups=tuple(group for value in replays for group in value.groups),
        identity=identity,
        goal_latents=torch.cat([value.goal_latents for value in replays]),
        physical_costs=torch.cat([value.physical_costs for value in replays]),
        successes=torch.cat([value.successes for value in replays]),
    )


def collect_true_transition_replay(
    manifest,
    pilot,
    *,
    level: int,
    device: str,
    on_group: Callable[[PlannerTransitionReplay], None] | None = None,
    ranking_model=None,
    cache_dir: str | Path | None = None,
) -> PlannerTransitionReplay:
    """Collect fixed-slot CEM/actor/Gaussian transitions from the simulator."""
    identity = build_replay_identity(manifest, pilot, level)
    cfg = compose_eval_config(
        pilot.task,
        (f"solver.device={device}", "output.save_video=false"),
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    cohort = select_eval_cohort(
        dataset,
        goal_offset_steps=cfg.eval.goal_offset_steps,
        num_eval=cfg.eval.num_eval,
        seed=cfg.seed,
    )
    scaler = fit_eval_processors(dataset, cfg.dataset.keys_to_cache)["action"]
    baseline_cfg = OmegaConf.load(pilot.baseline_run_config)
    actor_cfg = OmegaConf.load(pilot.actor_run_config)
    model = load_model_from_weights(baseline_cfg, pilot.baseline_checkpoint, device)
    actor = load_model_from_weights(actor_cfg, pilot.actor_checkpoint, device)
    ranking_model = model if ranking_model is None else ranking_model
    cache_dir = None if cache_dir is None else Path(cache_dir)
    trace = torch.load(pilot.trace, map_location="cpu", weights_only=False)
    plans = {
        (int(row["slot"]), int(row["replan"])): row for row in trace["plans"]
    }
    details = {
        (int(row["slot"]), int(row["replan"]), int(row["iteration"])): row
        for row in trace["details"]
    }
    replay_cfg = {**manifest.replay, "seed": manifest.seed}
    selected_slots = pilot.slots[: manifest.levels[int(level)].grounded_slots]
    groups = []
    for slot_position, slot in enumerate(selected_slots):
        cache_path = None if cache_dir is None else cache_dir / f"slot_{int(slot):02d}.pt"
        ranking_state_sha256 = _model_state_sha256(ranking_model)
        group = None
        if cache_path is not None and cache_path.is_file():
            cached = load_replay(cache_path)
            reuse_level_one_panel = (
                int(level) == 2
                and slot_position < manifest.levels[1].grounded_slots
            )
            expected = (
                identity["collection_sha256"],
                int(slot),
                (
                    cached.identity.get("ranking_state_sha256")
                    if reuse_level_one_panel
                    else ranking_state_sha256
                ),
            )
            actual = (
                cached.identity.get("collection_sha256"),
                int(cached.identity.get("slot", -1)),
                cached.identity.get("ranking_state_sha256"),
            )
            if actual != expected:
                raise RuntimeError(f"group replay identity mismatch: {cache_path}")
            group = cached
        if group is None:
            plan = plans[(int(slot), 0)]
            detail = details[(int(slot), 0, int(manifest.replay["iteration"]))]
            group = _collect_slot(
                cfg=cfg,
                dataset=dataset,
                scaler=scaler,
                model=model,
                ranking_model=ranking_model,
                actor=actor,
                plan=plan,
                detail=detail,
                episode=cohort.episode_ids[int(slot)],
                start=cohort.start_steps[int(slot)],
                task=pilot.task,
                slot=int(slot),
                replay_cfg=replay_cfg,
                device=device,
            )
            group = PlannerTransitionReplay(
                states=group.states,
                actions=group.actions,
                targets=group.targets,
                sources=group.sources,
                groups=group.groups,
                identity={
                    **dict(group.identity),
                    "collection_sha256": identity["collection_sha256"],
                    "ranking_state_sha256": ranking_state_sha256,
                },
                goal_latents=group.goal_latents,
                physical_costs=group.physical_costs,
                successes=group.successes,
            )
            if cache_path is not None:
                save_replay(cache_path, group)
        groups.append(group)
        if on_group is not None:
            on_group(group)
    return concatenate_replays(groups, identity=identity)


def collect_action_token_grounding(
    manifest, pilot, *, level: int, control_model, variant_model, device: str
):
    """Ground one raw-action panel and score it in 5x10 and 25x2 form."""
    from source.experiments.fast_lewam_parallel_pilots import reshape_action_blocks
    from source.diagnostics.stage_b_epoch_pair_ranking import candidate_actions_sha256

    cfg = compose_eval_config(
        pilot.task, (f"solver.device={device}", "output.save_video=false")
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    cohort = select_eval_cohort(
        dataset,
        goal_offset_steps=cfg.eval.goal_offset_steps,
        num_eval=cfg.eval.num_eval,
        seed=cfg.seed,
    )
    scaler = fit_eval_processors(dataset, cfg.dataset.keys_to_cache)["action"]
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    selected_slots = pilot.slots[: manifest.levels[int(level)].grounded_slots]
    cache_dir = pilot.output_dir / "action_token_grounding"
    rows = []
    for slot in selected_slots:
        episode = int(cohort.episode_ids[int(slot)])
        start = int(cohort.start_steps[int(slot)])
        horizon = int(cfg.plan_config.horizon)
        action_block = int(cfg.plan_config.action_block)
        chunk = dataset.load_chunk(
            np.asarray([episode]),
            np.asarray([start]),
            np.asarray([start + horizon * action_block]),
        )[0]
        expert = normalize_expert_plan(
            chunk, scaler=scaler, horizon=horizon, action_block=action_block
        )
        generator = torch.Generator().manual_seed(manifest.seed + int(slot))
        random = torch.empty(63, *expert.shape).uniform_(-1.0, 1.0, generator=generator)
        panel = torch.cat((expert.unsqueeze(0), random), dim=0).float()
        panel_sha = candidate_actions_sha256(panel)
        identity = {
            "schema_version": 2,
            "manifest_sha256": manifest.sha256,
            "collection_code_sha256": _file_sha256(Path(__file__)),
            "evaluation_sha256": hashlib.sha256(
                repr(OmegaConf.to_container(cfg, resolve=True)).encode("utf-8")
            ).hexdigest(),
            "task": pilot.task,
            "slot": int(slot),
            "episode": episode,
            "start": start,
            "candidate_sha256": panel_sha,
        }
        cache_path = cache_dir / f"slot_{int(slot):02d}.pt"
        if cache_path.is_file():
            grounded = torch.load(cache_path, map_location="cpu", weights_only=False)
            if grounded.get("identity") != identity:
                raise RuntimeError(f"action-token grounding identity mismatch: {cache_path}")
            if _grounding_payload_sha256(grounded) != grounded.get("outcome_sha256"):
                raise RuntimeError(f"action-token grounding content mismatch: {cache_path}")
        else:
            raw = build_prefix_replay_actions(
                panel, scaler=scaler, action_block=action_block, prefix_raw=None
            )
            physical = []
            successes = []
            initial = None
            for offset in range(0, len(raw), int(manifest.replay["simulator_batch_size"])):
                batch_raw = raw[offset : offset + int(manifest.replay["simulator_batch_size"])]
                policy = BlockTrajectoryPolicy(batch_raw, action_block=action_block)
                world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
                world_cfg["num_envs"] = len(batch_raw)
                world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
                world = swm.World(**world_cfg, image_shape=(224, 224))
                world.set_policy(policy)
                try:
                    metrics = world.evaluate(
                        dataset=dataset,
                        episodes_idx=[episode] * len(batch_raw),
                        start_steps=[start] * len(batch_raw),
                        goal_offset=int(cfg.eval.goal_offset_steps),
                        eval_budget=int(batch_raw.shape[1]),
                        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
                        reset_mode="wait",
                    )
                    if initial is None:
                        initial = {
                            key: value[:1]
                            for key, value in policy.initial_info().items()
                        }
                    physical.append(compute_pair_physical_terminal_cost(pilot.task, world.infos))
                    successes.append(
                        np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
                    )
                finally:
                    world.close()
            grounded = {
                "identity": identity,
                "initial": initial,
                "physical": np.concatenate(physical),
                "success": np.concatenate(successes),
            }
            grounded["outcome_sha256"] = _grounding_payload_sha256(grounded)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(grounded, temporary)
            temporary.replace(cache_path)
        planning_info = prepare_image_info(grounded["initial"], transform, device)
        control_cost = _rank_candidate_costs(
            control_model, planning_info, panel, device
        ).numpy()
        token_panel = reshape_action_blocks(panel, source_block=5, target_block=1)
        variant_cost = _rank_candidate_costs(
            variant_model, planning_info, token_panel, device
        ).numpy()
        rows.append({
            "slot": int(slot),
            "candidate_sha256": panel_sha,
            "physical": np.asarray(grounded["physical"]),
            "success": np.asarray(grounded["success"], dtype=bool),
            "control_cost": control_cost,
            "variant_cost": variant_cost,
        })
    return rows


__all__ = [
    "BlockTrajectoryPolicy",
    "build_replay_identity",
    "collect_true_transition_replay",
    "collect_action_token_grounding",
    "concatenate_replays",
]
