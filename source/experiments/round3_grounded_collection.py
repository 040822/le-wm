"""Fixed four-way grounded candidate collection for Round 3.

The public pure helper makes the candidate contract testable without a
simulator.  The runtime collector deliberately keeps the simulator imports
inside the entrypoint so replay planning and validation remain CPU-safe.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_grounded_candidates(
    actor_action: torch.Tensor,
    *,
    count: int = 4,
    perturbation_rms: float = 0.03,
    seed: int = 3072,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return one actor anchor and independent fixed-RMS perturbations.

    ``actor_action`` is normalized and has shape ``[H, A]``.  The returned
    panel keeps the anchor at row zero; every other row is a fresh Gaussian
    direction normalized to the requested per-candidate RMS.  Action-bound
    projection is intentionally a separate runtime step because bounds depend
    on the environment action space.
    """
    action = torch.as_tensor(actor_action).detach().float()
    if action.ndim != 2 or not torch.isfinite(action).all():
        raise ValueError("actor_action must be a finite [H,A] tensor")
    if int(count) != 4:
        raise ValueError("Round3 grounded groups require exactly four candidates")
    rms = float(perturbation_rms)
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError("perturbation_rms must be finite and positive")
    if generator is None:
        generator = torch.Generator(device=action.device).manual_seed(int(seed))
    noise = torch.randn(
        (int(count) - 1, *action.shape),
        device=action.device,
        dtype=action.dtype,
        generator=generator,
    )
    noise_rms = noise.square().mean(dim=(1, 2), keepdim=True).sqrt()
    if not torch.isfinite(noise_rms).all() or (noise_rms <= 0).any():
        raise FloatingPointError("grounded perturbation produced an invalid RMS")
    perturbations = noise * (rms / noise_rms)
    panel = torch.cat((action.unsqueeze(0), action.unsqueeze(0) + perturbations), dim=0)
    if not torch.isfinite(panel).all():
        raise FloatingPointError("grounded candidate panel is non-finite")
    source_label = f"rms_{rms:g}_perturbation"
    sources = ("actor_anchor", source_label, source_label, source_label)
    return panel, sources


def candidate_rms_offsets(panel: torch.Tensor) -> torch.Tensor:
    """Return each candidate's RMS offset from the actor anchor."""
    values = torch.as_tensor(panel).detach().float()
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("panel must have shape [N,H,A]")
    return (values - values[0:1]).square().mean(dim=(1, 2)).sqrt()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _costs_from_info(task: str, info: Mapping[str, Any]) -> np.ndarray:
    from source.diagnostics.stage_b_epoch_pair_ranking import (
        compute_pair_physical_terminal_cost,
    )

    values = compute_pair_physical_terminal_cost(task, dict(info))
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("physical cost contains a non-finite value")
    return values


def _load_group_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported grounded group manifest: {path}")
    return payload


def collect_grounded_replay(
    *,
    task: str,
    checkpoint: str | Path,
    pool_path: str | Path,
    output: str | Path,
    device: str,
    groups: int = 40,
    pool_offset: int = 0,
    seed: int = 3072,
    guidance_mode: str = "none",
    guidance_step_size: float = 0.01,
    guidance_last_steps: int = 5,
    guidance_inner_steps: int = 5,
    guidance_max_rms_offset: float = 0.20,
    collector_version: str = "round3_grounded_v1",
    panel_mode: str = "actor_rms",
    perturbation_rms: float = 0.03,
    model_override: Any | None = None,
    resolved_checkpoint_override: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Collect exactly ``groups`` four-candidate grounded groups.

    Each group uses one deterministic dataset reset, one actor anchor, and
    three RMS=.03 perturbations.  The panel is executed for five action
    blocks before its real successor windows are saved.
    """
    if str(task) not in {"reacher", "pusht"}:
        raise ValueError("grounded Round3 collection only supports reacher and pusht")
    if int(groups) < 1 or int(pool_offset) < 0:
        raise ValueError("groups must be positive and pool_offset must be non-negative")
    if str(panel_mode) not in {"actor_rms", "signed_gradient"}:
        raise ValueError("panel_mode must be actor_rms or signed_gradient")

    # Runtime-only imports keep pure candidate tests independent of EGL.  Bind
    # MuJoCo before importing the environment stack so CUDA and EGL cannot
    # silently select different devices.
    from source.common.round3_eval import validate_gpu_visibility

    validate_gpu_visibility(device)
    if str(device).startswith("cuda"):
        from source.common.gpu_environment import configure_mujoco_egl_device

        configure_mujoco_egl_device()
    from omegaconf import OmegaConf
    import stable_worldmodel as swm

    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import (
        compose_eval_config,
        fit_eval_processors,
        get_dataset,
        img_transform,
    )
    from source.common.round4_action_bounds import (
        compute_normalized_action_bounds,
        project_normalized_actions,
    )
    from source.diagnostics.stage_b_episode_failures import (
        build_prefix_replay_actions,
        prepare_image_info,
    )
    from source.experiments.planner_transition_collection import BlockTrajectoryPolicy
    from source.experiments.round3_phase2 import (
        concatenate_replays,
        save_transition_replay,
    )
    from source.experiments.round3_phase2_collection import ExecutedTransitionRecorder
    from source.model.fast_lewam.jepa import FastLeWAM
    from scripts.round3_phase2 import _online_start_pairs

    checkpoint = Path(checkpoint).resolve()
    pool_path = Path(pool_path).resolve()
    output = Path(output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not pool_path.is_file():
        raise FileNotFoundError(pool_path)
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite grounded replay: {output}")

    cfg = compose_eval_config(
        str(task),
        (f"solver.device={device}", "output.save_video=false"),
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = fit_eval_processors(dataset, cfg.dataset.keys_to_cache)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    scaler = process["action"]
    if model_override is None:
        model, resolved_checkpoint = load_policy_or_model(str(checkpoint))
    else:
        model = model_override
        resolved_checkpoint = (
            Path(resolved_checkpoint_override).resolve()
            if resolved_checkpoint_override is not None
            else checkpoint
        )
    model = getattr(model, "model", model)
    if not isinstance(model, FastLeWAM):
        raise TypeError("grounded collection requires a FastLeWAM checkpoint")
    model = model.to(device).eval()
    model.requires_grad_(False)
    if int(model.action_horizon) != int(cfg.plan_config.horizon):
        raise ValueError("checkpoint action horizon differs from the frozen Round3 config")
    action_block = int(cfg.plan_config.action_block)
    base_action_dim = len(scaler.mean_)
    expected_action_dim = action_block * base_action_dim
    if int(model.action_dim) != expected_action_dim:
        raise ValueError(
            f"model action_dim={model.action_dim} does not match "
            f"action_block*env_action_dim={expected_action_dim}"
        )

    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    candidate_ids = pool["online_episode_ids"][int(pool_offset) :]
    pairs = _online_start_pairs(
        dataset,
        candidate_ids,
        goal_offset=int(cfg.eval.goal_offset_steps),
        seed=int(seed),
        limit=int(groups),
    )
    group_manifest_path = output.with_suffix(output.suffix + ".groups.json")
    report_path = output.with_suffix(output.suffix + ".json")
    prior_group_results: dict[str, dict[str, Any]] = {}
    if resume and report_path.is_file():
        prior_report = json.loads(report_path.read_text(encoding="utf-8"))
        prior_group_results = {
            str(item["group_id"]): dict(item)
            for item in prior_report.get("groups", ())
            if "group_id" in item
        }

    def write_failure(group_id: Any, error: BaseException) -> None:
        _write_json_atomic(
            report_path,
            {
                "schema_version": 1,
                "phase": "round3_grounded_collection",
                "task": str(task),
                "status": "failed",
                "checkpoint": str(resolved_checkpoint or checkpoint),
                "checkpoint_sha256": group_payload["checkpoint_sha256"],
                "group_manifest": str(group_manifest_path),
                "failed_group_id": group_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_groups": group_results,
            },
        )
    group_payload = {
        "schema_version": 1,
        "task": str(task),
        "checkpoint": str(resolved_checkpoint or checkpoint),
        "checkpoint_sha256": _file_sha256(Path(resolved_checkpoint or checkpoint)),
        "pool": str(pool_path),
        "pool_sha256": _file_sha256(pool_path),
        "seed": int(seed),
        "groups": int(groups),
        "pool_offset": int(pool_offset),
        "guidance": {
            "mode": str(guidance_mode),
            "step_size": float(guidance_step_size),
            "last_steps": int(guidance_last_steps),
            "inner_steps": int(guidance_inner_steps),
            "max_rms_offset": float(guidance_max_rms_offset),
        },
        "panel_mode": str(panel_mode),
        "perturbation_rms": float(perturbation_rms),
        "entries": [
            {
                "group_id": f"grounded-{index:04d}",
                "snapshot_id": f"{task}-episode-{episode}-start-{start}",
                "episode_id": episode,
                "start_step": int(start),
            }
            for index, (episode, start) in enumerate(pairs)
        ],
    }
    if resume and group_manifest_path.is_file():
        old_payload = _load_group_manifest(group_manifest_path)
        if old_payload.get("entries") != group_payload["entries"]:
            raise ValueError("grounded group manifest does not match the requested run")
        group_payload = old_payload
    else:
        _write_json_atomic(group_manifest_path, group_payload)

    def group_path(index: int) -> Path:
        return output.with_suffix(output.suffix + f".group_{int(index):04d}.pt")

    class GroundedPanelPolicy(BlockTrajectoryPolicy):
        def __init__(self, *, group_seed: int, panel_config: Mapping[str, Any]):
            dummy = np.zeros(
                (
                    4,
                    int(model.action_horizon) * action_block,
                    base_action_dim,
                ),
                dtype=np.float32,
            )
            self.group_seed = int(group_seed)
            self.panel_config = dict(panel_config)
            self.panel: torch.Tensor | None = None
            self.panel_sources: tuple[str, ...] = ()
            self.panel_sha256: str | None = None
            self.latent_costs: torch.Tensor | None = None
            super().__init__(dummy, action_block=action_block)

        def _build_panel(self, info: Mapping[str, Any]) -> None:
            if "goal" not in info:
                raise ValueError("grounded collection requires goal pixels in simulator info")
            prepared = prepare_image_info(
                {"pixels": info["pixels"], "goal": info["goal"]},
                transform,
                device,
            )
            current = model._last_frame(prepared["pixels"])
            goal = model._last_frame(prepared["goal"])
            with torch.no_grad():
                encoded = model.encode_pixels(torch.cat((current, goal), dim=0))
                z0, goal_latent = encoded.split(len(current))
                generator = torch.Generator(device=device).manual_seed(self.group_seed)
                actor = model.sample_actions(
                    z0,
                    num_steps=int(cfg.get("fast_lewam", {}).get("inference_steps", 10)),
                    generator=generator,
                    goal_latent=goal_latent,
                    guidance_mode=str(self.panel_config["mode"]),
                    guidance_step_size=float(self.panel_config["step_size"]),
                    guidance_last_steps=int(self.panel_config["last_steps"]),
                    guidance_inner_steps=int(self.panel_config["inner_steps"]),
                    guidance_max_rms_offset=float(self.panel_config["max_rms_offset"]),
                )
            if actor.shape[0] != 4:
                raise RuntimeError("grounded group reset did not produce four envs")
            if str(panel_mode) == "signed_gradient":
                from source.experiments.round3_guidance_diagnostics import (
                    make_signed_guidance_candidates,
                )

                with torch.enable_grad():
                    anchor = actor[0:1].detach().requires_grad_(True)
                    anchor_cost = model._latent_cost_from_clean_actions(
                        z0[0:1], goal_latent[0:1], anchor, None
                    )
                    gradient = torch.autograd.grad(anchor_cost.sum(), anchor)[0]
                panel, sources = make_signed_guidance_candidates(
                    actor[0],
                    gradient[0],
                    rms=float(perturbation_rms),
                    seed=self.group_seed + 1,
                )
            else:
                panel, sources = make_grounded_candidates(
                    actor[0],
                    perturbation_rms=float(perturbation_rms),
                    seed=self.group_seed + 1,
                )
            bounds = compute_normalized_action_bounds(
                self.env.single_action_space,
                process["action"],
                action_block=action_block,
            )
            panel = project_normalized_actions(panel, bounds, mode="clip")
            with torch.no_grad():
                self.latent_costs = model.get_cost_from_latents(
                    z0[0:1],
                    goal_latent[0:1],
                    panel.unsqueeze(0).to(device),
                )[0].detach().cpu()
            raw = build_prefix_replay_actions(
                panel,
                scaler=process["action"],
                action_block=action_block,
            )
            self.actions = raw
            self.panel = panel.detach().cpu()
            self.panel_sources = sources
            self.panel_sha256 = _tensor_sha256(self.panel)

        def get_action(self, info: dict[str, Any]) -> np.ndarray:
            if self.panel is None:
                self._build_panel(info)
            return super().get_action(info)

    shards = []
    group_results = []
    for index, entry in enumerate(group_payload["entries"]):
        shard = group_path(index)
        if resume and shard.is_file():
            from source.experiments.round3_phase2 import load_transition_replay

            replay = load_transition_replay(shard)
            shards.append(replay)
            reused = prior_group_results.get(str(entry["group_id"]))
            if reused is None and str(panel_mode) == "signed_gradient":
                raise ValueError(
                    "cannot resume signed-gradient E1 without its prior group metadata"
                )
            if reused is None:
                reused = {
                    "group_id": entry["group_id"],
                    "path": str(shard),
                    "count": replay.count,
                    "replay_sha256": replay.content_sha256(),
                    "status": "reused",
                }
            else:
                reused["status"] = "reused"
            group_results.append(reused)
            continue

        policy = GroundedPanelPolicy(
            group_seed=int(seed) + index * 17,
            panel_config=group_payload["guidance"],
        )
        recorder = ExecutedTransitionRecorder(
            policy,
            episode_ids=[entry["episode_id"]] * 4,
            start_steps=[entry["start_step"]] * 4,
            history_size=int(model.action_horizon) - 2,
            action_block=action_block,
            model_version=f"{resolved_checkpoint or checkpoint}:{group_payload['checkpoint_sha256']}",
            normalize_action=lambda value: torch.as_tensor(
                process["action"].transform(value.numpy())
            ).float(),
            observation_transform=transform["pixels"],
            data_kind="grounded",
            group_ids=[entry["group_id"]] * 4,
            snapshot_ids=[entry["snapshot_id"]] * 4,
            collector_version=str(collector_version),
        )
        world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
        world_cfg["num_envs"] = 4
        world_cfg["max_episode_steps"] = 2 * int(model.action_horizon) * action_block
        world = swm.World(**world_cfg, image_shape=(224, 224))
        world.set_policy(recorder)
        try:
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=[entry["episode_id"]] * 4,
                start_steps=[entry["start_step"]] * 4,
                goal_offset=int(cfg.eval.goal_offset_steps),
                eval_budget=int(model.action_horizon) * action_block,
                callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
                reset_mode="wait",
            )
            if not recorder.initial_physical_info:
                raise RuntimeError("grounded collector did not receive physical reset fields")
            start_costs = _costs_from_info(task, recorder.initial_physical_info)
            terminal_costs = _costs_from_info(task, world.infos)
            successes = np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
            if len(start_costs) != 4 or len(terminal_costs) != 4 or len(successes) != 4:
                raise RuntimeError("grounded world returned an invalid four-way result")
            outcomes = {
                (entry["episode_id"], env_index): {
                    "success": bool(successes[env_index]),
                    "physical_start_cost": float(start_costs[env_index]),
                    "physical_terminal_cost": float(terminal_costs[env_index]),
                }
                for env_index in range(4)
            }
            replay = recorder.replay(source="online", outcomes=outcomes)
            save_transition_replay(shard, replay)
            group_results.append(
                {
                    "group_id": entry["group_id"],
                    "snapshot_id": entry["snapshot_id"],
                    "path": str(shard),
                    "count": replay.count,
                    "environment_steps": recorder.environment_steps,
                    "candidate_panel_sha256": policy.panel_sha256,
                    "candidate_sources": list(policy.panel_sources),
                    "requested_perturbation_rms": float(perturbation_rms),
                    "candidate_rms_offsets": candidate_rms_offsets(policy.panel).tolist(),
                    "candidate_latent_costs": policy.latent_costs.tolist(),
                    "successes": successes.tolist(),
                    "physical_start_costs": start_costs.tolist(),
                    "physical_terminal_costs": terminal_costs.tolist(),
                    "replay_sha256": replay.content_sha256(),
                    "status": "ok",
                }
            )
            shards.append(replay)
        except Exception as error:
            write_failure(entry["group_id"], error)
            raise
        finally:
            world.close()

    if not shards:
        raise RuntimeError("grounded collection produced no replay shards")
    replay = concatenate_replays(shards)
    save_transition_replay(output, replay)
    report = {
        "schema_version": 1,
        "phase": "round3_grounded_collection",
        "task": str(task),
        "status": "ok",
        "checkpoint": str(resolved_checkpoint or checkpoint),
        "checkpoint_sha256": group_payload["checkpoint_sha256"],
        "group_manifest": str(group_manifest_path),
        "group_manifest_sha256": _file_sha256(group_manifest_path),
        "replay": {
            "path": str(output),
            "sha256": replay.content_sha256(),
            "count": replay.count,
            "data_kind": "grounded",
        },
        "groups": group_results,
        "environment_steps": int(
            sum(int(item.get("environment_steps", 0)) for item in group_results)
        ),
        "collector_version": str(collector_version),
    }
    _write_json_atomic(output.with_suffix(output.suffix + ".json"), report)
    return report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "candidate_rms_offsets",
    "collect_grounded_replay",
    "make_grounded_candidates",
]
