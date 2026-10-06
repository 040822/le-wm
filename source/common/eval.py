import inspect
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


LEWM_EVAL_PROTOCOL = "lewm_upstream_v1"
_ROOT = Path(__file__).resolve().parents[2]
_FAST_LEWAM_STAGES = {
    "stage_a",
    "stage_a_shuffled_goal",
    "stage_b",
    "stage_b_actor_warm_start",
    "stage_c",
}


def _validate_eval_config(cfg):
    """Reject incompatible protocol parameters before loading a model or world."""
    positive = {
        "eval.num_eval": cfg.eval.num_eval,
        "eval.eval_budget": cfg.eval.eval_budget,
        "plan_config.horizon": cfg.plan_config.horizon,
        "plan_config.receding_horizon": cfg.plan_config.receding_horizon,
        "plan_config.action_block": cfg.plan_config.action_block,
    }
    for name, value in positive.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if int(cfg.eval.goal_offset_steps) < 0:
        raise ValueError("eval.goal_offset_steps must be non-negative")
    if int(cfg.plan_config.receding_horizon) > int(cfg.plan_config.horizon):
        raise ValueError("plan_config.receding_horizon cannot exceed horizon")
    planned_steps = int(cfg.plan_config.horizon) * int(
        cfg.plan_config.action_block
    )
    if planned_steps > int(cfg.eval.eval_budget):
        raise ValueError(
            "plan_config.horizon * action_block cannot exceed eval.eval_budget"
        )
    if int(cfg.world.num_envs) != int(cfg.eval.num_eval):
        raise ValueError("world.num_envs must equal eval.num_eval")


def compose_eval_config(config_name, overrides=()):
    """Compose one task's canonical eval config plus Hydra-style dotlist overrides."""
    config_path = _ROOT / "config" / "eval" / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"evaluation config not found: {config_path}")
    task = OmegaConf.load(config_path)
    task.pop("defaults", None)
    launcher = OmegaConf.load(_ROOT / "config" / "eval" / "launcher" / "local.yaml")
    launcher.pop("defaults", None)
    solver = OmegaConf.load(_ROOT / "config" / "eval" / "solver" / "cem.yaml")
    cfg = OmegaConf.merge(launcher, task, {"solver": solver})
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    OmegaConf.resolve(cfg)
    _validate_eval_config(cfg)
    return cfg


@dataclass(frozen=True)
class EvaluationCohort:
    """A reproducible set of dataset-backed evaluation start states."""

    row_indices: np.ndarray
    episode_ids: np.ndarray
    start_steps: np.ndarray


@dataclass(frozen=True)
class EpisodeEvaluation:
    """Observable outcome and provenance for one evaluation environment slot."""

    slot: int
    dataset_episode: Any
    start_step: int
    success: bool
    video: str | None = None
    goal_source_slot: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Canonical result shared by every LeWM-compatible evaluation entrypoint."""

    task: str
    local_dataset: str
    benchmark_dataset: str
    entrypoint: str
    policy_kind: str
    checkpoint: str | None
    epoch: int | None
    stage: str | None
    parameters: dict[str, Any]
    evaluation_seconds: float
    success_rate: float
    episodes: tuple[EpisodeEvaluation, ...]
    protocol: str = LEWM_EVAL_PROTOCOL
    schema_version: int = 1
    status: str = "ok"


@dataclass(frozen=True)
class EvaluationIdentity:
    """Caller-owned identity fields attached to a canonical evaluation."""

    entrypoint: str
    policy_kind: str
    checkpoint: str | None = None
    epoch: int | None = None
    stage: str | None = None
    actor_warm_start: bool = False
    guidance_mode: str = "none"
    guidance_step_size: float = 0.01
    guidance_last_steps: int = 5
    guidance_inner_steps: int = 5
    guidance_max_rms_offset: float = 0.20


def _jsonable(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_evaluation_artifacts(result, output_dir):
    """Atomically write the canonical JSON result and its readable text view."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(asdict(result))
    json_path = output_dir / "result.json"
    json_tmp = output_dir / "result.json.tmp"
    with json_tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    json_tmp.replace(json_path)

    lines = [
        f"evaluation_seconds: {result.evaluation_seconds:.6f}",
        f"success_rate: {result.success_rate:.6f}",
        "episodes:",
    ]
    for episode in result.episodes:
        line = (
            f"  slot={episode.slot} episode={episode.dataset_episode} "
            f"start={episode.start_step} success={str(episode.success).lower()}"
        )
        if episode.goal_source_slot is not None:
            line += f" goal_source_slot={episode.goal_source_slot}"
        if episode.video is not None:
            line += f" video={episode.video}"
        lines.append(line)
    text_path = output_dir / "metrics.txt"
    text_tmp = output_dir / "metrics.txt.tmp"
    text_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    text_tmp.replace(text_path)
    for stale in (output_dir / "failure.json", output_dir / "failure.txt"):
        if stale.exists():
            stale.unlink()


def write_evaluation_failure(output_dir, error, traceback_text):
    """Write matching machine- and human-readable failure artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": LEWM_EVAL_PROTOCOL,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
    }
    json_tmp = output_dir / "failure.json.tmp"
    with json_tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    json_tmp.replace(output_dir / "failure.json")
    text_tmp = output_dir / "failure.txt.tmp"
    text_tmp.write_text(traceback_text, encoding="utf-8")
    text_tmp.replace(output_dir / "failure.txt")


def _publish_evaluation_directory(temporary, target):
    """Replace one result leaf while restoring the prior snapshot on failure."""
    temporary = Path(temporary)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.backup-", dir=target.parent)
        )
        backup.rmdir()
        target.replace(backup)
    try:
        temporary.replace(target)
    except Exception:
        if backup is not None and backup.exists():
            backup.replace(target)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def select_eval_cohort(dataset, *, goal_offset_steps, num_eval, seed):
    """Reproduce the official LeWM dataset-start sampling protocol.

    The upstream evaluator samples positions from ``len(valid_indices) - 1``.
    That global-last-row exclusion is retained intentionally for benchmark
    compatibility and is versioned by :data:`LEWM_EVAL_PROTOCOL`.
    """
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = np.unique(dataset.get_col_data(col_name))
    episode_lengths = get_episodes_length(dataset, episode_ids)
    max_start = episode_lengths - int(goal_offset_steps) - 1
    limits = dict(zip(episode_ids, max_start))
    per_row_limit = np.array(
        [limits[value] for value in dataset.get_col_data(col_name)]
    )
    valid_indices = np.nonzero(
        dataset.get_col_data("step_idx") <= per_row_limit
    )[0]
    candidate_count = len(valid_indices) - 1
    if candidate_count < int(num_eval):
        raise ValueError(
            f"LeWM evaluation needs {num_eval} starts, only "
            f"{max(candidate_count, 0)} upstream-compatible candidates available"
        )
    positions = np.random.default_rng(int(seed)).choice(
        candidate_count,
        size=int(num_eval),
        replace=False,
    )
    rows = np.sort(valid_indices[positions])
    selected = dataset.get_row_data(rows)
    return EvaluationCohort(
        row_indices=rows,
        episode_ids=np.asarray(selected[col_name]),
        start_steps=np.asarray(selected["step_idx"]),
    )


def validate_evaluation_cohort(dataset, cohort, *, num_eval):
    "Validate and normalize a caller-supplied evaluation cohort."
    if not isinstance(cohort, EvaluationCohort):
        raise TypeError("cohort must be an EvaluationCohort")
    expected = int(num_eval)
    if expected < 1:
        raise ValueError("num_eval must be positive")
    rows = np.asarray(cohort.row_indices)
    episodes = np.asarray(cohort.episode_ids)
    starts = np.asarray(cohort.start_steps)
    for name, value in (("row_indices", rows), ("episode_ids", episodes), ("start_steps", starts)):
        if value.ndim != 1:
            raise ValueError(f"cohort.{name} must be one-dimensional")
        if len(value) != expected:
            raise ValueError(f"cohort.{name} has length {len(value)}; expected {expected}")
    if not np.issubdtype(rows.dtype, np.integer):
        if not np.all(np.isfinite(rows)) or not np.equal(rows, np.floor(rows)).all():
            raise ValueError("cohort.row_indices must contain integers")
        rows = rows.astype(np.int64)
    else:
        rows = rows.astype(np.int64, copy=True)
    if len(np.unique(rows)) != len(rows):
        raise ValueError("cohort.row_indices must not contain duplicates")
    try:
        column_names = tuple(dataset.column_names)
    except AttributeError as exc:
        raise TypeError("dataset must expose column_names") from exc
    episode_column = "episode_idx" if "episode_idx" in column_names else "ep_idx"
    if episode_column not in column_names or "step_idx" not in column_names:
        raise ValueError("dataset must expose episode_idx/ep_idx and step_idx")
    dataset_episodes = np.asarray(dataset.get_col_data(episode_column))
    dataset_steps = np.asarray(dataset.get_col_data("step_idx"))
    if dataset_episodes.ndim != 1 or dataset_steps.ndim != 1:
        raise ValueError("dataset episode and step columns must be one-dimensional")
    if len(dataset_episodes) != len(dataset_steps):
        raise ValueError("dataset episode and step columns have different lengths")
    if np.any(rows < 0) or np.any(rows >= len(dataset_steps)):
        raise ValueError("cohort.row_indices contain rows outside the dataset")
    if not np.array_equal(dataset_episodes[rows], episodes):
        raise ValueError("cohort.episode_ids do not match cohort.row_indices")
    if not np.array_equal(dataset_steps[rows], starts):
        raise ValueError("cohort.start_steps do not match cohort.row_indices")
    return EvaluationCohort(
        row_indices=rows,
        episode_ids=np.array(episodes, copy=True),
        start_steps=np.array(starts, copy=True),
    )


class DatasetEvaluationSession:
    """Cache one LeWM-compatible dataset cohort and evaluate policies against it."""

    def __init__(
        self,
        cfg,
        *,
        task,
        dataset=None,
        world_factory=None,
        cohort: EvaluationCohort | None = None,
    ):
        _validate_eval_config(cfg)
        self.cfg = cfg
        self.task = str(task)
        self.dataset = dataset or get_dataset(cfg, cfg.eval.dataset_name)
        self.world_factory = world_factory or swm.World
        self.cohort = (
            validate_evaluation_cohort(
                self.dataset, cohort, num_eval=cfg.eval.num_eval
            )
            if cohort is not None
            else select_eval_cohort(
                self.dataset,
                goal_offset_steps=cfg.eval.goal_offset_steps,
                num_eval=cfg.eval.num_eval,
                seed=cfg.seed,
            )
        )
        self.process = fit_eval_processors(
            self.dataset,
            cfg.dataset.keys_to_cache,
            normalizer_stats=OmegaConf.select(
                cfg, "eval.action_normalizer_stats", default=None
            ),
        )
        self.transform = {
            "pixels": img_transform(cfg),
            "goal": img_transform(cfg),
        }

    def _build_policy(self, policy_or_model, identity, device):
        if isinstance(policy_or_model, swm.policy.BasePolicy):
            return policy_or_model

        from source.model.fast_lewam.jepa import FastLeWAM
        from source.model.leflow.latent_planner import LatentPlannerRuntime
        from source.policy.dispatch import make_world_policy
        from source.policy.fast_lewam_eval import make_fast_lewam_policy
        from source.policy.leflow import make_leflow_policy

        model = getattr(policy_or_model, "model", policy_or_model)
        if isinstance(model, LatentPlannerRuntime):
            solver_cfg = self.cfg.solver
            target = str(OmegaConf.select(solver_cfg, "_target_", default=""))
            if not target.endswith("LearnedLatentPathSolver"):
                solver_cfg = OmegaConf.load(
                    _ROOT / "config" / "eval" / "solver" / "latent_flow.yaml"
                )
            solver_cfg.device = str(device)
            policy_seed = OmegaConf.select(self.cfg, "eval.policy_seed")
            if policy_seed is None:
                policy_seed = self.cfg.seed
            solver_cfg.seed = int(policy_seed)
            # The runtime is already loaded, so an unresolved `${policy}` in
            # the standalone solver config must not be evaluated here.
            solver_cfg.checkpoint = None
            self.cfg.solver = solver_cfg
            return make_leflow_policy(
                model,
                solver_cfg=solver_cfg,
                plan_config=self.cfg.plan_config,
                process=self.process,
                transform=self.transform,
                device=device,
            )
        if isinstance(model, FastLeWAM):
            if identity.stage not in _FAST_LEWAM_STAGES:
                raise ValueError(
                    "Fast-LeWAM evaluation requires one of "
                    f"{sorted(_FAST_LEWAM_STAGES)}"
                )
            if int(model.action_horizon) != int(self.cfg.plan_config.horizon):
                raise ValueError(
                    f"model action_horizon={model.action_horizon} does not match "
                    f"plan_config.horizon={self.cfg.plan_config.horizon}"
                )
            shuffled = identity.stage == "stage_a_shuffled_goal"
            actor_warm_start = (
                identity.stage == "stage_b_actor_warm_start"
                or bool(identity.actor_warm_start)
            )
            if shuffled and int(self.cfg.eval.num_eval) < 2:
                raise ValueError("shuffled-goal evaluation requires num_eval >= 2")
            if shuffled:
                mode = "stage_a"
            elif actor_warm_start:
                mode = "stage_b"
            else:
                mode = identity.stage
            inference_steps = self.cfg.get("fast_lewam", {}).get(
                "inference_steps", None
            )
            return make_fast_lewam_policy(
                policy_or_model,
                solver_cfg=self.cfg.solver,
                plan_config=self.cfg.plan_config,
                process=self.process,
                transform=self.transform,
                device=device,
                mode=mode,
                inference_steps=inference_steps,
                seed=int(
                    OmegaConf.select(
                        self.cfg,
                        "eval.policy_seed",
                        default=self.cfg.seed,
                    )
                ),
                goal_mode="cyclic_shift" if shuffled else "correct",
                actor_warm_start=actor_warm_start,
                guidance_mode=str(identity.guidance_mode),
                guidance_step_size=float(identity.guidance_step_size),
                guidance_last_steps=int(identity.guidance_last_steps),
                guidance_inner_steps=int(identity.guidance_inner_steps),
                guidance_max_rms_offset=float(identity.guidance_max_rms_offset),
            )
        if identity.stage is not None:
            raise ValueError("stage is only valid for Fast-LeWAM evaluation")
        return make_world_policy(
            policy_or_model,
            solver_cfg=self.cfg.solver,
            plan_config=self.cfg.plan_config,
            process=self.process,
            transform=self.transform,
            device=device,
        )

    def evaluate(self, policy_or_model, *, identity, output_dir, device=None):
        """Run one policy/stage and write its canonical result artifacts."""
        device = device or self.cfg.solver.get("device", "cuda")
        policy = self._build_policy(policy_or_model, identity, str(device))
        world_cfg = OmegaConf.to_container(self.cfg.world, resolve=True)
        world_cfg["max_episode_steps"] = 2 * int(self.cfg.eval.eval_budget)
        save_video = bool(self.cfg.get("output", {}).get("save_video", True))
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_context = tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}.pending-",
            dir=output_dir.parent,
        )
        artifact_dir = Path(temporary_context.name)
        try:
            video_dir = artifact_dir / "videos"
            if save_video:
                video_dir.mkdir(parents=True, exist_ok=True)
            world = self.world_factory(**world_cfg, image_shape=(224, 224))
            try:
                world.set_policy(policy)
                started = time.perf_counter()
                metrics = evaluate_from_dataset_compat(
                    world=world,
                    dataset=self.dataset,
                    eval_start_idx=self.cohort.start_steps,
                    eval_episodes=self.cohort.episode_ids,
                    cfg=self.cfg,
                    video_path=video_dir,
                    save_video=save_video,
                )
                evaluation_seconds = time.perf_counter() - started
            finally:
                if hasattr(world, "close"):
                    world.close()
                else:
                    env = getattr(world, "envs", None)
                    if env is None:
                        env = getattr(world, "env", None)
                    if env is not None and hasattr(env, "close"):
                        env.close()

            successes = np.asarray(
                metrics["episode_successes"], dtype=bool
            ).reshape(-1)
            if len(successes) != len(self.cohort.row_indices):
                raise ValueError(
                    "evaluation returned a different number of episode outcomes "
                    "than starts"
                )
            goal_sources = [None] * len(successes)
            if identity.stage == "stage_a_shuffled_goal":
                offset = 1 + int(self.cfg.seed) % (len(successes) - 1)
                goal_sources = (
                    (np.arange(len(successes)) + offset) % len(successes)
                ).tolist()
            episodes = tuple(
                EpisodeEvaluation(
                    slot=index,
                    dataset_episode=_jsonable(self.cohort.episode_ids[index]),
                    start_step=int(self.cohort.start_steps[index]),
                    success=bool(successes[index]),
                    video=f"videos/env_{index}.mp4" if save_video else None,
                    goal_source_slot=goal_sources[index],
                )
                for index in range(len(successes))
            )
            parameters = {
                "seed": int(self.cfg.seed),
                "num_eval": int(self.cfg.eval.num_eval),
                "goal_offset_steps": int(self.cfg.eval.goal_offset_steps),
                "eval_budget": int(self.cfg.eval.eval_budget),
                "start_rows": self.cohort.row_indices.tolist(),
                "episode_ids": _jsonable(self.cohort.episode_ids),
                "start_steps": self.cohort.start_steps.tolist(),
                "plan_config": OmegaConf.to_container(
                    self.cfg.plan_config, resolve=True
                ),
                "solver": OmegaConf.to_container(
                    self.cfg.solver, resolve=True
                ),
                "save_video": save_video,
            }
            model = getattr(policy_or_model, "model", policy_or_model)
            if getattr(model, "policy_kind", None) == "subjepa_official":
                parameters["model_provenance"] = dict(model.provenance)
            actor_warm_start = (
                identity.stage == "stage_b_actor_warm_start"
            )
            if identity.stage in {
                "stage_b",
                "stage_b_actor_warm_start",
            }:
                parameters["actor_warm_start"] = actor_warm_start
            if identity.stage in {
                "stage_a",
                "stage_a_shuffled_goal",
                "stage_b_actor_warm_start",
                "stage_c",
            }:
                parameters["inference_steps"] = int(
                    self.cfg.get("fast_lewam", {}).get("inference_steps")
                    or model.inference_steps
                )
            if actor_warm_start:
                parameters["actor_seed"] = int(self.cfg.seed)
            result = EvaluationResult(
                task=self.task,
                local_dataset=str(self.cfg.eval.dataset_name),
                benchmark_dataset=str(
                    self.cfg.eval.get(
                        "benchmark_dataset_name", self.cfg.eval.dataset_name
                    )
                ),
                entrypoint=identity.entrypoint,
                policy_kind=identity.policy_kind,
                checkpoint=identity.checkpoint,
                epoch=identity.epoch,
                stage=identity.stage,
                parameters=parameters,
                evaluation_seconds=float(evaluation_seconds),
                success_rate=float(metrics["success_rate"]),
                episodes=episodes,
            )
            write_evaluation_artifacts(result, artifact_dir)
            _publish_evaluation_directory(artifact_dir, output_dir)
            return result
        finally:
            temporary_context.cleanup()


def img_transform(cfg):
    """Build eval-time image transforms for current observations and goals."""
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    local_path = Path(dataset_name).expanduser()
    if local_path.is_file() and local_path.suffix in {".h5", ".lance"}:
        return swm.data.HDF5Dataset(
            path=local_path,
            keys_to_cache=cfg.dataset.keys_to_cache,
            cache_dir=dataset_path,
        )
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def fit_eval_processors(dataset, keys_to_cache, *, normalizer_stats=None):
    process = {}
    for col in keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    if normalizer_stats:
        for col, stats in normalizer_stats.items():
            if col not in process:
                raise ValueError(
                    f"training normalizer was provided for uncached eval column {col!r}"
                )
            processor = process[col]
            mean = np.asarray(stats["mean"], dtype=np.float64).reshape(-1)
            scale = np.asarray(stats["std"], dtype=np.float64).reshape(-1)
            if mean.shape != scale.shape or not np.all(np.isfinite(mean)):
                raise ValueError(f"invalid training normalizer stats for {col!r}")
            if not np.all(np.isfinite(scale)) or np.any(scale == 0):
                raise ValueError(f"invalid training normalizer scale for {col!r}")
            if processor.n_features_in_ != len(mean):
                raise ValueError(
                    f"training normalizer width for {col!r} does not match eval data"
                )
            processor.mean_ = mean
            processor.scale_ = scale
            processor.var_ = scale**2
            processor.n_samples_seen_ = int(stats.get("sample_count", 1))

    return process


def call_supported_kwargs(fn, **kwargs):
    params = inspect.signature(fn).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return fn(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in params}
    return fn(**supported)


def evaluate_from_dataset_compat(
    world,
    dataset,
    eval_start_idx,
    eval_episodes,
    cfg,
    video_path,
    save_video=True,
):
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    common_kwargs = {
        "dataset": dataset,
        "start_steps": eval_start_idx.tolist(),
        "episodes_idx": eval_episodes.tolist(),
        "eval_budget": cfg.eval.eval_budget,
        "callables": callables,
    }

    evaluate_params = inspect.signature(world.evaluate).parameters
    evaluate_accepts_dataset = "dataset" in evaluate_params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in evaluate_params.values()
    )
    if evaluate_accepts_dataset:
        return call_supported_kwargs(
            world.evaluate,
            **common_kwargs,
            goal_offset=cfg.eval.goal_offset_steps,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            video=video_path if save_video else None,
            video_path=video_path if save_video else None,
            save_video=save_video,
        )

    if hasattr(world, "evaluate_from_dataset"):
        return call_supported_kwargs(
            world.evaluate_from_dataset,
            **common_kwargs,
            goal_offset=cfg.eval.goal_offset_steps,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            video=video_path if save_video else None,
            video_path=video_path if save_video else None,
            save_video=save_video,
        )

    raise TypeError(
        "当前 stable_worldmodel 版本既不支持 world.evaluate(dataset=...)，"
        "也没有 world.evaluate_from_dataset(...)；请升级 stable-worldmodel，"
        "或使用支持数据集驱动评测的版本。"
    )
