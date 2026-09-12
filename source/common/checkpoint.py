from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


CHECKPOINTS_DIRNAME = "checkpoints"
EVAL_DIRNAME = "eval"
EVAL_VIDEOS_DIRNAME = "videos"
LAST_CHECKPOINT_FILENAME = "last.ckpt"


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


def load_initial_model_state(path):
    """Load model weights for an explicit weight-only training continuation.

    A Lightning ``last.ckpt`` restores optimizer and loop state.  Epoch
    snapshots only contain model weights, but are still useful after an
    interrupted run.  Callers should record that this fallback does not
    restore optimizer state.
    """
    payload = torch_load_compat(path, map_location="cpu")
    if isinstance(payload, dict):
        state_dict = payload.get("state_dict", payload)
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Expected a state dict in {path}, got {type(state_dict).__name__}"
            )
        if state_dict and all(str(key).startswith("model.") for key in state_dict):
            state_dict = {
                str(key)[len("model.") :]: value
                for key, value in state_dict.items()
            }
        return state_dict

    model = getattr(payload, "model", None)
    if model is not None and hasattr(model, "state_dict"):
        return model.state_dict()
    if hasattr(payload, "state_dict"):
        return payload.state_dict()
    raise TypeError(f"Expected model weights in {path}, got {type(payload).__name__}")


def _get_swm_cache_dir():
    import stable_worldmodel as swm

    return swm.data.utils.get_cache_dir()


def resolve_policy_path(policy_name, cache_dir=None):
    path = Path(policy_name)
    if not path.is_absolute():
        path = Path(cache_dir if cache_dir is not None else _get_swm_cache_dir(), path)
    return path


def policy_checkpoint_candidates(policy_name, cache_dir=None):
    path = resolve_policy_path(policy_name, cache_dir=cache_dir)
    primary_candidates = []

    if path.suffix == ".ckpt":
        primary_candidates.append(path)
        stem = path.with_suffix("")
        if stem.name.endswith("_policy"):
            stem = stem.with_name(stem.name[: -len("_policy")])
        elif stem.name.endswith("_object"):
            stem = stem.with_name(stem.name[: -len("_object")])
    else:
        stem = path

    primary_candidates.append(stem.with_name(f"{stem.name}_policy.ckpt"))
    primary_candidates.append(stem.with_name(f"{stem.name}_object.ckpt"))
    primary_candidates.append(stem.with_suffix(".ckpt"))

    candidates = []
    for candidate in primary_candidates:
        candidates.append(candidate)
        if candidate.parent.name != CHECKPOINTS_DIRNAME:
            candidates.append(candidate.parent / CHECKPOINTS_DIRNAME / candidate.name)

    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _format_candidates(candidates):
    return "\n".join(str(path) for path in candidates)


def _load_remapped_pretrained(policy_name, cache_dir=None):
    from source.common.remap import load_pretrained_remapped

    try:
        return load_pretrained_remapped(policy_name, cache_dir=cache_dir)
    except ValueError as exc:
        if "Cannot resolve" in str(exc):
            raise FileNotFoundError(str(exc)) from exc
        raise


def _load_training_epoch_weights(checkpoint_path):
    import hydra
    from omegaconf import OmegaConf

    checkpoint_path = Path(checkpoint_path)
    run_dir = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == CHECKPOINTS_DIRNAME
        else checkpoint_path.parent
    )
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training run config not found for epoch weights: {config_path}"
        )

    run_cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.select(run_cfg, "policy.model")
    if model_cfg is None:
        raise ValueError(f"policy.model not found in training config: {config_path}")
    model = hydra.utils.instantiate(model_cfg)
    state_dict = torch_load_compat(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state_dict mapping in {checkpoint_path}, "
            f"got {type(state_dict).__name__}"
        )
    model.load_state_dict(state_dict, strict=True)
    return model


def _is_training_epoch_weights(path):
    name = Path(path).name
    if not name.endswith(".pt"):
        return False
    prefix, marker, epoch = name[:-3].rpartition("_weights_epoch_")
    return bool(prefix and marker and epoch.isdigit())


def _leflow_checkpoint_candidates(policy_name, cache_dir=None):
    """Return direct and stable-worldmodel-cache candidates for LeFlow payloads."""
    raw = Path(policy_name).expanduser()
    if raw.is_absolute():
        roots = ()
    else:
        cache_root = Path(cache_dir if cache_dir is not None else _get_swm_cache_dir())
        roots = (cache_root / CHECKPOINTS_DIRNAME, cache_root)

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        # CLI documentation and experiment manifests commonly pass paths such
        # as ``data/checkpoints/leflow/pusht/latent_planner.pt``.  Keep the
        # caller's direct relative path ahead of the SWM cache fallbacks.
        candidates.append(raw)
        for root in roots:
            candidates.append(root / raw)
            if raw.suffix != ".pt":
                candidates.append(root / f"{raw}.pt")

    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _load_leflow_runtime(policy_name, cache_dir=None):
    """Load a LeFlow runtime when ``policy_name`` names its payload format."""
    name_text = str(policy_name).lower()
    if "leflow" not in name_text and Path(policy_name).name != "latent_planner.pt":
        return None

    from source.model.leflow.latent_planner import (
        LatentPlannerRuntime,
        is_leflow_checkpoint,
    )

    for candidate in _leflow_checkpoint_candidates(policy_name, cache_dir=cache_dir):
        if candidate.is_file() and is_leflow_checkpoint(candidate):
            return LatentPlannerRuntime.from_checkpoint(candidate, device="cpu"), candidate
    return None


def load_policy_or_model(policy_name, cache_dir=None):
    leflow = _load_leflow_runtime(policy_name, cache_dir=cache_dir)
    if leflow is not None:
        return leflow

    if _is_training_epoch_weights(policy_name):
        path = Path(policy_name).expanduser()
        epoch_candidates = [path]
        if not path.is_absolute():
            epoch_candidates.append(resolve_policy_path(path, cache_dir=cache_dir))
        for candidate in epoch_candidates:
            if candidate.is_file():
                candidate = candidate.resolve()
                return _load_training_epoch_weights(candidate), candidate

    candidates = policy_checkpoint_candidates(policy_name, cache_dir=cache_dir)
    for candidate in candidates:
        if candidate.exists():
            return torch_load_compat(candidate), candidate

    try:
        model = _load_remapped_pretrained(policy_name, cache_dir=cache_dir)
        return model, None
    except FileNotFoundError as exc:
        remap_error = exc

    raise FileNotFoundError(
        f"Could not find policy checkpoint or remappable LeWM weights for {policy_name}.\n"
        f"Tried policy/object checkpoints:\n{_format_candidates(candidates)}\n"
        f"Remapped weights fallback failed: {remap_error}"
    )


def get_policy_results_path(policy_name, ckpt_path=None, cache_dir=None):
    if ckpt_path is not None:
        ckpt_dir = Path(ckpt_path).parent
        if ckpt_dir.name == CHECKPOINTS_DIRNAME:
            return ckpt_dir.parent
        return ckpt_dir

    results_path = resolve_policy_path(policy_name, cache_dir=cache_dir).parent
    if results_path.name == CHECKPOINTS_DIRNAME:
        return results_path.parent
    return results_path


def get_policy_eval_paths(policy_name, ckpt_path=None, cache_dir=None):
    results_path = get_policy_results_path(
        policy_name,
        ckpt_path=ckpt_path,
        cache_dir=cache_dir,
    )
    eval_path = results_path / EVAL_DIRNAME
    return eval_path, eval_path / EVAL_VIDEOS_DIRNAME


def training_resume_checkpoint_candidates(run_dir, output_model_name):
    run_dir = Path(run_dir)
    return [
        run_dir / CHECKPOINTS_DIRNAME / LAST_CHECKPOINT_FILENAME,
        run_dir / CHECKPOINTS_DIRNAME / f"{output_model_name}_weights.ckpt",
        run_dir / f"{output_model_name}_weights.ckpt",
    ]


def resolve_training_resume_checkpoint(resume_ckpt, run_dir, output_model_name):
    if resume_ckpt:
        return str(resume_ckpt)

    for candidate in training_resume_checkpoint_candidates(run_dir, output_model_name):
        if candidate.exists():
            return str(candidate)

    return None


class SaveCkptCallback(Callback):
    """Save both the Lightning policy object and the bare JEPA model."""

    def __init__(
        self,
        run_name,
        cfg=None,
        epoch_interval: int = 1,
        output_dir=None,
        epoch_offset: int = 0,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.epoch_offset = int(epoch_offset)

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(
                    pl_module,
                    trainer.current_epoch + 1 + self.epoch_offset,
                )

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(
                    pl_module,
                    trainer.current_epoch + 1 + self.epoch_offset,
                )

    def _save(self, pl_module, epoch):
        if self.output_dir is None:
            output_dir = Path(_get_swm_cache_dir(), CHECKPOINTS_DIRNAME)
        else:
            output_dir = self.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        model = getattr(pl_module, "model", pl_module)
        torch.save(pl_module, output_dir / f"{self.run_name}_policy.ckpt")
        torch.save(model, output_dir / f"{self.run_name}_object.ckpt")
        torch.save(model.state_dict(), output_dir / f"{self.run_name}_weights_epoch_{epoch}.pt")
