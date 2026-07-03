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


def load_policy_or_model(policy_name, cache_dir=None):
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

    def __init__(self, run_name, cfg=None, epoch_interval: int = 1, output_dir=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module, trainer.current_epoch + 1)

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
