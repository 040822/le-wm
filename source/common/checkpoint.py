from pathlib import Path

import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


def resolve_policy_path(policy_name, cache_dir=None):
    path = Path(policy_name)
    if not path.is_absolute():
        path = Path(cache_dir or swm.data.utils.get_cache_dir(), path)
    return path


def policy_checkpoint_candidates(policy_name, cache_dir=None):
    path = resolve_policy_path(policy_name, cache_dir=cache_dir)
    candidates = []

    if path.suffix == ".ckpt":
        candidates.append(path)
        stem = path.with_suffix("")
    else:
        stem = path

    if not stem.name.endswith("_policy"):
        candidates.append(stem.with_name(f"{stem.name}_policy.ckpt"))
    if not stem.name.endswith("_object"):
        candidates.append(stem.with_name(f"{stem.name}_object.ckpt"))
    candidates.append(stem.with_suffix(".ckpt"))

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
        return Path(ckpt_path).parent
    return resolve_policy_path(policy_name, cache_dir=cache_dir).parent


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
            output_dir = Path(swm.data.utils.get_cache_dir(), "checkpoints")
        else:
            output_dir = self.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        model = getattr(pl_module, "model", pl_module)
        torch.save(pl_module, output_dir / f"{self.run_name}_policy.ckpt")
        torch.save(model, output_dir / f"{self.run_name}_object.ckpt")
        torch.save(model.state_dict(), output_dir / f"{self.run_name}_weights_epoch_{epoch}.pt")
