from typing import Any, Dict

import torch
from lightning.pytorch import LightningModule


class BasePolicy(LightningModule):
    """Lightweight policy interface shared by current and future algorithms."""

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def make_world_policy(self, *args, **kwargs) -> Any:
        raise NotImplementedError
