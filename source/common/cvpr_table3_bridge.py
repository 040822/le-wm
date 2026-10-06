"""State-indexed candidate schedules for the CVPR Table 3 closed-loop bridge."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEDULE_VERSION = "cvpr_table3_state_indexed_schedule_v1"


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def make_state_indexed_schedules(
    *, task: str, evaluation_seed: int, policy_seed: int
):
    """Return paired action-noise and random-selection schedules.

    Each candidate pool and random choice is keyed by the frozen cohort slot and
    that slot's replan count. Paired policies therefore receive the same noise
    at the same indexed replan, even when their closed-loop states diverge.
    """
    identity = (SCHEDULE_VERSION, str(task), int(evaluation_seed), int(policy_seed))

    def candidate_noise(
        keys,
        *,
        num_candidates: int,
        action_horizon: int,
        action_dim: int,
        device,
        dtype,
    ):
        import torch

        rows = []
        for slot, replan_index in keys:
            generator = torch.Generator(device=device)
            generator.manual_seed(
                _stable_seed(*identity, "candidate_noise", int(slot), int(replan_index))
            )
            rows.append(
                torch.randn(
                    int(num_candidates),
                    int(action_horizon),
                    int(action_dim),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        if not rows:
            raise ValueError("candidate noise schedule requires at least one active state")
        return torch.stack(rows, dim=0)

    def random_selection(keys, *, num_candidates: int, device):
        import numpy as np
        import torch

        indices = [
            int(
                np.random.default_rng(
                    _stable_seed(
                        *identity,
                        "random_selection",
                        int(slot),
                        int(replan_index),
                    )
                ).integers(int(num_candidates))
            )
            for slot, replan_index in keys
        ]
        return torch.as_tensor(indices, dtype=torch.long, device=device)

    metadata = {
        "version": SCHEDULE_VERSION,
        "key": ["task", "evaluation_seed", "policy_seed", "cohort_slot", "replan_index"],
        "candidate_noise_seed_domain": "candidate_noise",
        "random_selection_seed_domain": "random_selection",
    }
    return candidate_noise, random_selection, metadata

