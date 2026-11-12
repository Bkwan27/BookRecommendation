"""The contract every expert implements so the MoE router can combine them.

An expert is anything that, given user ids, scores every book from TRAIN history only:

    class MyExpert:
        name = "lightgcn"
        def score_all(self, users: np.ndarray) -> torch.Tensor:          # [B, n_items] ranking scores
        def rating_logits(self, users, items) -> torch.Tensor | None:    # [B, C, 5] P(1..5 stars), optional

Ranking scores from different experts live on different scales (BPR only constrains differences
within a user), so the router mixes per-user standardised scores; see standardize_per_user().
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Expert(Protocol):
    name: str

    def score_all(self, users: np.ndarray) -> torch.Tensor:
        """Ranking scores for every book, shape [len(users), n_items]. Train history only."""
        ...

    def rating_logits(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor | None:
        """Star logits [B, C, 5] for candidate items [B, C], or None if the expert has no rating head."""
        ...


def standardize_per_user(scores: torch.Tensor, mask: torch.Tensor | None = None,
                         eps: float = 1e-6) -> torch.Tensor:
    """z-score each row (user) over its candidates, ignoring masked-out entries (mask=False).
    Makes scores from different experts comparable before the router mixes them."""
    if mask is None:
        mask = torch.isfinite(scores)
    m = mask.float()
    n = m.sum(-1, keepdim=True).clamp(min=1)
    x = torch.where(mask, scores, torch.zeros_like(scores))
    mean = x.sum(-1, keepdim=True) / n
    var = (((x - mean) * m) ** 2).sum(-1, keepdim=True) / n
    return torch.where(mask, (scores - mean) / (var.sqrt() + eps), scores)
