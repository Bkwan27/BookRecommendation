"""Full-ranking top-K evaluation (all items scored, already-seen items masked)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch


def evaluate(score_fn: Callable[[np.ndarray], torch.Tensor], target: sp.csr_matrix,
             exclude: sp.csr_matrix, ks=(10, 20, 50), batch_size: int = 4096) -> dict[str, float]:
    """score_fn(user_ids) -> (B, n_items) scores. `target` holds the held-out positives,
    `exclude` the interactions that must not be recommended (e.g. train, or train+val).
    Returns Recall@K, Precision@K, NDCG@K, HitRate@K averaged over users with >=1 target."""
    users = np.flatnonzero(np.diff(target.indptr))
    kmax = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))
    sums = {f"{m}@{k}": 0.0 for k in ks for m in ("recall", "precision", "ndcg", "hit")}
    for s in range(0, len(users), batch_size):
        b = users[s: s + batch_size]
        scores = score_fn(b).float()
        ex = exclude[b].tocoo()
        r, c = (torch.from_numpy(x).long().to(scores.device) for x in (ex.row, ex.col))
        scores[r, c] = -torch.inf
        top = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()
        tgt = target[b]
        hits = np.take_along_axis(tgt.toarray(), top, axis=1).astype(np.float32)  # (B, kmax)
        n_tgt = np.diff(tgt.indptr).astype(np.float32)
        for k in ks:
            h = hits[:, :k]
            nh = h.sum(1)
            idcg = np.cumsum(discounts[:k])[np.minimum(n_tgt, k).astype(int) - 1]
            sums[f"recall@{k}"] += (nh / n_tgt).sum()
            sums[f"precision@{k}"] += (nh / k).sum()
            sums[f"ndcg@{k}"] += ((h * discounts[:k]).sum(1) / idcg).sum()
            sums[f"hit@{k}"] += (nh > 0).sum()
    return {m: float(v) / len(users) for m, v in sums.items()}
