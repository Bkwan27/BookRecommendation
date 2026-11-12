"""
Top-K evaluation. Every item gets scored, items the user has already seen are
masked out, and then we look at the top K.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch


def evaluate(score_fn: Callable[[np.ndarray], torch.Tensor],
             target: sp.csr_matrix,
             exclude: sp.csr_matrix,
             ks=(10, 20, 50),
             batch_size: int = 4096) -> dict[str, float]:
    """
    score_fn takes an array of user ids and returns a (batch, n_items) score tensor.
    target has the held-out positives for each user.
    exclude has the items that shouldn't be recommended (e.g. train, or train + val).

    Returns recall, precision, ndcg and hit rate at each K, averaged over the
    users who have at least one target item.
    """
    items_per_user = np.diff(target.indptr)
    eval_users = np.flatnonzero(items_per_user)

    max_k = max(ks)
    positions = np.arange(2, max_k + 2)
    discounts = 1.0 / np.log2(positions)

    totals = {}
    for k in ks:
        totals[f"recall@{k}"] = 0.0
        totals[f"precision@{k}"] = 0.0
        totals[f"ndcg@{k}"] = 0.0
        totals[f"hit@{k}"] = 0.0

    for start in range(0, len(eval_users), batch_size):
        batch = eval_users[start:start + batch_size]

        scores = score_fn(batch)
        scores = scores.float()

        # don't recommend things the user already rated
        seen = exclude[batch].tocoo()
        seen_rows = torch.from_numpy(seen.row).long().to(scores.device)
        seen_cols = torch.from_numpy(seen.col).long().to(scores.device)
        scores[seen_rows, seen_cols] = -torch.inf

        top_items = torch.topk(scores, max_k, dim=1).indices
        top_items = top_items.cpu().numpy()

        batch_targets = target[batch]
        target_dense = batch_targets.toarray()

        # hits[i, j] = 1 if the j-th recommendation for user i is a target item
        hits = np.take_along_axis(target_dense, top_items, axis=1)
        hits = hits.astype(np.float32)

        n_targets = np.diff(batch_targets.indptr)
        n_targets = n_targets.astype(np.float32)

        for k in ks:
            hits_k = hits[:, :k]
            n_hits = hits_k.sum(axis=1)

            # ideal DCG: all targets ranked at the top (capped at k)
            n_ideal = np.minimum(n_targets, k).astype(int)
            ideal_dcg_table = np.cumsum(discounts[:k])
            idcg = ideal_dcg_table[n_ideal - 1]
            dcg = (hits_k * discounts[:k]).sum(axis=1)

            totals[f"recall@{k}"] += (n_hits / n_targets).sum()
            totals[f"precision@{k}"] += (n_hits / k).sum()
            totals[f"ndcg@{k}"] += (dcg / idcg).sum()
            totals[f"hit@{k}"] += (n_hits > 0).sum()

    results = {}
    for name, value in totals.items():
        results[name] = float(value) / len(eval_users)
    return results
