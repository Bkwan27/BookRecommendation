"""Shared evaluation. Every expert and the MoE report metrics through these functions.

ranking_metrics is adapted from Bryan's lightgcn/metrics.py (full ranking over all books,
already-seen books masked). Use it with the masks from common.process.load_processed():

    data = load_processed()
    val  = ranking_metrics(score_fn, data["val"],  data["seen_before_val"])
    test = ranking_metrics(score_fn, data["test"], data["seen_before_test"])   # masks train AND val

score_fn(user_ids: np.ndarray) -> torch.Tensor [len(user_ids), n_items] (higher = better). It must only
use train data as input history.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch

KS = (10, 20, 50)


def ranking_metrics(score_fn: Callable[[np.ndarray], torch.Tensor], target: sp.csr_matrix,
                    exclude: sp.csr_matrix, ks=KS, batch_size: int = 1024,
                    users: np.ndarray | None = None) -> dict[str, float]:
    """Recall/Precision/NDCG/HitRate@K averaged over users with >= 1 target.

    target   held-out positives (data["val"] or data["test"])
    exclude  books that must never be recommended (data["seen_before_val"] / ["seen_before_test"])
    users    optional subset of users to evaluate (e.g. a random sample for fast per-epoch val)
    """
    assert target.shape == exclude.shape
    has_target = np.diff(target.indptr) > 0
    users = np.flatnonzero(has_target) if users is None else np.asarray(users)[has_target[users]]
    kmax = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))
    sums = {f"{m}@{k}": 0.0 for k in ks for m in ("recall", "precision", "ndcg", "hit")}
    for s in range(0, len(users), batch_size):
        b = users[s: s + batch_size]
        scores = score_fn(b).float().clone()
        assert scores.shape == (len(b), target.shape[1]), "score_fn must score every book"
        ex = exclude[b].tocoo()
        r, c = (torch.from_numpy(x).long().to(scores.device) for x in (ex.row, ex.col))
        scores[r, c] = -torch.inf
        top = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()
        tgt = target[b]
        hits = np.take_along_axis(tgt.toarray(), top, axis=1).astype(np.float32)   # (B, kmax)
        n_tgt = np.diff(tgt.indptr).astype(np.float32)
        for k in ks:
            h = hits[:, :k]
            nh = h.sum(1)
            idcg = np.cumsum(discounts[:k])[np.minimum(n_tgt, k).astype(int) - 1]
            sums[f"recall@{k}"] += (nh / n_tgt).sum()
            sums[f"precision@{k}"] += (nh / k).sum()
            sums[f"ndcg@{k}"] += ((h * discounts[:k]).sum(1) / idcg).sum()
            sums[f"hit@{k}"] += (nh > 0).sum()
    out = {m: float(v) / max(len(users), 1) for m, v in sums.items()}
    out["n_users"] = int(len(users))
    return out


def rating_metrics(pred: np.ndarray, rating: np.ndarray, pred_class: np.ndarray | None = None) -> dict:
    """RMSE of predicted (expected) stars and, if given, accuracy of the predicted star class.
    Computed on ALL held-out ratings of the split (data["val_rows"] / data["test_rows"])."""
    pred, rating = np.asarray(pred, np.float64), np.asarray(rating, np.float64)
    out = {"rmse": float(np.sqrt(((pred - rating) ** 2).mean())), "n_ratings": int(len(rating))}
    if pred_class is not None:
        out["rating_acc"] = float((np.asarray(pred_class) == rating).mean())
    return out


def popularity_score_fn(data: dict, device="cpu") -> Callable[[np.ndarray], torch.Tensor]:
    """MostPop baseline: score = number of positive TRAIN interactions (never Goodreads' counters)."""
    pop = torch.as_tensor(np.asarray(data["train"].sum(0)).ravel(), dtype=torch.float32, device=device)
    # clone(): callers such as lightgcn/metrics.evaluate write -inf into the result in place, which
    # would otherwise corrupt the shared popularity vector through the broadcast view.
    return lambda b: pop.expand(len(b), -1).clone()


def constant_rating_baselines(data: dict, split: str) -> dict:
    """Predict the train mean (RMSE) / the train majority star (accuracy)."""
    tr = data["train_all"]["rating"]
    held = data[f"{split}_rows"]["rating"]
    mode = np.bincount(tr).argmax()
    return {"rmse_const_mean": float(np.sqrt(((held - tr.mean()) ** 2).mean())),
            "acc_majority_class": float((held == mode).mean())}
