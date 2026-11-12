"""Graph-transformer evaluation through the SHARED protocol (common/metrics.py).

For split in {"val", "test"}:
  * input history (context) = the user's TRAIN ratings only (most recent ``max_context``)
  * every book is scored; masked books = ds.exclude[split]
        val : everything rated in train
        test: everything rated in train OR val   (a val-period book is already read at test time)
  * ranking ground truth = ds.targets[split] (warm positives), via common.metrics.ranking_metrics
  * rating metrics on ALL held-out ratings of the split (any stars)
"""
from __future__ import annotations

import numpy as np
import torch

from common.metrics import KS, constant_rating_baselines, popularity_score_fn, ranking_metrics, rating_metrics
from .data import Dataset
from .losses import expected_rating
from .sampling import build_context


def _pad(rows, width, fill):
    out = np.full((len(rows), width), fill, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _encode_users(model, E, ds: Dataset, users, max_context, device):
    """Ego-graph tokens from TRAIN history only."""
    cb, cr, cc = [], [], []
    for u in users:
        books, ratings, _ = ds.train.get(u)
        b, r, rec = build_context(books, ratings, np.arange(len(books)), len(books), max_context)
        cb.append(b); cr.append(r); cc.append(rec)
    L = max(1, max(len(x) for x in cb))
    t = lambda a: torch.as_tensor(a, device=device)
    ctx_mask = t(_pad([np.ones(len(x), np.int64) for x in cb], L, 0)).bool()
    return model.encode_user(E, t(_pad(cb, L, ds.n_items)), t(_pad(cr, L, 0)), t(_pad(cc, L, 0)), ctx_mask)


def make_score_fn(model, ds: Dataset, *, device, max_context: int = 128, cand_chunk: int = 2500):
    """score_fn(users) -> [B, n_items] for common.metrics.ranking_metrics."""
    E = model.books()

    def score_fn(users: np.ndarray) -> torch.Tensor:
        tokens, mask = _encode_users(model, E, ds, users, max_context, device)
        B, n = len(users), ds.n_items
        scores = torch.empty(B, n, device=device)
        for a in range(0, n, cand_chunk):
            cand = torch.arange(a, min(a + cand_chunk, n), device=device).expand(B, -1)
            scores[:, a:a + cand_chunk] = model.score(E, tokens, mask, cand)[0]
        return scores

    return score_fn, E


@torch.no_grad()
def evaluate(model, ds: Dataset, split: str, users: np.ndarray | None = None, *, device,
             ks=KS, max_context: int = 128, batch_users: int = 32, cand_chunk: int = 2500) -> dict:
    assert split in ("val", "test")
    model.eval()
    score_fn, E = make_score_fn(model, ds, device=device, max_context=max_context, cand_chunk=cand_chunk)
    out = ranking_metrics(score_fn, ds.targets[split], ds.exclude[split], ks=ks,
                          batch_size=batch_users, users=users)

    # Rating metrics on every held-out rating (any stars) of the evaluated users.
    held = ds.history(split)
    eval_users = np.arange(ds.n_users) if users is None else np.asarray(users)
    eval_users = eval_users[held.counts()[eval_users] > 0]
    preds, classes, truth = [], [], []
    for s in range(0, len(eval_users), batch_users):
        ub = eval_users[s:s + batch_users]
        tokens, mask = _encode_users(model, E, ds, ub, max_context, device)
        hb = [held.get(u)[0] for u in ub]
        H = max(len(x) for x in hb)
        _, logits = model.score(E, tokens, mask, torch.as_tensor(_pad(hb, H, ds.n_items), device=device))
        p, c = expected_rating(logits).cpu().numpy(), logits.argmax(-1).cpu().numpy() + 1
        for i, u in enumerate(ub):
            n = len(hb[i])
            preds.append(p[i, :n]); classes.append(c[i, :n]); truth.append(held.get(u)[1])
    if truth:
        out.update(rating_metrics(np.concatenate(preds), np.concatenate(truth), np.concatenate(classes)))
    model.train()
    return out


def baselines(ds: Dataset, split: str, users=None, ks=KS) -> dict:
    """MostPop ranking + constant rating baselines, through the same shared functions."""
    pop = ranking_metrics(popularity_score_fn({"train": ds.train_pos}), ds.targets[split],
                          ds.exclude[split], ks=ks, users=users)
    out = {f"pop_{k}": v for k, v in pop.items() if k.startswith(("recall", "ndcg"))}
    out.update(constant_rating_baselines(
        {"train_all": {"rating": ds.train.rating}, f"{split}_rows": {"rating": ds.history(split).rating}},
        split))
    return out
