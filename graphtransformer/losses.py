"""Objective for the graph-transformer expert.

    L = L_BPR + w_graded * L_graded + lambda * L_rating

L_BPR    personalised BPR (CS224W lec. 11): for each user, every positive should
         outscore every negative. Per-user mean, then mean over users.
           positives: target edges rated >= pos_threshold (4-5 stars) and to-read books
           negatives: random unrated books, PinSAGE-style hard negatives, and
                      target edges rated <= dislike_threshold (1-2 stars, weight w_dislike):
                      read-and-disliked books are the hardest, most trustworthy negatives.
L_graded within-user pairwise ordering on targets, e.g. 5 stars > 3 stars
         (pairs whose rating gap is >= graded_min_gap).
L_rating auxiliary rating prediction on observed target edges: 5-class cross-entropy
         ("ce") or MSE of the expected rating ("mse").
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

STARS = torch.arange(1, 6, dtype=torch.float32)


@dataclass
class LossConfig:
    pos_threshold: int = 4
    dislike_threshold: int = 2
    w_dislike: float = 1.0
    w_graded: float = 0.5
    graded_min_gap: int = 2
    rating_weight: float = 0.5       # lambda
    rating_loss: str = "ce"          # "ce" | "mse"


def _per_user_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """loss/weight: [B, ...]. Weighted mean within each user, then mean over users with pairs."""
    w = weight.flatten(1)
    num = (loss.flatten(1) * w).sum(1)
    den = w.sum(1)
    has = den > 0
    if not has.any():
        return loss.sum() * 0.0
    return (num[has] / den[has]).mean()


def bpr_loss(pos_score, pos_mask, neg_score, neg_weight):
    """pos_score [B,P], pos_mask [B,P], neg_score [B,N], neg_weight [B,N] (0 = invalid)."""
    diff = pos_score.unsqueeze(2) - neg_score.unsqueeze(1)          # [B,P,N]
    w = pos_mask.unsqueeze(2).float() * neg_weight.unsqueeze(1)
    return _per_user_mean(F.softplus(-diff), w)


def graded_loss(score, rating, min_gap: int):
    """Within-user pairs (i, j) with rating_i - rating_j >= min_gap; rating 0 = padding."""
    valid = rating > 0
    gap = rating.unsqueeze(2) - rating.unsqueeze(1)                  # [B,T,T]
    w = ((gap >= min_gap) & valid.unsqueeze(2) & valid.unsqueeze(1)).float()
    diff = score.unsqueeze(2) - score.unsqueeze(1)
    return _per_user_mean(F.softplus(-diff), w)


def rating_loss(logits, rating, kind: str = "ce"):
    """logits [B,T,5], rating [B,T] in 1..5 (0 = padding)."""
    valid = rating > 0
    if not valid.any():
        return logits.sum() * 0.0
    lg, r = logits[valid], rating[valid]
    if kind == "ce":
        return F.cross_entropy(lg, r - 1)
    if kind == "mse":
        return F.mse_loss(expected_rating(lg), r.float())
    raise ValueError(kind)


def expected_rating(logits: torch.Tensor) -> torch.Tensor:
    return (logits.softmax(-1) * STARS.to(logits.device)).sum(-1)


def objective(score, logits, batch, cfg: LossConfig) -> tuple[torch.Tensor, dict]:
    """``score``/``logits`` are over ``batch.cand_book`` = [targets | to-read | negatives]."""
    T, R = batch.tgt_book.shape[1], batch.read_book.shape[1]
    s_t, s_r, s_n = score[:, :T], score[:, T:T + R], score[:, T + R:]
    rt = batch.tgt_rating

    pos_score = torch.cat([s_t, s_r], 1)
    pos_mask = torch.cat([rt >= cfg.pos_threshold, batch.read_mask], 1)
    disliked = (rt > 0) & (rt <= cfg.dislike_threshold)
    neg_score = torch.cat([s_n, s_t], 1)
    neg_weight = torch.cat([batch.neg_mask.float(), disliked.float() * cfg.w_dislike], 1)

    l_bpr = bpr_loss(pos_score, pos_mask, neg_score, neg_weight)
    l_graded = graded_loss(s_t, rt, cfg.graded_min_gap) if cfg.w_graded > 0 else score.sum() * 0
    l_rating = rating_loss(logits[:, :T], rt, cfg.rating_loss) if cfg.rating_weight > 0 else score.sum() * 0
    total = l_bpr + cfg.w_graded * l_graded + cfg.rating_weight * l_rating
    return total, {"loss": total.item(), "bpr": l_bpr.item(), "graded": l_graded.item(),
                   "rating": l_rating.item()}
