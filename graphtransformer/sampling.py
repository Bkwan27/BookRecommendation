"""Training-batch construction: context/target masking and negative sampling.

Each training step, for every user in the batch we split their *train* ratings
into
  * context edges  - what the model sees (the user's "library"), and
  * target edges   - what it must score, hidden from the context.
This mirrors serving, where the library is known and candidates are unseen.

Negatives:
  * random  - uniform books the user has neither rated (train) nor marked to-read;
  * hard    - PinSAGE-style "close but not connected" books: co-rated neighbours
              (ranks ``hard_rank[0]..hard_rank[1]``) of a random context book. This
              is a 3-hop approximation (u -> book -> users -> book) of random walk
              with restart visit counts;
  * disliked targets (1-2 stars) are added as negatives in the loss, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .data import Dataset, UserHistory

N_RECENCY = 10


def recency_bucket(rank_from_end: np.ndarray) -> np.ndarray:
    """0 = most recent rating, then log2 buckets, capped at N_RECENCY - 1."""
    return np.minimum(np.floor(np.log2(rank_from_end + 1)).astype(np.int64), N_RECENCY - 1)


def corated_neighbors(hist: UserHistory, n_books: int, k: int = 200, chunk: int = 1000,
                      cache: Path | str | None = None) -> np.ndarray:
    """Top-``k`` books by cosine co-rating similarity, from train edges only. [n_books, k]."""
    if cache is not None and Path(cache).exists():
        nb = np.load(cache)["nbrs"]
        if nb.shape == (n_books, k):
            return nb
    users = np.repeat(np.arange(hist.n_users), hist.counts())
    R = sp.csr_matrix((np.ones(len(users), np.float32), (users, hist.book)),
                      shape=(hist.n_users, n_books))
    deg = np.asarray(R.sum(0)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    Rt = R.T.tocsr()
    R = R.tocsc()
    out = np.empty((n_books, k), dtype=np.int32)
    for a in range(0, n_books, chunk):
        b = min(a + chunk, n_books)
        C = (Rt @ R[:, a:b]).toarray()                     # [n_books, b-a] co-counts
        C *= inv[:, None]
        C *= inv[None, a:b]
        C[np.arange(a, b), np.arange(b - a)] = -1.0          # drop self
        top = np.argpartition(-C, k, axis=0)[:k]              # [k, b-a] unordered
        vals = np.take_along_axis(C, top, axis=0)
        order = np.argsort(-vals, axis=0)
        out[a:b] = np.take_along_axis(top, order, axis=0).T
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, nbrs=out)
    return out


@dataclass
class Batch:
    users: torch.Tensor        # [B]
    ctx_book: torch.Tensor     # [B, L] (pad = n_books)
    ctx_rating: torch.Tensor   # [B, L] 1..5 (pad = 0)
    ctx_recency: torch.Tensor  # [B, L]
    ctx_mask: torch.Tensor     # [B, L] True = real token
    tgt_book: torch.Tensor     # [B, T]
    tgt_rating: torch.Tensor   # [B, T] 1..5 (pad = 0)
    read_book: torch.Tensor    # [B, R] to-read positives
    read_mask: torch.Tensor    # [B, R]
    neg_book: torch.Tensor     # [B, N] sampled negatives (random + hard)
    neg_mask: torch.Tensor     # [B, N]
    neg_is_hard: torch.Tensor  # [B, N]

    def to(self, device):
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})

    @property
    def cand_book(self) -> torch.Tensor:
        """All books the model scores for each user: targets | to-read | negatives."""
        return torch.cat([self.tgt_book, self.read_book, self.neg_book], dim=1)


def _pad(rows, width, fill, dtype=np.int64):
    out = np.full((len(rows), width), fill, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r[:width]
    return out


def build_context(books, ratings, keep_idx, n_total, max_context, rng=None):
    """Select up to ``max_context`` context edges (time order kept) and their recency."""
    if len(keep_idx) > max_context:
        if rng is None:      # deterministic (eval): most recent edges
            keep_idx = keep_idx[-max_context:]
        else:
            keep_idx = np.sort(rng.choice(keep_idx, max_context, replace=False))
    rec = recency_bucket(n_total - 1 - keep_idx)
    return books[keep_idx], ratings[keep_idx], rec


class TrainSampler:
    def __init__(self, ds: Dataset, nbrs: np.ndarray | None, *, max_context: int = 128,
                 max_targets: int = 32, target_frac: float = 0.2, n_rand_neg: int = 16,
                 hard_rank: tuple[int, int] = (10, 100), n_to_read: int = 4, seed: int = 0):
        self.ds, self.nbrs = ds, nbrs
        self.max_context, self.max_targets = max_context, max_targets
        self.target_frac, self.n_rand_neg = target_frac, n_rand_neg
        self.hard_lo, self.hard_hi = hard_rank
        if nbrs is not None:
            assert self.hard_hi <= nbrs.shape[1]
        self.n_to_read = n_to_read if ds.to_read is not None else 0
        self.rng = np.random.default_rng(seed)
        self.n_hard = 0     # curriculum: set by the training loop each epoch

    def trainable_users(self) -> np.ndarray:
        return np.flatnonzero(self.ds.train.counts() >= 2)

    def sample(self, users: np.ndarray) -> Batch:
        ds, rng, nb = self.ds, self.rng, self.ds.n_books
        cb, cr, cc, tb, tr, rb, ngb, hard_flags = [], [], [], [], [], [], [], []
        for u in users:
            books, ratings, _ = ds.train.get(u)
            n = len(books)
            k = int(np.clip(round(self.target_frac * n), 1, min(self.max_targets, n - 1)))
            perm = rng.permutation(n)
            tgt_idx, ctx_idx = np.sort(perm[:k]), np.sort(perm[k:])
            b, r, rec = build_context(books, ratings, ctx_idx, n, self.max_context, rng)
            cb.append(b); cr.append(r); cc.append(rec)
            tb.append(books[tgt_idx]); tr.append(ratings[tgt_idx])

            if self.n_to_read:
                tr_books = ds.to_read.get(u)[0]
                if len(tr_books) > self.n_to_read:
                    tr_books = rng.choice(tr_books, self.n_to_read, replace=False)
                rb.append(tr_books)
            else:
                rb.append(np.empty(0, np.int64))

            neg = rng.integers(0, nb, self.n_rand_neg)
            flags = np.zeros(self.n_rand_neg, bool)
            if self.n_hard and self.nbrs is not None:
                anchors = rng.choice(b, self.n_hard)
                ranks = rng.integers(self.hard_lo, self.hard_hi, self.n_hard)
                neg = np.concatenate([neg, self.nbrs[anchors, ranks]])
                flags = np.concatenate([flags, np.ones(self.n_hard, bool)])
            ngb.append(neg); hard_flags.append(flags)

        L = max(len(x) for x in cb)
        T = max(len(x) for x in tb)
        R = max(1, max(len(x) for x in rb))
        neg = np.stack(ngb).astype(np.int64)
        U = np.repeat(np.asarray(users), neg.shape[1]).reshape(neg.shape)
        # Negatives must be unrated (train) and not on the to-read shelf. A random
        # negative can still be a held-out positive; that is unavoidable without
        # peeking at val/test and is standard practice.
        neg_mask = ~ds.known.contains(U, neg)
        read_book = _pad(rb, R, nb)
        t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt)
        return Batch(
            users=t(np.asarray(users)),
            ctx_book=t(_pad(cb, L, nb)), ctx_rating=t(_pad(cr, L, 0)),
            ctx_recency=t(_pad(cc, L, 0)),
            ctx_mask=t(_pad([np.ones(len(x), bool) for x in cb], L, False, bool), torch.bool),
            tgt_book=t(_pad(tb, T, nb)), tgt_rating=t(_pad(tr, T, 0)),
            read_book=t(read_book), read_mask=t(read_book != nb, torch.bool),
            neg_book=t(neg), neg_mask=t(neg_mask, torch.bool),
            neg_is_hard=t(np.stack(hard_flags), torch.bool),
        )
