"""Graph-transformer view of the SHARED processed data (common/process.py).

This module no longer builds its own split. It wraps common.process.load_processed() into per-user
time-ordered histories for the sampler. Ids are the processed indices used by every expert.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from common.process import DEFAULT_PROCESSED, load_processed


class UserHistory:
    """CSR store of each user's edges, in time order. ``get(u) -> (items, ratings, times)``."""

    def __init__(self, user, item, rating, time, n_users):
        user = np.asarray(user, np.int64)
        order = np.lexsort((time, user))
        self.book = np.asarray(item, np.int64)[order]
        self.rating = np.asarray(rating, np.int64)[order]
        self.time = np.asarray(time, np.int64)[order]
        counts = np.bincount(user, minlength=n_users)
        self.indptr = np.concatenate([[0], np.cumsum(counts)])
        self.n_users = n_users

    @classmethod
    def from_rows(cls, rows: dict, n_users: int) -> "UserHistory":
        return cls(rows["user"], rows["item"], rows["rating"], rows["t"], n_users)

    def get(self, u: int):
        a, b = self.indptr[u], self.indptr[u + 1]
        return self.book[a:b], self.rating[a:b], self.time[a:b]

    def count(self, u: int) -> int:
        return int(self.indptr[u + 1] - self.indptr[u])

    def counts(self) -> np.ndarray:
        return np.diff(self.indptr)


class PairSet:
    """Vectorised membership test for (user, item) pairs."""

    def __init__(self, user, item, n_items):
        self.n_items = n_items
        self.keys = np.unique(np.asarray(user, np.int64) * n_items + np.asarray(item, np.int64))

    def contains(self, user, item) -> np.ndarray:
        k = np.asarray(user, dtype=np.int64) * self.n_items + np.asarray(item, dtype=np.int64)
        if len(self.keys) == 0:
            return np.zeros(k.shape, dtype=bool)
        idx = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        return self.keys[idx] == k


@dataclass
class Dataset:
    n_users: int
    n_items: int
    positive_threshold: int
    fingerprint: str
    train: UserHistory            # ALL train ratings (any stars): the only model input
    val: UserHistory              # all held-out val ratings: rating metrics only
    test: UserHistory             # all held-out test ratings: rating metrics only
    targets: dict[str, sp.csr_matrix]   # "val"/"test": warm positives (shared ground truth)
    exclude: dict[str, sp.csr_matrix]   # "val": train rated; "test": train + val rated (any stars)
    known: PairSet                # never sampled as negatives: train-rated (+ to_read if used)
    to_read: UserHistory | None = None
    train_pos: sp.csr_matrix | None = None   # positive train edges (for the popularity baseline)

    @property
    def n_books(self) -> int:     # alias used by the sampler / model
        return self.n_items

    def history(self, split: str) -> UserHistory:
        return {"val": self.val, "test": self.test}[split]


def from_processed(data: dict, use_to_read: bool = False) -> Dataset:
    n_u, n_i = data["n_users"], data["n_items"]
    tr = data["train_all"]
    to_read, known_u, known_i = None, tr["user"], tr["item"]
    if use_to_read:   # ablation only: the shared protocol does not use to_read as positives
        rd = data["to_read"]
        to_read = UserHistory(rd["user"], rd["item"], np.zeros_like(rd["item"]),
                              np.arange(len(rd["item"])), n_u)
        known_u = np.concatenate([known_u, rd["user"]])
        known_i = np.concatenate([known_i, rd["item"]])
    return Dataset(
        n_users=n_u, n_items=n_i, positive_threshold=data["positive_threshold"],
        fingerprint=data["fingerprint"],
        train=UserHistory.from_rows(tr, n_u),
        val=UserHistory.from_rows(data["val_rows"], n_u),
        test=UserHistory.from_rows(data["test_rows"], n_u),
        targets={"val": data["val"], "test": data["test"]},
        exclude={"val": data["seen_before_val"], "test": data["seen_before_test"]},
        known=PairSet(known_u, known_i, n_i), to_read=to_read, train_pos=data["train"],
    )


def load_dataset(processed_dir: str | Path = DEFAULT_PROCESSED, use_to_read: bool = False) -> Dataset:
    return from_processed(load_processed(processed_dir, with_to_read=use_to_read), use_to_read)
