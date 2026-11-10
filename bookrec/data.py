"""Loading goodbooks-10k, building leak-free splits, and per-user histories.

Conventions
-----------
* Internal ids are 0-based: ``user = user_id - 1`` and ``book = book_id - 1``.
* ``ratings.csv`` is sorted by time across *all* users (rows are interleaved),
  so the row index works as a global clock. A stable sort by user therefore keeps
  each user's ratings in time order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_DIR = Path(__file__).resolve().parent.parent / "splits"
N_BOOKS = 10_000
N_USERS = 53_424

TRAIN, VAL, TEST = 0, 1, 2


@dataclass
class Ratings:
    user: np.ndarray    # int64 [E], 0-based
    book: np.ndarray    # int64 [E], 0-based
    rating: np.ndarray  # int64 [E], 1..5
    time: np.ndarray    # int64 [E], row index in ratings.csv (global time order)
    n_users: int
    n_books: int


def load_ratings(data_dir: Path | str = DATA_DIR) -> Ratings:
    df = pd.read_csv(Path(data_dir) / "ratings.csv")
    return Ratings(
        user=df.user_id.to_numpy(np.int64) - 1,
        book=df.book_id.to_numpy(np.int64) - 1,
        rating=df.rating.to_numpy(np.int64),
        time=np.arange(len(df), dtype=np.int64),
        n_users=N_USERS,
        n_books=N_BOOKS,
    )


def load_to_read(data_dir: Path | str = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(Path(data_dir) / "to_read.csv")
    return df.user_id.to_numpy(np.int64) - 1, df.book_id.to_numpy(np.int64) - 1


def make_split(r: Ratings, mode: str = "temporal", val_frac: float = 0.1,
               test_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """Return an int8 label (TRAIN/VAL/TEST) for every rating edge.

    temporal : for each user, the last ``test_frac`` of their ratings go to test
               and the ``val_frac`` before those go to val ("predict what they
               rate next"). Every user keeps at least one train edge.
    random   : a global random edge split (80/10/10 by default).
    """
    split = np.zeros(len(r.user), dtype=np.int8)
    if mode == "random":
        rng = np.random.default_rng(seed)
        x = rng.random(len(r.user))
        split[x < test_frac] = TEST
        split[(x >= test_frac) & (x < test_frac + val_frac)] = VAL
        return split
    if mode != "temporal":
        raise ValueError(f"unknown split mode {mode!r}")

    order = np.lexsort((r.time, r.user))              # by user, then time
    users_sorted = r.user[order]
    counts = np.bincount(r.user, minlength=r.n_users)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(order)) - starts[users_sorted]   # 0..n_u-1 within user
    n = counts[users_sorted]
    n_test = np.maximum(1, np.round(n * test_frac)).astype(np.int64)
    n_test = np.minimum(n_test, n - 1)                    # keep >= 1 train edge
    n_val = np.maximum(1, np.round(n * val_frac)).astype(np.int64)
    n_val = np.minimum(n_val, n - n_test - 1)
    lab = np.full(len(order), TRAIN, dtype=np.int8)
    lab[rank >= n - n_test] = TEST
    lab[(rank >= n - n_test - n_val) & (rank < n - n_test)] = VAL
    split[order] = lab
    return split


def load_or_make_split(r: Ratings, mode: str = "temporal", seed: int = 0,
                       split_dir: Path | str = SPLIT_DIR, **kw) -> np.ndarray:
    """Freeze splits on disk so every teammate and model sees identical data."""
    p = Path(split_dir) / f"{mode}_seed{seed}.npz"
    if p.exists():
        s = np.load(p)["split"]
        assert len(s) == len(r.user), "cached split does not match ratings.csv"
        return s
    s = make_split(r, mode=mode, seed=seed, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, split=s)
    return s


class UserHistory:
    """CSR store of each user's edges, in time order. ``get(u) -> (books, ratings, times)``."""

    def __init__(self, user, book, rating, time, n_users):
        order = np.lexsort((time, user))
        self.book = book[order]
        self.rating = rating[order]
        self.time = time[order]
        counts = np.bincount(user, minlength=n_users)
        self.indptr = np.concatenate([[0], np.cumsum(counts)])
        self.n_users = n_users

    def get(self, u: int):
        a, b = self.indptr[u], self.indptr[u + 1]
        return self.book[a:b], self.rating[a:b], self.time[a:b]

    def count(self, u: int) -> int:
        return int(self.indptr[u + 1] - self.indptr[u])

    def counts(self) -> np.ndarray:
        return np.diff(self.indptr)


class PairSet:
    """Vectorised membership test for (user, book) pairs."""

    def __init__(self, user, book, n_books):
        self.n_books = n_books
        self.keys = np.unique(np.asarray(user, np.int64) * n_books + np.asarray(book, np.int64))

    def contains(self, user, book) -> np.ndarray:
        k = np.asarray(user, dtype=np.int64) * self.n_books + np.asarray(book, dtype=np.int64)
        if len(self.keys) == 0:
            return np.zeros(k.shape, dtype=bool)
        idx = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        return self.keys[idx] == k


@dataclass
class Dataset:
    ratings: Ratings
    split: np.ndarray
    train: UserHistory
    val: UserHistory
    test: UserHistory
    to_read: UserHistory | None   # leak-filtered "to read" marks (rating field unused)
    known: PairSet                # train ratings ∪ to_read: never sampled as negatives

    @property
    def n_users(self):
        return self.ratings.n_users

    @property
    def n_books(self):
        return self.ratings.n_books


def build_dataset(r: Ratings, split: np.ndarray,
                  to_read: tuple[np.ndarray, np.ndarray] | None = None) -> Dataset:
    def hist(mask):
        return UserHistory(r.user[mask], r.book[mask], r.rating[mask], r.time[mask], r.n_users)

    tr = split == TRAIN
    tr_user, tr_book = r.user[tr], r.book[tr]
    tr_read = None
    known_user, known_book = tr_user, tr_book
    if to_read is not None:
        tu, tb = to_read
        # Drop "to read" marks on books whose rating is held out (a leak: it hints the
        # user knows the book) and on books already rated in train (no new signal).
        held = PairSet(r.user[~tr], r.book[~tr], r.n_books)
        rated_tr = PairSet(tr_user, tr_book, r.n_books)
        keep = ~held.contains(tu, tb) & ~rated_tr.contains(tu, tb)
        tu, tb = tu[keep], tb[keep]
        tr_read = UserHistory(tu, tb, np.zeros_like(tb), np.arange(len(tb)), r.n_users)
        known_user = np.concatenate([tr_user, tu])
        known_book = np.concatenate([tr_book, tb])

    return Dataset(
        ratings=r, split=split, train=hist(tr), val=hist(split == VAL),
        test=hist(split == TEST), to_read=tr_read,
        known=PairSet(known_user, known_book, r.n_books),
    )
