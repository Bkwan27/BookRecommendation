"""Implicit-feedback data + per-user chronological train/val/test split for top-K recommendation.

ratings.csv is sorted by time (no timestamps), so file order is used as the time axis:
for every user, the earliest 80% of their positive interactions go to train, the next 10% to
validation, and the latest 10% to test (a "leave-latest-out" style split, no future leakage).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")


def split_dir(min_rating: int, val_frac: float, test_frac: float) -> str:
    return os.path.join(PROC_DIR, f"implicit_r{min_rating}_v{val_frac}_t{test_frac}")


def make_split(min_rating: int = 4, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    """Build train/val/test CSVs (user_idx, item_idx, rating) with contiguous 0-based ids."""
    out = split_dir(min_rating, val_frac, test_frac)
    if all(os.path.exists(os.path.join(out, f"{s}.csv")) for s in ("train", "val", "test")):
        return out
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    df["order"] = np.arange(len(df))  # file order == time order
    df = df[df["rating"] >= min_rating].copy()

    # Contiguous ids over users/books that have at least one positive.
    df["user_idx"], users = pd.factorize(df["user_id"], sort=True)
    df["item_idx"], items = pd.factorize(df["book_id"], sort=True)
    pd.DataFrame({"user_idx": np.arange(len(users)), "user_id": users}).to_csv(
        os.path.join(out, "user_map.csv"), index=False)
    pd.DataFrame({"item_idx": np.arange(len(items)), "book_id": items}).to_csv(
        os.path.join(out, "item_map.csv"), index=False)

    df = df.sort_values(["user_idx", "order"], kind="stable")
    rank = df.groupby("user_idx").cumcount().to_numpy()
    n = df.groupby("user_idx")["user_idx"].transform("size").to_numpy()
    n_test = np.floor(n * test_frac).astype(int)
    n_val = np.floor(n * val_frac).astype(int)
    # Users with >=3 positives always get at least one val and one test item.
    eligible = n >= 3
    n_test = np.where(eligible, np.maximum(n_test, 1), 0)
    n_val = np.where(eligible, np.maximum(n_val, 1), 0)
    n_train = n - n_val - n_test

    split = np.where(rank < n_train, "train", np.where(rank < n_train + n_val, "val", "test"))
    cols = ["user_idx", "item_idx", "rating"]
    for s in ("train", "val", "test"):
        df.loc[split == s, cols].to_csv(os.path.join(out, f"{s}.csv"), index=False)
    return out


@dataclass
class Dataset:
    n_users: int
    n_items: int
    train: np.ndarray  # (E, 2) int64 [user, item]
    val: np.ndarray
    test: np.ndarray
    train_csr: sp.csr_matrix  # users x items, binary
    val_csr: sp.csr_matrix
    test_csr: sp.csr_matrix
    path: str


def load(min_rating: int = 4, val_frac: float = 0.1, test_frac: float = 0.1) -> Dataset:
    path = make_split(min_rating, val_frac, test_frac)
    n_users = len(pd.read_csv(os.path.join(path, "user_map.csv")))
    n_items = len(pd.read_csv(os.path.join(path, "item_map.csv")))
    arrs, csrs = {}, {}
    for s in ("train", "val", "test"):
        e = pd.read_csv(os.path.join(path, f"{s}.csv"), usecols=["user_idx", "item_idx"]).to_numpy(np.int64)
        arrs[s] = e
        csrs[s] = sp.csr_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(n_users, n_items))
    return Dataset(n_users, n_items, arrs["train"], arrs["val"], arrs["test"],
                   csrs["train"], csrs["val"], csrs["test"], path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min_rating", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    a = p.parse_args()
    d = load(a.min_rating, a.val_frac, a.test_frac)
    print(f"split dir: {d.path}")
    print(f"users={d.n_users} items={d.n_items} train={len(d.train)} val={len(d.val)} test={len(d.test)}")
    print(f"density={len(d.train) / (d.n_users * d.n_items):.4%}")
    print(f"users with val items={np.diff(d.val_csr.indptr).astype(bool).sum()}, "
          f"test items={np.diff(d.test_csr.indptr).astype(bool).sum()}")
    cold = np.setdiff1d(np.unique(d.test[:, 1]), np.unique(d.train[:, 1]))
    print(f"test items never seen in train: {len(cold)}")
