"""
Loads the preprocessed data split made by process.py for LightGCN.

All the cleaning, k-core filtering, splitting and masking happens in process.py,
so every model gets evaluated on the same users, books and ground truth.
To build the data once:

    python process.py --raw-dir data --out-dir data/processed

How the split works (see process.load_processed):
  - each user's ratings are split 80/10/10 by time (using all ratings)
  - graph edges / BPR positives are train ratings with >= 4 stars
  - val/test ground truth are warm held-out ratings with >= 4 stars
  - for val we mask everything the user rated in train; for test we mask
    everything rated in train or val (any star value)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from process import load_processed  # noqa: E402


@dataclass
class Dataset:
    n_users: int
    n_items: int

    # (num_edges, 2) array of [user, item] positive train edges
    train: np.ndarray

    # users x items matrices
    train_csr: sp.csr_matrix     # positive train edges (used for the graph and BPR)
    val_csr: sp.csr_matrix       # val ground truth
    test_csr: sp.csr_matrix      # test ground truth
    val_exclude: sp.csr_matrix   # items to mask for val (rated in train)
    test_exclude: sp.csr_matrix  # items to mask for test (rated in train or val)

    path: str
    config: dict


def load(path: str = PROC_DIR) -> Dataset:
    interactions_file = os.path.join(path, "interactions.npz")
    if not os.path.exists(interactions_file):
        raise FileNotFoundError(
            f"{path} has no processed data; run: python process.py --raw-dir data --out-dir {path}"
        )

    processed = load_processed(path)

    train_coo = processed["train"].tocoo()
    edges = np.stack([train_coo.row, train_coo.col], axis=1)
    edges = edges.astype(np.int64)

    with open(os.path.join(path, "stats.json")) as f:
        stats = json.load(f)
    config = stats["config"]

    dataset = Dataset(
        n_users=processed["n_users"],
        n_items=processed["n_items"],
        train=edges,
        train_csr=processed["train"],
        val_csr=processed["val"],
        test_csr=processed["test"],
        val_exclude=processed["seen_before_val"],
        test_exclude=processed["seen_before_test"],
        path=path,
        config=config,
    )
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=PROC_DIR)
    args = parser.parse_args()

    ds = load(args.data_dir)

    n_train = ds.train_csr.nnz
    n_val = ds.val_csr.nnz
    n_test = ds.test_csr.nnz
    density = n_train / (ds.n_users * ds.n_items)

    users_with_val = np.diff(ds.val_csr.indptr).astype(bool).sum()
    users_with_test = np.diff(ds.test_csr.indptr).astype(bool).sum()

    print(f"data dir: {ds.path}  split={ds.config['split']} positive>={ds.config['positive_threshold']}")
    print(f"users={ds.n_users} items={ds.n_items} train={n_train} val={n_val} test={n_test}")
    print(f"density={density:.4%}")
    print(f"users with val targets={users_with_val}, test targets={users_with_test}")
