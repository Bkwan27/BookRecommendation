"""
Trains LightGCN with BPR loss for top-K book recommendation.

Uses the shared split from common/process.py (see lightgcn/data.py). We pick the best
model / early stop using validation Recall@20 (books rated in train are masked),
and then evaluate that checkpoint on test (books rated in train or val are masked).

  python -m lightgcn.train                    # LightGCN, 3 layers
  python -m lightgcn.train --layers 0         # same as MF-BPR (no propagation)
  python -m lightgcn.train --popularity       # just the most-popular baseline
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import data
from .metrics import evaluate
from .model import LightGCN, normalized_adj

RUNS_DIR = os.path.join(data.ROOT, "runs")

PRINT_KEYS = ("recall@20", "ndcg@20", "precision@20", "hit@20")


def sample_negatives(users: np.ndarray, train_csr, n_items: int, rng: np.random.Generator) -> np.ndarray:
    """
    Pick a random item for each user. If the item is one of the user's train
    positives, resample it (up to 20 tries).
    """
    neg = rng.integers(0, n_items, len(users))

    for attempt in range(20):
        is_positive = np.asarray(train_csr[users, neg]).flatten() > 0
        bad = np.flatnonzero(is_positive)
        if len(bad) == 0:
            break
        neg[bad] = rng.integers(0, n_items, len(bad))

    return neg


def fmt(metrics: dict, keys=PRINT_KEYS) -> str:
    parts = []
    for k in keys:
        parts.append(f"{k}={metrics[k]:.4f}")
    return " ".join(parts)


def popularity(ds: data.Dataset, ks) -> dict:
    """Baseline that recommends the most popular books to everyone."""
    counts = np.asarray(ds.train_csr.sum(axis=0)).flatten()
    pop = torch.from_numpy(counts)

    def score(batch):
        return pop.expand(len(batch), -1).clone()

    val = evaluate(score, ds.val_csr, ds.val_exclude, ks)
    test = evaluate(score, ds.test_csr, ds.test_exclude, ks)
    return {"val": val, "test": test}


def make_scorer(model):
    user_emb, item_emb = model.embeddings()

    def score(batch):
        batch_users = torch.from_numpy(batch).to(user_emb.device)
        return user_emb[batch_users] @ item_emb.T

    return score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=data.PROC_DIR, help="output dir of `python -m common.process`")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4, help="L2 weight on ego embeddings")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5, help="evals without val improvement before stopping")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--popularity", action="store_true", help="only run the most-popular baseline")
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ds = data.load(args.data_dir)
    print(f"users={ds.n_users} items={ds.n_items} train={len(ds.train)} "
          f"val={ds.val_csr.nnz} test={ds.test_csr.nnz}")

    if args.name:
        run_name = args.name
    elif args.popularity:
        run_name = "popularity"
    else:
        run_name = f"lgcn_L{args.layers}_d{args.dim}_lr{args.lr}_reg{args.reg}"

    out_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.json")

    # popularity baseline, no training needed
    if args.popularity:
        res = popularity(ds, args.ks)
        print("val ", fmt(res["val"]))
        print("test", fmt(res["test"]))

        output = {"model": "popularity", "fingerprint": ds.fingerprint, "args": vars(args), "data_config": ds.config}
        output["val"] = res["val"]
        output["test"] = res["test"]
        save_json(output, results_path)
        return

    adj = normalized_adj(ds.train, ds.n_users, ds.n_items)
    model = LightGCN(ds.n_users, ds.n_items, adj, args.dim, args.layers)
    model = model.to(args.device)
    print(f"device={args.device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_recall = -1.0
    best_epoch = 0
    evals_since_best = 0
    ckpt_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()

        perm = rng.permutation(len(ds.train))
        total_bpr = 0.0
        total_reg = 0.0
        n_batches = 0

        for start in range(0, len(perm), args.batch_size):
            batch_edges = ds.train[perm[start:start + args.batch_size]]
            users = batch_edges[:, 0]
            pos = batch_edges[:, 1]
            neg = sample_negatives(users, ds.train_csr, ds.n_items, rng)

            users_t = torch.from_numpy(users).to(args.device)
            pos_t = torch.from_numpy(pos).to(args.device)
            neg_t = torch.from_numpy(neg).to(args.device)

            bpr_loss, reg_loss = model.bpr_loss(users_t, pos_t, neg_t, args.reg)
            loss = bpr_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bpr += bpr_loss.item()
            total_reg += reg_loss.item()
            n_batches += 1

        epoch_time = time.time() - start_time
        record = {
            "epoch": epoch,
            "bpr": total_bpr / n_batches,
            "reg": total_reg / n_batches,
            "time": epoch_time,
        }
        msg = f"epoch {epoch:3d} bpr={record['bpr']:.4f} reg={record['reg']:.5f} ({epoch_time:.0f}s)"

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            model.eval()
            val = evaluate(make_scorer(model), ds.val_csr, ds.val_exclude, args.ks)
            record["val"] = val
            msg += " | val " + fmt(val)

            if val["recall@20"] > best_recall:
                best_recall = val["recall@20"]
                best_epoch = epoch
                evals_since_best = 0
                torch.save(model.state_dict(), ckpt_path)
                msg += " *"
            else:
                evals_since_best += 1

        history.append(record)
        print(msg, flush=True)

        if evals_since_best >= args.patience:
            print(f"early stopping (best epoch {best_epoch})")
            break

    # reload the best checkpoint and evaluate on val and test
    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    val = evaluate(make_scorer(model), ds.val_csr, ds.val_exclude, args.ks)
    test = evaluate(make_scorer(model), ds.test_csr, ds.test_exclude, args.ks)

    print(f"best epoch {best_epoch}")
    print(f"val  {fmt(val)}")
    print(f"test {fmt(test)}")

    output = {
        "model": "lightgcn",
        "fingerprint": ds.fingerprint,
        "args": vars(args),
        "data_config": ds.config,
        "best_epoch": best_epoch,
        "val": val,
        "test": test,
        "history": history,
    }
    save_json(output, results_path)


if __name__ == "__main__":
    main()
