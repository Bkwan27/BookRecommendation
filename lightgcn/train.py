"""Train LightGCN with BPR loss for top-K book recommendation.

Model selection / early stopping on validation Recall@20 (train items masked); the best
checkpoint is then evaluated on test (train + val items masked).

  python -m lightgcn.train                    # LightGCN, 3 layers
  python -m lightgcn.train --layers 0         # == MF-BPR baseline (no propagation)
  python -m lightgcn.train --popularity       # most-popular baseline only
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


def sample_negatives(users: np.ndarray, train_csr, n_items: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform negatives, resampling any that collide with a user's train positives."""
    neg = rng.integers(0, n_items, len(users))
    for _ in range(20):
        bad = np.flatnonzero(np.asarray(train_csr[users, neg]).ravel() > 0)
        if len(bad) == 0:
            break
        neg[bad] = rng.integers(0, n_items, len(bad))
    return neg


def fmt(m: dict, keys=("recall@20", "ndcg@20", "precision@20", "hit@20")) -> str:
    return " ".join(f"{k}={m[k]:.4f}" for k in keys)


def popularity(ds: data.Dataset, ks) -> dict:
    pop = torch.from_numpy(np.asarray(ds.train_csr.sum(0)).ravel())
    score = lambda b: pop.expand(len(b), -1).clone()
    val = evaluate(score, ds.val_csr, ds.train_csr, ks)
    test = evaluate(score, ds.test_csr, ds.train_csr + ds.val_csr, ks)
    return {"val": val, "test": test}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min_rating", type=int, default=4, help="ratings >= this are positive interactions")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-4, help="L2 weight on ego embeddings")
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--patience", type=int, default=5, help="evals without val improvement before stopping")
    p.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--popularity", action="store_true", help="only run the most-popular baseline")
    p.add_argument("--name", type=str, default=None)
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ds = data.load(a.min_rating)
    print(f"users={ds.n_users} items={ds.n_items} train={len(ds.train)} val={len(ds.val)} test={len(ds.test)}")

    name = a.name or ("popularity" if a.popularity else f"lgcn_L{a.layers}_d{a.dim}_lr{a.lr}_reg{a.reg}")
    out_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    if a.popularity:
        res = popularity(ds, a.ks)
        print("val ", fmt(res["val"]), "\ntest", fmt(res["test"]))
        json.dump({"args": vars(a), **res}, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
        return

    model = LightGCN(ds.n_users, ds.n_items, normalized_adj(ds.train, ds.n_users, ds.n_items), a.dim, a.layers).to(a.device)
    print(f"device={a.device}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def scorer():
        U, I = model.embeddings()
        return lambda b: U[torch.from_numpy(b).to(U.device)] @ I.T

    history, best, best_epoch, bad = [], -1.0, 0, 0
    ckpt = os.path.join(out_dir, "best.pt")
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(len(ds.train))
        tot, tot_reg, nb = 0.0, 0.0, 0
        for s in range(0, len(perm), a.batch_size):
            e = ds.train[perm[s: s + a.batch_size]]
            users, pos = e[:, 0], e[:, 1]
            neg = sample_negatives(users, ds.train_csr, ds.n_items, rng)
            loss, reg = model.bpr_loss(*(torch.from_numpy(x).to(a.device) for x in (users, pos, neg)), a.reg)
            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
            tot += loss.item(); tot_reg += reg.item(); nb += 1
        rec = {"epoch": epoch, "bpr": tot / nb, "reg": tot_reg / nb, "time": time.time() - t0}
        msg = f"epoch {epoch:3d} bpr={rec['bpr']:.4f} reg={rec['reg']:.5f} ({rec['time']:.0f}s)"

        if epoch % a.eval_every == 0 or epoch == a.epochs:
            model.eval()
            val = evaluate(scorer(), ds.val_csr, ds.train_csr, a.ks)
            rec["val"] = val
            msg += " | val " + fmt(val)
            if val["recall@20"] > best:
                best, best_epoch, bad = val["recall@20"], epoch, 0
                torch.save(model.state_dict(), ckpt)
                msg += " *"
            else:
                bad += 1
        history.append(rec)
        print(msg, flush=True)
        if bad >= a.patience:
            print(f"early stopping (best epoch {best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt, map_location=a.device))
    model.eval()
    val = evaluate(scorer(), ds.val_csr, ds.train_csr, a.ks)
    test = evaluate(scorer(), ds.test_csr, ds.train_csr + ds.val_csr, a.ks)
    print(f"best epoch {best_epoch}\nval  {fmt(val)}\ntest {fmt(test)}")
    json.dump({"args": vars(a), "best_epoch": best_epoch, "val": val, "test": test, "history": history},
              open(os.path.join(out_dir, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
