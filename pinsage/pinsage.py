#!/usr/bin/env python3
"""PinSage expert (first draft). Run from the repo root:

    python -m pinsage.pinsage
    python -m pinsage.pinsage --dim 64 --epochs 10

1. Neighborhoods: random walks (book -> reader -> book, with restarts) on the train graph; each
   book's most-visited books become its neighbors, weighted by visit counts.
2. Book input: numeric features + author/series/language/era embeddings + weighted tag bag + book id.
3. Two PinSage convolutions (importance pooling, concat, dense + ReLU, L2 norm) and a skip connection.
4. Max-margin loss: two books a user liked close together in time vs. random books shared by the batch.
5. Scoring: user query = normalised mean of their latest train books; score = cosine + book bias.
All 10k books are embedded in one pass per step, so no minibatch subgraph sampling is needed.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.metrics import ranking_metrics  # the shared eval module; adjust the name if yours differs
from common.process import DEFAULT_PROCESSED, load_item_features, load_processed

CATEGORICAL = ("author", "series", "language", "year_bucket")
SELECT_BY = "ndcg@20"


@dataclass
class Config:
    data: str = str(DEFAULT_PROCESSED)
    out: str = str(Path(__file__).resolve().parent.parent / "runs" / "pinsage")
    num_walks: int = 200       # random walks started from each book
    walk_hops: int = 3         # max book -> reader -> book hops per walk
    restart_prob: float = 0.5  # chance to stop after each hop
    num_neighbors: int = 10    # most-visited books kept as neighbours
    dim: int = 128
    layers: int = 2
    dropout: float = 0.2
    window: int = 5            # positive pair: liked books within 5 positions in a user's history
    batch_size: int = 2048
    num_negatives: int = 1024  # random negatives shared by the whole batch
    margin: float = 0.5
    lr: float = 1e-3
    epochs: int = 20
    patience: int = 3
    pairs_per_epoch: int = 0   # 0 = number of positive train edges
    recent: int = 5            # user query = mean of the latest `recent` train books (0 = all)
    seed: int = 42


def neighbourhoods(train: sp.csr_matrix, cfg: Config, rng) -> tuple[np.ndarray, np.ndarray]:
    """Top-T books visited by random walks from each book, and their normalised visit counts."""
    n = train.shape[1]
    by_user, by_book = train.tocsr(), train.T.tocsr()
    deg_u, deg_b = np.diff(by_user.indptr), np.diff(by_book.indptr)
    start = np.repeat(np.arange(n), cfg.num_walks)
    start = start[deg_b[start] > 0]
    cur, alive, keys = start.copy(), np.ones(len(start), bool), []
    for _ in range(cfg.walk_hops):
        idx = np.flatnonzero(alive)
        b = cur[idx]
        u = by_book.indices[by_book.indptr[b] + (rng.random(len(b)) * deg_b[b]).astype(np.int64)]
        nxt = by_user.indices[by_user.indptr[u] + (rng.random(len(u)) * deg_u[u]).astype(np.int64)]
        ok = nxt != start[idx]
        keys.append(start[idx][ok] * n + nxt[ok])
        cur[idx] = nxt
        alive[idx] = rng.random(len(idx)) >= cfg.restart_prob
    key, count = np.unique(np.concatenate(keys), return_counts=True)
    src, dst = key // n, key % n
    order = np.lexsort((-count, src))  # per book, most visited first
    src, dst, count = src[order], dst[order], count[order]
    first = np.r_[0, np.flatnonzero(np.diff(src)) + 1]
    rank = np.arange(len(src)) - np.repeat(first, np.diff(np.r_[first, len(src)]))
    keep = rank < cfg.num_neighbors
    nbr = np.repeat(np.arange(n)[:, None], cfg.num_neighbors, axis=1)  # padding: the book itself ...
    w = np.zeros(nbr.shape, np.float32)                                # ... with weight 0
    nbr[src[keep], rank[keep]] = dst[keep]
    w[src[keep], rank[keep]] = count[keep]
    return nbr, w / np.maximum(w.sum(1, keepdims=True), 1e-12)


class PairSampler:
    """(query, positive) pairs: two books one user liked within `window` positions in their train history."""

    def __init__(self, edges: dict, window: int, rng) -> None:
        order = np.lexsort((edges["t"], edges["user"]))
        user, self.item = edges["user"][order], edges["item"][order]
        first = np.r_[0, np.flatnonzero(np.diff(user)) + 1]
        length = np.diff(np.r_[first, len(user)])
        self.start, self.end = np.repeat(first, length), np.repeat(first + length, length)
        self.edges = np.flatnonzero(np.repeat(length, length) >= 2)  # a single book forms no pair
        self.window, self.rng = window, rng

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        e = self.edges[self.rng.integers(0, len(self.edges), n)]
        left = np.minimum(e - self.start[e], self.window)
        right = np.minimum(self.end[e] - 1 - e, self.window)
        r = (self.rng.random(n) * (left + right)).astype(np.int64)  # uniform over the window
        return self.item[e], self.item[np.where(r < left, e - 1 - r, e + 1 + r - left)]


def history(edges: dict, n_users: int, n_items: int, recent: int) -> sp.csr_matrix:
    """[users x books] indicator of each user's `recent` latest train positives (0 = all)."""
    u, i = edges["user"], edges["item"]
    if recent:
        order = np.lexsort((-edges["t"], u))  # per user, newest first
        first = np.r_[0, np.flatnonzero(np.diff(u[order])) + 1]
        rank = np.arange(len(u)) - np.repeat(first, np.diff(np.r_[first, len(u)]))
        keep = order[rank < recent]
        u, i = u[keep], i[keep]
    return sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(n_users, n_items))


class PinSage(nn.Module):
    def __init__(self, feats: dict, nbr: np.ndarray, nbr_w: np.ndarray, cfg: Config) -> None:
        super().__init__()
        n, d, tags = len(nbr), cfg.dim, feats["tags"]
        buffers = {"numeric": torch.as_tensor(feats["numeric"], dtype=torch.float32),
                   "tag_idx": torch.as_tensor(tags.indices, dtype=torch.long),
                   "tag_off": torch.as_tensor(tags.indptr[:-1], dtype=torch.long),
                   "tag_w": torch.as_tensor(tags.data, dtype=torch.float32),
                   "nbr": torch.as_tensor(nbr, dtype=torch.long),
                   "nbr_w": torch.as_tensor(nbr_w, dtype=torch.float32),
                   **{c: torch.as_tensor(feats[c], dtype=torch.long) for c in CATEGORICAL}}
        for name, value in buffers.items():
            self.register_buffer(name, value)
        self.num = nn.Linear(feats["numeric"].shape[1], d)
        self.cat = nn.ModuleDict({c: nn.Embedding(len(feats[f"vocab_{c}"]), d) for c in CATEGORICAL})
        self.tags = nn.EmbeddingBag(tags.shape[1], d, mode="sum")  # weighted bag = linear layer on tag weights
        self.ids = nn.Embedding(n, d)
        self.q = nn.ModuleList(nn.Linear(d, d) for _ in range(cfg.layers))
        self.w = nn.ModuleList(nn.Linear(2 * d, d) for _ in range(cfg.layers))
        self.bias = nn.Parameter(torch.zeros(n))  # per-book bias absorbs popularity
        self.drop = nn.Dropout(cfg.dropout)
        for emb in [*self.cat.values(), self.tags, self.ids]:
            nn.init.normal_(emb.weight, std=0.05)

    def forward(self) -> torch.Tensor:
        """Unit-norm embeddings of all books, [n_items x dim]."""
        h0 = (self.num(self.numeric) + self.ids.weight
              + self.tags(self.tag_idx, self.tag_off, per_sample_weights=self.tag_w)
              + sum(self.cat[c](getattr(self, c)) for c in CATEGORICAL))
        h = h0
        for q, w in zip(self.q, self.w):
            pooled = torch.einsum("nt,ntd->nd", self.nbr_w, F.relu(q(self.drop(h)))[self.nbr])  # importance pooling
            h = F.normalize(F.relu(w(self.drop(torch.cat([h, pooled], dim=1)))), dim=1)
        return F.normalize(h0 + h, dim=1)  # skip connection from the input, as in DGL's example


@torch.no_grad()
def scorer(model: PinSage, hist: sp.csr_matrix):
    """score_fn for ranking_metrics: cosine(user query, book) + bias(book), i.e. the training score."""
    model.eval()
    z, b = model(), model.bias.detach()
    model.train()
    q = F.normalize(torch.as_tensor(hist @ z.cpu().numpy(), device=z.device), dim=1)
    return lambda users: q[torch.as_tensor(users, device=z.device)] @ z.T + b


def fit(cfg: Config, data: dict, feats: dict, device: torch.device):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = PinSage(feats, *neighbourhoods(data["train"], cfg, rng), cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pairs = PairSampler(data["train_edges"], cfg.window, rng)
    hist = history(data["train_edges"], data["n_users"], data["n_items"], cfg.recent)  # train only
    steps = max(1, (cfg.pairs_per_epoch or data["train"].nnz) // cfg.batch_size)
    best, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, cfg.epochs + 1):
        t0, total = time.time(), 0.0
        for _ in range(steps):
            q, p = (torch.as_tensor(x, device=device) for x in pairs.sample(cfg.batch_size))
            neg = torch.as_tensor(rng.integers(0, data["n_items"], cfg.num_negatives), device=device)
            z = model()
            s_pos = (z[q] * z[p]).sum(1) + model.bias[p]
            s_neg = z[q] @ z[neg].T + model.bias[neg]
            valid = (neg[None] != p[:, None]) & (neg[None] != q[:, None])  # a negative equal to q or p is not one
            loss = (F.relu(cfg.margin + s_neg - s_pos[:, None]) * valid).sum() / valid.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        val = ranking_metrics(scorer(model, hist), data["val"], data["seen_before_val"])
        print(f"epoch {epoch}: loss {total / steps:.4f} | val recall@20 {val['recall@20']:.4f} "
              f"ndcg@20 {val['ndcg@20']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val[SELECT_BY] > best:
            best, best_epoch = val[SELECT_BY], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model, hist, best_epoch


def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, feats = load_processed(cfg.data), load_item_features(cfg.data)
    model, hist, best_epoch = fit(cfg, data, feats, device)
    score = scorer(model, hist)
    val = ranking_metrics(score, data["val"], data["seen_before_val"])
    test = ranking_metrics(score, data["test"], data["seen_before_test"])
    print(f"best epoch {best_epoch} | test recall@20 {test['recall@20']:.4f} ndcg@20 {test['ndcg@20']:.4f}")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "results.json").write_text(json.dumps(
        {"model": "pinsage", "fingerprint": data["fingerprint"], "config": asdict(cfg),
         "best_epoch": best_epoch, "val": val, "test": test}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate the PinSage expert.")
    for f in fields(Config):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    main(Config(**vars(ap.parse_args())))
