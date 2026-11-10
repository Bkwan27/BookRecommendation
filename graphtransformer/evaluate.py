"""Full-catalogue evaluation: Recall@K / NDCG@K for ranking, RMSE / accuracy for ratings.

For each evaluated user the context is their train edges (most recent ``max_context``),
every book is scored, books rated in train are excluded, and positives are held-out
edges rated >= pos_threshold.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import Dataset, UserHistory
from .losses import expected_rating
from .sampling import build_context


def _pad(rows, width, fill):
    out = np.full((len(rows), width), fill, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


@torch.no_grad()
def evaluate(model, ds: Dataset, held: UserHistory, users: np.ndarray, *, device,
             ks=(10, 20, 50), max_context: int = 128, pos_threshold: int = 4,
             batch_users: int = 32, cand_chunk: int = 2500) -> dict:
    model.eval()
    E = model.books()
    nb = ds.n_books
    recall = {k: [] for k in ks}
    ndcg = {k: [] for k in ks}
    sq_err, correct, n_rated = 0.0, 0, 0
    kmax = max(ks)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))

    users = np.asarray([u for u in users if ds.train.count(u) > 0 and held.count(u) > 0])
    for s in range(0, len(users), batch_users):
        ub = users[s:s + batch_users]
        cb, cr, cc, hb, hr = [], [], [], [], []
        for u in ub:
            books, ratings, _ = ds.train.get(u)
            b, r, rec = build_context(books, ratings, np.arange(len(books)), len(books), max_context)
            cb.append(b); cr.append(r); cc.append(rec)
            hbk, hrt, _ = held.get(u)
            hb.append(hbk); hr.append(hrt)
        L = max(len(x) for x in cb)
        t = lambda a: torch.as_tensor(a, device=device)
        ctx_mask = t(_pad([np.ones(len(x), np.int64) for x in cb], L, 0)).bool()
        tokens, mask = model.encode_user(E, t(_pad(cb, L, nb)), t(_pad(cr, L, 0)),
                                         t(_pad(cc, L, 0)), ctx_mask)
        B = len(ub)
        scores = torch.empty(B, nb, device=device)
        for a in range(0, nb, cand_chunk):
            cand = torch.arange(a, min(a + cand_chunk, nb), device=device).expand(B, -1)
            scores[:, a:a + cand_chunk] = model.score(E, tokens, mask, cand)[0]
        for i, u in enumerate(ub):                       # exclude already-read books
            scores[i, t(ds.train.get(u)[0])] = float("-inf")
        top = scores.topk(kmax, dim=1).indices.cpu().numpy()

        H = max(len(x) for x in hb)
        hb_t = t(_pad(hb, H, nb))
        _, logits = model.score(E, tokens, mask, hb_t)
        pred = expected_rating(logits).cpu().numpy()
        cls = logits.argmax(-1).cpu().numpy() + 1

        for i in range(B):
            n = len(hb[i])
            sq_err += float(((pred[i, :n] - hr[i]) ** 2).sum())
            correct += int((cls[i, :n] == hr[i]).sum())
            n_rated += n
            pos = set(hb[i][hr[i] >= pos_threshold].tolist())
            if not pos:
                continue
            hit = np.isin(top[i], list(pos))
            for k in ks:
                recall[k].append(hit[:k].sum() / len(pos))
                idcg = disc[:min(len(pos), k)].sum()
                ndcg[k].append((hit[:k] * disc[:k]).sum() / idcg)

    out = {f"recall@{k}": float(np.mean(recall[k])) if recall[k] else float("nan") for k in ks}
    out.update({f"ndcg@{k}": float(np.mean(ndcg[k])) if ndcg[k] else float("nan") for k in ks})
    out["rmse"] = float(np.sqrt(sq_err / max(n_rated, 1)))
    out["rating_acc"] = correct / max(n_rated, 1)
    out["n_users"] = int(len(users))
    model.train()
    return out


def constant_baselines(ds: Dataset, held: UserHistory, pos_threshold: int = 4) -> dict:
    """Reference numbers: predict the train mean rating; recommend the most-rated books."""
    mu = ds.train.rating.mean()
    rmse = float(np.sqrt(((held.rating - mu) ** 2).mean()))
    mode = np.bincount(ds.train.rating).argmax()
    acc = float((held.rating == mode).mean())
    return {"rmse_const_mean": rmse, "acc_majority_class": acc, "train_mean": float(mu)}


@torch.no_grad()
def popularity_recall(ds: Dataset, held: UserHistory, users, ks=(10, 20, 50),
                      pos_threshold: int = 4) -> dict:
    """Recall@K for recommending the most-rated (in train) unread books: the baseline to beat."""
    pop = np.bincount(ds.train.book, minlength=ds.n_books).astype(np.float64)
    order = np.argsort(-pop)
    res = {k: [] for k in ks}
    for u in users:
        hbk, hrt, _ = held.get(u)
        pos = set(hbk[hrt >= pos_threshold].tolist())
        if not pos or ds.train.count(u) == 0:
            continue
        seen = set(ds.train.get(u)[0].tolist())
        top = [b for b in order[:max(ks) + len(seen)] if b not in seen][:max(ks)]
        hit = np.isin(top, list(pos))
        for k in ks:
            res[k].append(hit[:k].sum() / len(pos))
    return {f"pop_recall@{k}": float(np.mean(v)) for k, v in res.items()}
