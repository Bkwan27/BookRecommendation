"""LightGCN (He et al., SIGIR 2020): linear propagation on the normalized user-item graph,
no feature transforms or nonlinearities; final embedding = mean of all layer embeddings."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def normalized_adj(train: np.ndarray, n_users: int, n_items: int) -> torch.Tensor:
    """Symmetric-normalized bipartite adjacency D^-1/2 A D^-1/2 over (users + items) nodes."""
    u, i = train[:, 0], train[:, 1] + n_users
    n = n_users + n_items
    rows = np.concatenate([u, i])
    cols = np.concatenate([i, u])
    A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    d = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
    A = sp.diags(d) @ A @ sp.diags(d)
    A = A.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int64)), torch.from_numpy(A.indices.astype(np.int64)),
        torch.from_numpy(A.data), size=(n, n))


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, adj: torch.Tensor, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.register_buffer("adj", adj, persistent=False)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb.weight
        out = x
        for _ in range(self.n_layers):
            x = self.adj @ x
            out = out + x
        out = out / (self.n_layers + 1)
        return out[: self.n_users], out[self.n_users:]

    def bpr_loss(self, users, pos, neg, reg: float):
        U, I = self.propagate()
        u, p, n = U[users], I[pos], I[neg]
        pos_s, neg_s = (u * p).sum(-1), (u * n).sum(-1)
        loss = torch.nn.functional.softplus(neg_s - pos_s).mean()  # = -log sigmoid(pos - neg)
        # L2 on layer-0 (ego) embeddings of the batch, as in the paper.
        e = self.emb.weight
        l2 = (e[users].pow(2).sum() + e[self.n_users + pos].pow(2).sum()
              + e[self.n_users + neg].pow(2).sum()) / (2 * len(users))
        return loss, reg * l2

    @torch.no_grad()
    def embeddings(self):
        return self.propagate()
