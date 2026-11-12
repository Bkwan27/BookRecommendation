"""
LightGCN model (He et al., SIGIR 2020).

Each layer just multiplies the embeddings by the normalized user-item adjacency
matrix. There are no weight matrices or activations. The final embedding for a
node is the average of its embeddings from every layer (including layer 0).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adj(train: np.ndarray, n_users: int, n_items: int) -> torch.Tensor:
    """
    Build the symmetric normalized adjacency matrix D^-1/2 A D^-1/2.

    Users are nodes 0 .. n_users-1 and items are nodes n_users .. n_users+n_items-1.
    """
    n_nodes = n_users + n_items

    user_nodes = train[:, 0]
    item_nodes = train[:, 1] + n_users

    # add both directions so the matrix is symmetric
    rows = np.concatenate([user_nodes, item_nodes])
    cols = np.concatenate([item_nodes, user_nodes])
    values = np.ones(len(rows), dtype=np.float32)

    adj = sp.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = adj.tocsr()

    # D^-1/2, leaving nodes with no edges at 0 instead of inf
    degree = np.asarray(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    d_inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    d_mat = sp.diags(d_inv_sqrt)

    norm_adj = d_mat @ adj @ d_mat
    norm_adj = norm_adj.tocsr()
    norm_adj = norm_adj.astype(np.float32)

    crow_indices = torch.from_numpy(norm_adj.indptr.astype(np.int64))
    col_indices = torch.from_numpy(norm_adj.indices.astype(np.int64))
    values = torch.from_numpy(norm_adj.data)

    adj_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(n_nodes, n_nodes))
    return adj_tensor


class LightGCN(nn.Module):

    def __init__(self, n_users: int, n_items: int, adj: torch.Tensor, dim: int = 64, n_layers: int = 3):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        # one table for users and items together, users first
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

        self.register_buffer("adj", adj, persistent=False)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb.weight
        total = x

        for layer in range(self.n_layers):
            x = self.adj @ x
            total = total + x

        final_emb = total / (self.n_layers + 1)

        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        return user_emb, item_emb

    def bpr_loss(self, users, pos, neg, reg: float):
        user_emb, item_emb = self.propagate()

        u = user_emb[users]
        pos_items = item_emb[pos]
        neg_items = item_emb[neg]

        pos_scores = (u * pos_items).sum(dim=-1)
        neg_scores = (u * neg_items).sum(dim=-1)

        # softplus(neg - pos) is the same as -log(sigmoid(pos - neg))
        loss = F.softplus(neg_scores - pos_scores)
        loss = loss.mean()

        # L2 regularization on the layer-0 embeddings, like in the paper
        ego = self.emb.weight
        user_ego = ego[users]
        pos_ego = ego[self.n_users + pos]
        neg_ego = ego[self.n_users + neg]

        l2 = user_ego.pow(2).sum()
        l2 = l2 + pos_ego.pow(2).sum()
        l2 = l2 + neg_ego.pow(2).sum()
        l2 = l2 / (2 * len(users))

        reg_loss = reg * l2
        return loss, reg_loss

    @torch.no_grad()
    def embeddings(self):
        return self.propagate()
