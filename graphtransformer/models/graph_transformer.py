"""Graph-transformer expert: candidate-aware attention over a user's ego-graph.

Graph view
----------
For user u, the ego-graph is u plus its 1-hop book neighbours (the "library").
  * nodes   -> tokens: a [USER] token + one token per library book
  * edges   -> edge-type encodings: the star rating (1..5) of each user-book edge
  * position-> recency encoding: how recently u rated that book (log buckets)
  * 2-hop   -> book embeddings are smoothed over the book-book co-rating graph
               (books rated by the same users; built from train edges only)
A transformer encoder attends over the whole ego-graph (fully connected attention
within the subgraph, as in ego-graph graph transformers). A candidate book v then
cross-attends over the encoded library, so it is scored mostly against the parts of
the library that are relevant to it (e.g. a history book vs. the user's history
reads), instead of against one averaged user vector.

Inductive: a user is defined purely by their library, so new users need no
retraining. Books combine an ID embedding (randomly dropped in training to simulate
rarely-rated books) with the shared content features from common/process.py (tags, author,
series, language, year bucket, numeric).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import BookFeatures
from ..sampling import N_RECENCY


class BookEncoder(nn.Module):
    def __init__(self, feats: BookFeatures, n_books: int, d: int, id_dropout: float = 0.1,
                 nbrs: torch.Tensor | None = None, n_graph_nbrs: int = 20):
        super().__init__()
        self.n_books, self.id_dropout = n_books, id_dropout
        assert feats.tags.shape[0] == n_books, "features and interactions come from different runs"
        self.register_buffer("tags", torch.as_tensor(feats.tags, dtype=torch.float32))
        self.register_buffer("numeric", torch.as_tensor(feats.numeric, dtype=torch.float32))
        self.register_buffer("author", torch.as_tensor(feats.author))
        self.register_buffer("series", torch.as_tensor(feats.series))
        self.register_buffer("language", torch.as_tensor(feats.language))
        self.register_buffer("year", torch.as_tensor(feats.year))
        self.id_emb = nn.Embedding(n_books, d)
        nn.init.normal_(self.id_emb.weight, std=0.02)
        self.tag_proj = nn.Linear(feats.tags.shape[1], d)
        self.num_proj = nn.Linear(feats.numeric.shape[1], d) if feats.numeric.shape[1] else None
        self.author_emb = nn.Embedding(feats.n_authors, d)
        self.series_emb = nn.Embedding(feats.n_series, d)
        self.lang_emb = nn.Embedding(feats.n_languages, d)
        self.year_emb = nn.Embedding(feats.n_year_buckets, d)
        self.norm = nn.LayerNorm(d)
        if nbrs is not None:
            self.register_buffer("nbrs", torch.as_tensor(nbrs[:, :n_graph_nbrs], dtype=torch.long))
            self.graph_proj = nn.Linear(d, d)
            self.graph_norm = nn.LayerNorm(d)
        else:
            self.nbrs = None

    def forward(self) -> torch.Tensor:
        """Embeddings for all books plus a zero padding row: [n_books + 1, d]."""
        content = (self.tag_proj(self.tags) + self.author_emb(self.author)
                   + self.series_emb(self.series) + self.lang_emb(self.language)
                   + self.year_emb(self.year))
        if self.num_proj is not None:
            content = content + self.num_proj(self.numeric)
        ids = self.id_emb.weight
        if self.training and self.id_dropout > 0:
            keep = (torch.rand(self.n_books, 1, device=ids.device) >= self.id_dropout).float()
            ids = ids * keep
        e = self.norm(content + ids)
        if self.nbrs is not None:   # one round of book-book (2-hop) message passing
            e = self.graph_norm(e + self.graph_proj(e[self.nbrs].mean(1)))
        return torch.cat([e, e.new_zeros(1, e.shape[1])], 0)


class CandidateCrossAttention(nn.Module):
    """Multi-head attention with queries = candidate books, keys/values = ego-graph tokens."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)
        self.norm_q, self.norm_kv = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, kv, kv_mask, return_attn: bool = False):
        """q [B,C,d], kv [B,L,d], kv_mask [B,L] (True = attend)."""
        B, C, _ = q.shape
        L = kv.shape[1]
        qn, kvn = self.norm_q(q), self.norm_kv(kv)
        Q = self.q(qn).view(B, C, self.h, self.dh)
        K = self.k(kvn).view(B, L, self.h, self.dh)
        V = self.v(kvn).view(B, L, self.h, self.dh)
        att = torch.einsum("bchd,blhd->bhcl", Q, K) / math.sqrt(self.dh)
        att = att.masked_fill(~kv_mask[:, None, None, :], float("-inf")).softmax(-1)
        out = torch.einsum("bhcl,blhd->bchd", self.drop(att), V).reshape(B, C, -1)
        out = q + self.o(out)
        return (out, att) if return_attn else out


class GraphTransformerRecommender(nn.Module):
    def __init__(self, feats: BookFeatures, n_books: int, d: int = 64, heads: int = 4,
                 layers: int = 2, dropout: float = 0.1, id_dropout: float = 0.1,
                 nbrs=None, n_graph_nbrs: int = 20):
        super().__init__()
        self.n_books = n_books
        self.books = BookEncoder(feats, n_books, d, id_dropout, nbrs, n_graph_nbrs)
        self.rating_emb = nn.Embedding(6, d)          # 0 = pad, 1..5 stars (edge type)
        self.recency_emb = nn.Embedding(N_RECENCY, d)
        self.user_token = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.cross = CandidateCrossAttention(d, heads, dropout)
        self.pair = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, d), nn.GELU())
        self.score_head = nn.Linear(d, 1)    # ranking score (BPR)
        self.rating_head = nn.Linear(d, 5)   # P(1..5 stars) (auxiliary)

    def encode_user(self, E, ctx_book, ctx_rating, ctx_recency, ctx_mask):
        """Returns ego-graph tokens [B, 1+L, d] (index 0 = [USER]) and mask [B, 1+L]."""
        tok = E[ctx_book] + self.rating_emb(ctx_rating) + self.recency_emb(ctx_recency)
        B = tok.shape[0]
        tok = torch.cat([self.user_token.expand(B, 1, -1), tok], 1)
        mask = torch.cat([ctx_mask.new_ones(B, 1), ctx_mask], 1)
        return self.encoder(tok, src_key_padding_mask=~mask), mask

    def score(self, E, tokens, mask, cand_book):
        """Score candidates [B, C] -> (ranking score [B, C], rating logits [B, C, 5])."""
        q = E[cand_book]
        c = self.cross(q, tokens, mask)
        u = tokens[:, :1].expand_as(c)
        h = self.pair(torch.cat([c, q, c * q, u], -1))
        return self.score_head(h).squeeze(-1), self.rating_head(h)

    def forward(self, batch):
        E = self.books()
        tokens, mask = self.encode_user(E, batch.ctx_book, batch.ctx_rating,
                                        batch.ctx_recency, batch.ctx_mask)
        return self.score(E, tokens, mask, batch.cand_book)
