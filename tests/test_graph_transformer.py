"""Fast tests on a synthetic dataset (no CSVs needed)."""
import numpy as np
import pytest
import torch

from graphtransformer.data import TEST, TRAIN, VAL, PairSet, Ratings, build_dataset, make_split
from graphtransformer.evaluate import evaluate
from graphtransformer.features import BookFeatures
from graphtransformer.losses import LossConfig, bpr_loss, graded_loss, objective
from graphtransformer.models.graph_transformer import GraphTransformerRecommender
from graphtransformer.sampling import TrainSampler, corated_neighbors

N_USERS, N_BOOKS = 40, 60


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    users, books = [], []
    for u in range(N_USERS):
        bs = rng.choice(N_BOOKS, rng.integers(10, 30), replace=False)
        users += [u] * len(bs)
        books += bs.tolist()
    perm = rng.permutation(len(users))           # interleave users in "time", like ratings.csv
    r = Ratings(user=np.array(users)[perm], book=np.array(books)[perm],
                rating=rng.integers(1, 6, len(users)), time=np.arange(len(users)),
                n_users=N_USERS, n_books=N_BOOKS)
    split = make_split(r, "temporal")
    tr_u = rng.integers(0, N_USERS, 200)
    tr_b = rng.integers(0, N_BOOKS, 200)
    ds = build_dataset(r, split, (tr_u, tr_b))
    feats = BookFeatures(tags=rng.random((N_BOOKS, 8)).astype(np.float32),
                         author=rng.integers(0, 5, N_BOOKS), language=np.zeros(N_BOOKS, np.int64),
                         year=rng.integers(0, 16, N_BOOKS), tag_names=[str(i) for i in range(8)],
                         n_authors=5, n_languages=1)
    nbrs = corated_neighbors(ds.train, N_BOOKS, k=20)
    return r, split, ds, feats, nbrs


def test_temporal_split_is_per_user_and_later(synth):
    r, split, ds, *_ = synth
    for u in range(N_USERS):
        m = r.user == u
        t, s = r.time[m], split[m]
        assert (s == TRAIN).sum() >= 1 and (s == TEST).sum() >= 1
        assert t[s == TRAIN].max() < t[s == VAL].min() <= t[s == VAL].max() < t[s == TEST].min()


def test_to_read_does_not_leak_heldout(synth):
    r, split, ds, *_ = synth
    held = PairSet(r.user[split != TRAIN], r.book[split != TRAIN], N_BOOKS)
    users = np.repeat(np.arange(N_USERS), ds.to_read.counts())
    assert not held.contains(users, ds.to_read.book).any()


def test_neighbors_exclude_self(synth):
    *_, nbrs = synth
    assert nbrs.shape == (N_BOOKS, 20)
    assert not (nbrs == np.arange(N_BOOKS)[:, None]).any()


def test_sampler_targets_hidden_and_negatives_valid(synth):
    r, split, ds, feats, nbrs = synth
    s = TrainSampler(ds, nbrs, max_context=16, n_rand_neg=8, hard_rank=(2, 20))
    s.n_hard = 4
    users = np.arange(N_USERS)
    b = s.sample(users)
    for i, u in enumerate(users):
        ctx = set(b.ctx_book[i][b.ctx_mask[i]].tolist())
        tgt = set(b.tgt_book[i][b.tgt_rating[i] > 0].tolist())
        assert ctx and tgt and not (ctx & tgt), "targets must be hidden from the context"
        train_books = set(ds.train.get(u)[0].tolist())
        assert ctx | tgt <= train_books
        negs = b.neg_book[i][b.neg_mask[i]].tolist()
        assert not (set(negs) & train_books), "valid negatives must be unrated"
    assert b.neg_is_hard.sum() == 4 * N_USERS


def test_bpr_prefers_correct_order():
    pos = torch.tensor([[2.0]]); neg = torch.tensor([[0.0]])
    good = bpr_loss(pos, torch.ones(1, 1, dtype=torch.bool), neg, torch.ones(1, 1))
    bad = bpr_loss(neg, torch.ones(1, 1, dtype=torch.bool), pos, torch.ones(1, 1))
    assert good < bad
    assert torch.isclose(good, torch.nn.functional.softplus(torch.tensor(-2.0)))


def test_bpr_is_per_user_mean():
    # user 0 has 1 pair, user 1 has 4 pairs; per-user averaging weights users equally.
    pos = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    pm = torch.tensor([[True, False], [True, True]])
    neg = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    nw = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    sp = torch.nn.functional.softplus
    expect = (sp(torch.tensor(0.0)) + sp(torch.tensor(5.0))) / 2
    assert torch.isclose(bpr_loss(pos, pm, neg, nw), expect)


def test_graded_only_uses_large_gaps():
    score = torch.tensor([[0.0, 0.0, 0.0]])
    rating = torch.tensor([[5, 4, 3]])
    # gap >= 2 -> only (5, 3) counts
    assert torch.isclose(graded_loss(score, rating, 2), torch.log(torch.tensor(2.0)))
    assert graded_loss(score, torch.tensor([[5, 5, 0]]), 2) == 0


def test_model_trains_and_evaluates(synth):
    r, split, ds, feats, nbrs = synth
    torch.manual_seed(0)
    model = GraphTransformerRecommender(feats, N_BOOKS, d=16, heads=2, layers=1, nbrs=nbrs,
                                        n_graph_nbrs=5)
    s = TrainSampler(ds, nbrs, max_context=16, n_rand_neg=8, hard_rank=(2, 20))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = s.sample(np.arange(N_USERS))
    score, logits = model(batch)
    assert score.shape == batch.cand_book.shape and logits.shape == (*score.shape, 5)
    first = None
    for _ in range(30):   # overfit one batch: loss must go down
        loss, parts = objective(*model(batch), batch, LossConfig())
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first
    res = evaluate(model, ds, ds.val, np.arange(N_USERS), device="cpu", ks=(5, 10),
                   batch_users=8, cand_chunk=25)
    assert 0 <= res["recall@10"] <= 1 and res["rmse"] > 0 and res["n_users"] > 0


def test_padding_does_not_change_scores(synth):
    """Extra padded context tokens must not affect a user's scores."""
    r, split, ds, feats, nbrs = synth
    model = GraphTransformerRecommender(feats, N_BOOKS, d=16, heads=2, layers=1).eval()
    E = model.books()
    ctx = torch.tensor([[1, 2, 3]]); rat = torch.tensor([[5, 3, 1]]); rec = torch.tensor([[2, 1, 0]])
    cand = torch.tensor([[7, 8]])
    t1, m1 = model.encode_user(E, ctx, rat, rec, torch.ones(1, 3, dtype=torch.bool))
    s1 = model.score(E, t1, m1, cand)[0]
    pad = torch.full((1, 2), N_BOOKS)
    t2, m2 = model.encode_user(E, torch.cat([ctx, pad], 1), torch.cat([rat, torch.zeros(1, 2, dtype=torch.long)], 1),
                               torch.cat([rec, torch.zeros(1, 2, dtype=torch.long)], 1),
                               torch.tensor([[True, True, True, False, False]]))
    s2 = model.score(E, t2, m2, cand)[0]
    assert torch.allclose(s1, s2, atol=1e-5)
