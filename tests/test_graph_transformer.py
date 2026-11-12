"""Graph-transformer tests on the shared synthetic processed data (see conftest.py)."""
import numpy as np
import pytest
import torch

import graphtransformer.evaluate as gt_eval
from graphtransformer.data import load_dataset
from graphtransformer.evaluate import evaluate
from graphtransformer.features import load_book_features
from graphtransformer.losses import LossConfig, bpr_loss, graded_loss, objective
from graphtransformer.models.graph_transformer import GraphTransformerRecommender
from graphtransformer.sampling import TrainSampler, corated_neighbors


@pytest.fixture(scope="module")
def setup(processed_dir):
    ds = load_dataset(processed_dir)
    feats = load_book_features(processed_dir)
    nbrs = corated_neighbors(ds.train, ds.n_items, k=20)
    return ds, feats, nbrs


def _small_model(ds, feats, nbrs=None):
    torch.manual_seed(0)
    return GraphTransformerRecommender(feats, ds.n_items, d=16, heads=2, layers=1, nbrs=nbrs, n_graph_nbrs=5)


def test_dataset_uses_shared_processing(setup, processed_dir):
    from common.process import load_processed
    ds, feats, _ = setup
    d = load_processed(processed_dir)
    assert ds.fingerprint == d["fingerprint"] and ds.n_items == d["n_items"] == feats.tags.shape[0]
    assert len(ds.train.book) == len(d["train_all"]["item"])     # all stars, not just positives
    assert ds.to_read is None                                      # off in the shared protocol


def test_neighbors_exclude_self(setup):
    ds, _, nbrs = setup
    assert nbrs.shape == (ds.n_items, 20)
    assert not (nbrs == np.arange(ds.n_items)[:, None]).any()


def test_sampler_targets_hidden_and_negatives_valid(setup):
    ds, _, nbrs = setup
    s = TrainSampler(ds, nbrs, max_context=16, n_rand_neg=8, hard_rank=(2, 20))
    s.n_hard = 4
    users = s.trainable_users()
    b = s.sample(users)
    for i, u in enumerate(users):
        ctx = set(b.ctx_book[i][b.ctx_mask[i]].tolist())
        tgt = set(b.tgt_book[i][b.tgt_rating[i] > 0].tolist())
        assert ctx and tgt and not (ctx & tgt), "targets must be hidden from the context"
        train_books = set(ds.train.get(u)[0].tolist())
        assert ctx | tgt <= train_books, "only train ratings may be used as input or targets"
        negs = b.neg_book[i][b.neg_mask[i]].tolist()
        assert not (set(negs) & train_books), "valid negatives must be unrated"
    assert b.neg_is_hard.sum() == 4 * len(users)


def test_bpr_prefers_correct_order():
    pos, neg = torch.tensor([[2.0]]), torch.tensor([[0.0]])
    one = torch.ones(1, 1, dtype=torch.bool)
    good, bad = bpr_loss(pos, one, neg, torch.ones(1, 1)), bpr_loss(neg, one, pos, torch.ones(1, 1))
    assert good < bad
    assert torch.isclose(good, torch.nn.functional.softplus(torch.tensor(-2.0)))


def test_bpr_is_per_user_mean():
    pos = torch.zeros(2, 2)
    pm = torch.tensor([[True, False], [True, True]])
    neg = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    nw = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    sp_ = torch.nn.functional.softplus
    assert torch.isclose(bpr_loss(pos, pm, neg, nw), (sp_(torch.tensor(0.0)) + sp_(torch.tensor(5.0))) / 2)


def test_graded_only_uses_large_gaps():
    score = torch.zeros(1, 3)
    assert torch.isclose(graded_loss(score, torch.tensor([[5, 4, 3]]), 2), torch.log(torch.tensor(2.0)))
    assert graded_loss(score, torch.tensor([[5, 5, 0]]), 2) == 0


def test_model_trains_and_evaluates(setup):
    ds, feats, nbrs = setup
    model = _small_model(ds, feats, nbrs)
    s = TrainSampler(ds, nbrs, max_context=16, n_rand_neg=8, hard_rank=(2, 20))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = s.sample(s.trainable_users())
    score, logits = model(batch)
    assert score.shape == batch.cand_book.shape and logits.shape == (*score.shape, 5)
    first = None
    for _ in range(30):   # overfit one batch: loss must go down
        loss, _ = objective(*model(batch), batch, LossConfig())
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first
    for split in ("val", "test"):
        res = evaluate(model, ds, split, device="cpu", ks=(5, 10), batch_users=8, cand_chunk=25)
        assert 0 <= res["recall@10"] <= 1 and res["rmse"] > 0 and res["n_users"] > 0


def test_test_eval_masks_val_period_books(setup, monkeypatch):
    """Regression: test-time ranking must exclude books rated in train AND val (any stars)."""
    ds, feats, _ = setup
    seen = {}

    def spy(score_fn, target, exclude, **kw):
        seen["target"], seen["exclude"] = target, exclude
        return {"recall@10": 0.0, "n_users": 0}

    monkeypatch.setattr(gt_eval, "ranking_metrics", spy)
    evaluate(_small_model(ds, feats), ds, "test", device="cpu", ks=(10,), batch_users=8, cand_chunk=25)
    ex = seen["exclude"].tocoo()
    excluded = set(zip(ex.row.tolist(), ex.col.tolist()))
    for hist in (ds.train, ds.val):
        users = np.repeat(np.arange(ds.n_users), hist.counts())
        assert set(zip(users.tolist(), hist.book.tolist())) <= excluded
    assert seen["target"] is ds.targets["test"]


def test_eval_context_is_train_only(setup, monkeypatch):
    """Val/test ratings must never enter the model's input history at evaluation time."""
    ds, feats, _ = setup
    model = _small_model(ds, feats)
    used = []
    orig = model.encode_user

    def spy(E, ctx_book, *a):
        used.append(ctx_book.clone())
        return orig(E, ctx_book, *a)

    monkeypatch.setattr(model, "encode_user", spy)
    users = np.arange(min(8, ds.n_users))
    evaluate(model, ds, "test", users, device="cpu", ks=(5,), batch_users=8, cand_chunk=25)
    ctx = used[0]
    for i, u in enumerate(users):
        books = set(ctx[i][ctx[i] < ds.n_items].tolist())
        assert books <= set(ds.train.get(u)[0].tolist())


def test_padding_does_not_change_scores(setup):
    ds, feats, _ = setup
    model = _small_model(ds, feats).eval()
    E = model.books()
    ctx = torch.tensor([[1, 2, 3]]); rat = torch.tensor([[5, 3, 1]]); rec = torch.tensor([[2, 1, 0]])
    cand = torch.tensor([[7, 8]])
    t1, m1 = model.encode_user(E, ctx, rat, rec, torch.ones(1, 3, dtype=torch.bool))
    s1 = model.score(E, t1, m1, cand)[0]
    pad, z = torch.full((1, 2), ds.n_items), torch.zeros(1, 2, dtype=torch.long)
    t2, m2 = model.encode_user(E, torch.cat([ctx, pad], 1), torch.cat([rat, z], 1), torch.cat([rec, z], 1),
                               torch.tensor([[True, True, True, False, False]]))
    assert torch.allclose(s1, model.score(E, t2, m2, cand)[0], atol=1e-5)
