"""All three experts load the SAME processed data, train briefly, and are scored by the SAME metrics.
Guards against the experts drifting apart (import paths, ids, masks, score shapes)."""
import numpy as np
import torch

from common.expert import standardize_per_user
from common.metrics import popularity_score_fn, ranking_metrics
from common.process import load_item_features, load_processed


def _check_scorer(name, score_fn, data):
    users = np.flatnonzero(np.diff(data["test"].indptr))[:8]
    out = score_fn(users)
    assert out.shape == (len(users), data["n_items"]), name
    assert torch.isfinite(standardize_per_user(out.float())).all(), name
    for split in ("val", "test"):
        r = ranking_metrics(score_fn, data[split], data[f"seen_before_{split}"], ks=(5, 10))
        assert 0.0 <= r["recall@10"] <= 1.0 and r["n_users"] > 0, (name, split)


def test_lightgcn_on_shared_data(processed_dir):
    from lightgcn import data as ldata
    from lightgcn.metrics import evaluate as lgcn_evaluate
    from lightgcn.model import LightGCN, normalized_adj
    from lightgcn.train import make_scorer, sample_negatives

    data = load_processed(processed_dir)
    ds = ldata.load(str(processed_dir))
    assert ds.fingerprint == data["fingerprint"] and (ds.test_exclude != data["seen_before_test"]).nnz == 0
    model = LightGCN(ds.n_users, ds.n_items, normalized_adj(ds.train, ds.n_users, ds.n_items), 8, 2)
    opt, rng = torch.optim.Adam(model.parameters(), 1e-2), np.random.default_rng(0)
    users, pos = ds.train[:, 0], ds.train[:, 1]
    neg = sample_negatives(users, ds.train_csr, ds.n_items, rng)
    loss, reg = model.bpr_loss(*(torch.from_numpy(x) for x in (users, pos, neg)), 1e-4)
    opt.zero_grad(); (loss + reg).backward(); opt.step()
    model.eval()
    _check_scorer("lightgcn", make_scorer(model), data)
    # Bryan's metric implementation and the shared one agree
    a = lgcn_evaluate(popularity_score_fn(data), data["test"], data["seen_before_test"])
    b = ranking_metrics(popularity_score_fn(data), data["test"], data["seen_before_test"])
    assert all(abs(a[k] - b[k]) < 1e-6 for k in a)


def test_pinsage_on_shared_data(processed_dir):
    from pinsage.pinsage import Config, fit, scorer

    data, feats = load_processed(processed_dir), load_item_features(processed_dir)
    cfg = Config(data=str(processed_dir), epochs=1, pairs_per_epoch=256, batch_size=128,
                 num_negatives=16, num_walks=5, num_neighbors=3, dim=8)
    model, hist, _ = fit(cfg, data, feats, torch.device("cpu"))
    assert (hist.multiply(data["seen_before_val"]) != hist).nnz == 0   # query history is train-only
    _check_scorer("pinsage", scorer(model, hist), data)


def test_graph_transformer_on_shared_data(processed_dir):
    from graphtransformer.data import from_processed
    from graphtransformer.evaluate import make_score_fn
    from graphtransformer.features import book_features_from
    from graphtransformer.models.graph_transformer import GraphTransformerRecommender

    data, feats = load_processed(processed_dir), load_item_features(processed_dir)
    ds = from_processed(data)
    model = GraphTransformerRecommender(book_features_from(feats), ds.n_items, d=8, heads=2, layers=1).eval()
    score_fn, _ = make_score_fn(model, ds, device="cpu", cand_chunk=25)
    _check_scorer("graph_transformer", torch.no_grad()(score_fn), data)
