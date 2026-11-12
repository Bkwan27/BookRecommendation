"""Invariants of the shared pipeline (common/) that every expert relies on."""
import numpy as np
import scipy.sparse as sp
import torch

from common.expert import standardize_per_user
from common.metrics import ranking_metrics
from common.process import load_item_features, load_processed

from conftest import run_pipeline


def _pairs(m: sp.csr_matrix) -> set:
    c = m.tocoo()
    return set(zip(c.row.tolist(), c.col.tolist()))


def _rows(d: dict) -> set:
    return set(zip(d["user"].tolist(), d["item"].tolist()))


def test_split_is_per_user_temporal(processed_dir):
    d = load_processed(processed_dir)
    tr, va, te = d["train_all"], d["val_rows"], d["test_rows"]
    for u in range(d["n_users"]):
        t_tr, t_va, t_te = (x["t"][x["user"] == u] for x in (tr, va, te))
        assert len(t_tr) and len(t_va) and len(t_te)
        assert t_tr.max() < t_va.min() and t_va.max() < t_te.min()


def test_masks_cover_all_stars_and_val_is_masked_at_test(processed_dir):
    d = load_processed(processed_dir)
    train_all, val_all = _rows(d["train_all"]), _rows(d["val_rows"])
    assert _pairs(d["seen_before_val"]) == train_all                  # any stars, not just positives
    assert _pairs(d["seen_before_test"]) == train_all | val_all       # val-period books masked at test
    assert (d["train_all"]["rating"] < d["positive_threshold"]).any()  # low ratings are kept
    assert _pairs(d["train"]) <= train_all


def test_targets_are_heldout_warm_positives(processed_dir):
    d = load_processed(processed_dir)
    train_all = _rows(d["train_all"])
    for split in ("val", "test"):
        tgt = _pairs(d[split])
        held = {(u, i): r for u, i, r in zip(d[f"{split}_rows"]["user"].tolist(),
                                            d[f"{split}_rows"]["item"].tolist(),
                                            d[f"{split}_rows"]["rating"].tolist())}
        assert tgt and not (tgt & train_all)
        assert all(held[p] >= d["positive_threshold"] for p in tgt)


def test_no_popularity_leak_by_default(processed_dir):
    f = load_item_features(processed_dir)
    names = list(f["numeric_names"])
    assert not any(n.startswith(("frac_", "log_ratings", "log_text_reviews")) for n in names), names
    tags = open(processed_dir / "tag_vocab.csv").read()
    assert "5-stars" not in tags and "to-read" not in tags and "favorites" not in tags
    assert f["tags"].shape[0] == load_processed(processed_dir)["n_items"]


def test_fingerprint_is_deterministic_and_checked(processed_dir, tmp_path):
    a = load_processed(processed_dir)["fingerprint"]
    again = run_pipeline(processed_dir.parent, name="processed_again")
    assert load_processed(again)["fingerprint"] == a
    other = run_pipeline(processed_dir.parent, name="processed_random", split="random")
    assert load_processed(other)["fingerprint"] != a


def test_ranking_metrics_exclusion_and_values():
    # 1 user, 5 books. Book 0 is masked (already read) and scored highest; target = {1, 3}.
    target = sp.csr_matrix(np.array([[0, 1, 0, 1, 0]], np.float32))
    exclude = sp.csr_matrix(np.array([[1, 0, 0, 0, 0]], np.float32))
    scores = torch.tensor([[9.0, 5.0, 4.0, 3.0, 2.0]])
    m = ranking_metrics(lambda b: scores, target, exclude, ks=(1, 2, 3))
    # ranking without book 0: [1, 2, 3, 4] -> hits [1, 0, 1, 0]
    assert m["recall@1"] == 0.5 and m["recall@2"] == 0.5 and m["recall@3"] == 1.0
    assert np.isclose(m["ndcg@3"], (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
    unmasked = ranking_metrics(lambda b: scores, target, sp.csr_matrix(target.shape, dtype=np.float32), ks=(1,))
    assert unmasked["recall@1"] == 0.0   # without the mask the read book would take the top slot


def test_standardize_per_user():
    x = torch.tensor([[1.0, 2.0, 3.0, -float("inf")], [10.0, 10.0, 40.0, 70.0]])
    z = standardize_per_user(x)
    for row in z:
        v = row[torch.isfinite(row)]
        assert torch.isclose(v.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(v.pow(2).mean(), torch.tensor(1.0), atol=1e-3)
    assert z[0, 3] == -float("inf")
