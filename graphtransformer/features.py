"""Book features for the graph transformer, read from the SHARED processed output
(common.process.load_item_features). The tag cleaning, vocabularies and leakage rules live in
common/process.py so PinSage and the graph transformer see identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.process import DEFAULT_PROCESSED, load_item_features


@dataclass
class BookFeatures:
    tags: np.ndarray        # float32 [n_items, n_tags], weights in [0, 1]
    numeric: np.ndarray     # float32 [n_items, k] z-scored (k may be 0)
    author: np.ndarray      # int64 [n_items], 0 = rare/unknown
    series: np.ndarray
    language: np.ndarray
    year: np.ndarray
    n_authors: int
    n_series: int
    n_languages: int
    n_year_buckets: int


def book_features_from(f: dict) -> BookFeatures:
    return BookFeatures(
        tags=f["tags"].toarray().astype(np.float32),
        numeric=np.asarray(f["numeric"], np.float32).reshape(f["tags"].shape[0], -1),
        author=f["author"].astype(np.int64), series=f["series"].astype(np.int64),
        language=f["language"].astype(np.int64), year=f["year_bucket"].astype(np.int64),
        n_authors=len(f["vocab_author"]), n_series=len(f["vocab_series"]),
        n_languages=len(f["vocab_language"]), n_year_buckets=len(f["vocab_year_bucket"]),
    )


def load_book_features(processed_dir: str | Path = DEFAULT_PROCESSED) -> BookFeatures:
    return book_features_from(load_item_features(processed_dir))
