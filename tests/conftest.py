"""A small synthetic goodbooks-like dataset, pushed through the REAL shared pipeline
(common.process.main), so tests exercise exactly what every expert loads."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.process import Config, main

N_USERS, N_BOOKS = 60, 80
CONTENT_TAGS = ["fantasy", "romance", "mystery", "science-fiction", "history", "young-adult",
                "thriller", "classics"]
NOISE_TAGS = ["to-read", "favorites", "5-stars", "owned", "kindle"]


def write_raw(raw: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    raw.mkdir(parents=True, exist_ok=True)
    users, books = [], []
    for u in range(1, N_USERS + 1):
        bs = rng.choice(np.arange(1, N_BOOKS + 1), rng.integers(25, 41), replace=False)
        users += [u] * len(bs)
        books += bs.tolist()
    perm = rng.permutation(len(users))              # interleaved in "time", like ratings.csv
    stars = rng.choice([1, 2, 3, 4, 5], len(users), p=[0.05, 0.1, 0.2, 0.35, 0.3])
    pd.DataFrame({"user_id": np.array(users)[perm], "book_id": np.array(books)[perm],
                  "rating": stars}).to_csv(raw / "ratings.csv", index=False)

    ids = np.arange(1, N_BOOKS + 1)
    titles = [f"Book {i} (Saga, #{i % 5 + 1})" if i % 3 == 0 else f"Book {i}" for i in ids]
    hist = rng.integers(1, 100, (N_BOOKS, 5))
    pd.DataFrame({
        "book_id": ids, "goodreads_book_id": ids * 7, "best_book_id": ids * 7, "work_id": ids * 11,
        "books_count": rng.integers(1, 50, N_BOOKS), "isbn": ["123456789"] * N_BOOKS,
        "isbn13": [9.78e12] * N_BOOKS, "authors": [f"Author {i % 12}, Illustrator" for i in ids],
        "original_publication_year": np.where(ids % 17 == 0, np.nan, 1950 + ids % 60),
        "original_title": titles, "title": titles,
        "language_code": rng.choice(["eng", "en-US", "fre", None], N_BOOKS),
        "average_rating": 3.9, "ratings_count": hist.sum(1), "work_ratings_count": hist.sum(1),
        "work_text_reviews_count": rng.integers(1, 30, N_BOOKS),
        **{f"ratings_{k}": hist[:, k - 1] for k in range(1, 6)},
        "image_url": "http://x/img.jpg", "small_image_url": "http://x/s.jpg",
    }).to_csv(raw / "books.csv", index=False)

    tags = CONTENT_TAGS + NOISE_TAGS
    pd.DataFrame({"tag_id": np.arange(len(tags)), "tag_name": tags}).to_csv(raw / "tags.csv", index=False)
    rows = []
    for i in ids:
        for t in rng.choice(len(CONTENT_TAGS), 4, replace=False):
            rows.append((i * 7, int(t), int(rng.integers(1, 500))))
        for t in range(len(CONTENT_TAGS), len(tags)):
            rows.append((i * 7, t, int(rng.integers(1, 500))))
    pd.DataFrame(rows, columns=["goodreads_book_id", "tag_id", "count"]).to_csv(raw / "book_tags.csv", index=False)
    pd.DataFrame({"user_id": rng.integers(1, N_USERS + 1, 300),
                  "book_id": rng.integers(1, N_BOOKS + 1, 300)}).to_csv(raw / "to_read.csv", index=False)


def run_pipeline(tmp: Path, name: str = "processed", **overrides) -> Path:
    raw = tmp / "raw"
    if not (raw / "ratings.csv").exists():
        write_raw(raw)
    out = tmp / name
    cfg = dict(raw_dir=str(raw), out_dir=str(out), min_user_pos=5, min_item_pos=5, min_books_per_tag=5,
               min_books_per_language=1)
    cfg.update(overrides)
    main(Config(**cfg))
    return out


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory) -> Path:
    return run_pipeline(tmp_path_factory.mktemp("goodbooks"))
