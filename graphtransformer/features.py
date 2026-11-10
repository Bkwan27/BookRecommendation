"""Book content features from books.csv / book_tags.csv.

Deliberately excluded: average_rating, ratings_count, work_ratings_count,
work_text_reviews_count, ratings_1..5. They are computed from *all* ratings,
including held-out ones, so using them would leak test labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, N_BOOKS

# Shelf names that describe the reader's logistics or reading status, not the book.
_NON_CONTENT = re.compile(
    r"(to-?read|currently-?reading|(my-|all-time-)?favou?rites?|.*own.*|kindle|e-?books?|"
    r"audio-?books?|audible|library|book-?club|default|to-?buy|wish-?list|re-?read|dnf|"
    r"did-?not-?finish|abandoned|on-?hold|maybe|books?|read|read-in-\d+|\d{4}(-reads?)?|"
    r".*-\d{4}|my-books|have|series|unfinished|tbr|to-be-read|borrowed|bookshelf|"
    r"shelfari.*|goodreads.*|calibre|nook|ibooks|i-own|wishlist|must-read|favorite-books|"
    r".*library.*|audio.*|finished|paperbacks?|hardcovers?|hardback|reviewed|english|"
    # rating shelves ("5-stars", "five-stars", ...) are aggregate ratings in disguise: leak
    r".*stars?|.*-star-.*|one-star|two-star|three-star|four-star|five-star|loved|"
    r"loved-it|amazing|meh|disappointing|didn-t-like|not-for-me|dislike.*|"
    # sentiment shelves are also aggregate opinions
    r".*favou?rite.*|.*(couldn-?t|could-not|never|didn-?t)-finish.*|stopped-reading|partially-read|re-?reads?|to-re-?read|"
    r"read-again|read-more-than-once|recommended|must-reads?|best-books.*)"
)

YEAR_EDGES = np.array([1800, 1900, 1950, 1970, 1980, 1990, 1995, 2000, 2003,
                       2006, 2008, 2010, 2012, 2014])


@dataclass
class BookFeatures:
    tags: np.ndarray       # float32 [N_BOOKS, n_tags], tf-idf on log counts, L2-normalised
    author: np.ndarray     # int64 [N_BOOKS], 0 = unknown / single-book author
    language: np.ndarray   # int64 [N_BOOKS], 0 = unknown
    year: np.ndarray       # int64 [N_BOOKS], bucket id, 0 = unknown
    tag_names: list[str]
    n_authors: int
    n_languages: int
    n_year_buckets: int = len(YEAR_EDGES) + 2


def year_bucket(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    b = np.digitize(np.nan_to_num(y, nan=0.0), YEAR_EDGES) + 1   # 1..len+1
    b[np.isnan(y)] = 0
    return b.astype(np.int64)


def load_book_features(data_dir: Path | str = DATA_DIR, n_tags: int = 512,
                       min_author_books: int = 2) -> BookFeatures:
    data_dir = Path(data_dir)
    books = pd.read_csv(data_dir / "books.csv")
    assert (books.book_id.to_numpy() == np.arange(1, N_BOOKS + 1)).all()
    bt = pd.read_csv(data_dir / "book_tags.csv")
    tags = pd.read_csv(data_dir / "tags.csv")

    # book_tags uses goodreads_book_id -> map to our 0-based book index.
    gr2idx = pd.Series(np.arange(N_BOOKS), index=books.goodreads_book_id.to_numpy())
    bt = bt[bt.goodreads_book_id.isin(gr2idx.index)].copy()
    bt["book"] = gr2idx.loc[bt.goodreads_book_id].to_numpy()
    bt["count"] = bt["count"].clip(lower=0)
    bt = bt.merge(tags, on="tag_id")
    bt = bt[~bt.tag_name.str.fullmatch(_NON_CONTENT)]
    bt = bt.groupby(["book", "tag_id"], as_index=False)["count"].sum()

    # Keep the tags present on the most books (document frequency), then tf-idf.
    df_ = bt.groupby("tag_id").book.nunique().sort_values(ascending=False)
    keep = df_.index[:n_tags]
    col = pd.Series(np.arange(len(keep)), index=keep)
    sub = bt[bt.tag_id.isin(keep)]
    X = np.zeros((N_BOOKS, len(keep)), dtype=np.float32)
    X[sub.book.to_numpy(), col.loc[sub.tag_id].to_numpy()] = np.log1p(sub["count"].to_numpy())
    idf = np.log(N_BOOKS / (1.0 + df_.loc[keep].to_numpy())).astype(np.float32)
    X *= idf[None, :]
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-6)
    tag_names = tags.set_index("tag_id").loc[keep, "tag_name"].tolist()

    # First listed author; authors with < min_author_books books share id 0.
    first = books.authors.fillna("").str.split(",").str[0].str.strip()
    vc = first.value_counts()
    amap = {a: i + 1 for i, a in enumerate(sorted(vc[vc >= min_author_books].index))}
    author = first.map(amap).fillna(0).astype(np.int64).to_numpy()

    lang = books.language_code.fillna("").str.lower().replace(
        {"en-us": "eng", "en-gb": "eng", "en-ca": "eng", "en": "eng"})
    lmap = {l: i + 1 for i, l in enumerate(sorted(set(lang) - {""}))}
    language = lang.map(lmap).fillna(0).astype(np.int64).to_numpy()

    return BookFeatures(tags=X, author=author, language=language,
                        year=year_bucket(books.original_publication_year.to_numpy()),
                        tag_names=tag_names, n_authors=len(amap) + 1,
                        n_languages=len(lmap) + 1)
