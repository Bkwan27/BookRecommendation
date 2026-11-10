
#!/usr/bin/env python3
"""
goodbooks-10k preprocessing for graph recommenders (LightGCN, PinSage) and other models.

    python preprocess_goodbooks.py --download                  # fetch the raw CSVs, then preprocess
    python preprocess_goodbooks.py --raw-dir data/raw --out-dir data/processed --split global_temporal

Requires Python 3.9+, numpy, pandas, scipy. No deep-learning framework is needed: every output is a
plain numpy / scipy / CSV file that loads directly into PyTorch, PyG or DGL.

FACTS ABOUT THE RAW DATA THAT DRIVE THE CHOICES BELOW (each one was checked on the actual files)
  * ratings.csv: 5,976,479 ratings (1-5 stars) from 53,424 users on 10,000 books, no duplicate
    (user, book) pairs. Every user has 19-200 ratings and every book at least 8, so the matrix is
    already pre-filtered and fairly dense (1.1%).
  * There is no timestamp column, but rows are in chronological order across the whole file (the
    README says "sorted by time"). Verified two ways: books first published in 2017 receive their
    first rating ~69% of the way into the file and half of their ratings in the last 1%; and when a
    user rated volumes #k and #k+1 of a series, #k comes first in 76% of cases (88% when the two
    ratings are far apart in the file) versus 50% if the order were arbitrary. The row number is
    therefore used as a pseudo-timestamp, which makes temporal splits possible.
  * book_id in ratings.csv / to_read.csv identifies a *work* (all editions aggregated), while
    book_tags.csv uses goodreads_book_id, so tags have to be joined through books.csv.
  * book_tags.csv lists the ~100 most popular user shelves of every book. The most frequent shelves
    describe the reader, not the book ("to-read", "owned", "kindle", "favorites", "read-in-2015").
    It also contains 8 duplicated (book, tag) pairs and 6 rows with count -1.
  * books.csv: English is spread over 5 codes (eng, en-US, en-GB, en-CA, en) and 1,084 codes are
    missing; publication years run from -1750 to 2017 (21 missing); series membership exists only
    inside titles ("Title (Series, #2)"); 3,332 cover URLs are "nophoto" placeholders; isbn lost its
    leading zeros and isbn13 went through a float and lost digits.

OUTPUTS (in --out-dir). Index u / i denotes the same user / book in every file.
  interactions.npz   Source of truth: every kept rating (all star values), sorted by user then time.
                     Arrays: user, item, rating, t (row number in ratings.csv = time rank),
                     split (0 train, 1 val, 2 test), is_target (val/test rows used as ground truth).
  lightgcn/          train.txt, val.txt, test.txt with positives as "user item item ..." lines, the
                     format of the reference LightGCN / NGCF code (0-based ids).
  item_features.npz  PinSage inputs: z-scored numeric matrix (+ column names) and integer categorical
                     codes for author / series / language / year_bucket (+ their vocabularies).
  item_tags.npz      scipy.sparse CSR [n_items x n_tags] of tag weights in (0, 1]; see tag_vocab.csv.
  books.csv          cleaned metadata, row i = item i (clean title, series, authors, top tags, ...).
  users.csv          user index -> original user_id.
  to_read.npz        optional auxiliary "wants to read" edges (read the warning in process_to_read).
  stats.json         the configuration used and statistics after every stage.

DELIBERATELY NOT DONE HERE (it belongs in the training code)
  * Negative sampling: it must be re-drawn every epoch; a precomputed sample would be frozen forever.
  * Graph normalisation (e.g. LightGCN's D^-1/2 A D^-1/2): model-specific, built from train edges.
  * Random walks / neighbour sampling for PinSage: sampled on the fly during training.
  * Text or image embeddings: books.csv keeps the clean text and the cover URLs to compute them.

QUICK START IN TRAINING CODE (see load_processed() at the bottom of this file)
    data = load_processed("data/processed")
    coo = data["train"].tocoo()                         # positive training edges
    # PyG LightGCN: users are nodes 0..U-1, books U..U+I-1, message passing needs both directions
    ei = torch.tensor(np.vstack([coo.row, coo.col + data["n_users"]]), dtype=torch.long)
    edge_index = torch.cat([ei, ei.flip(0)], dim=1)
    # DGL PinSage: bipartite heterograph, then attach the item_features.npz arrays to "book" nodes
    g = dgl.heterograph({("user", "reads", "book"): (coo.row, coo.col),
                         ("book", "read-by", "user"): (coo.col, coo.row)},
                        num_nodes_dict={"user": data["n_users"], "book": data["n_items"]})
    # Evaluation: score all books, mask data["seen_before_val"] (or "seen_before_test"),
    # and compare the top-K against data["val"] (or data["test"]).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Pin a commit hash instead of "master" if you need byte-for-byte reproducibility of the download.
GITHUB_RAW = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
RAW_FILES = ("ratings.csv", "books.csv", "book_tags.csv", "tags.csv", "to_read.csv")
SPLITS = ("user_temporal", "global_temporal", "random")
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}  # int8 codes stored in interactions.npz


# ==============================================================================================
# Configuration
# ==============================================================================================
@dataclass
class Config:
    raw_dir: str = "data/raw"
    out_dir: str = "data/processed"
    download: bool = False

    # Implicit feedback. LightGCN (BPR loss) and PinSage (max-margin loss) learn from positive
    # interactions only, so each rating must become "liked" or "not a positive".
    # >= 4 keeps 69% of the ratings. Not >= 1 ("they read it"): a 1-2 star rating means the user read
    # the book and disliked it, and treating it as a positive teaches the model to recommend books
    # people dislike. Not >= 3 (Goodreads labels 3 stars "liked it"): the mean rating is 3.9, so
    # 3 stars sits below most users' own average and reads as lukewarm; >= 4 is also the common
    # convention for 5-star data (e.g. MovieLens), which keeps results comparable to the literature.
    # Splits are made on all ratings regardless of stars, so this threshold only decides which train
    # rows become graph edges and which val/test rows count as hits.
    positive_threshold: int = 4

    # k-core filtering on positives: every user and book must keep at least this many positives.
    # 10-core is the convention of the datasets LightGCN was benchmarked on (Gowalla, Yelp2018,
    # Amazon-Book). It guarantees every user has enough positives to spread over train/val/test and
    # every book enough edges to learn an embedding from. It is cheap here: it removes ~0.25% of
    # users and 4 books, because the dataset author already filtered the data.
    min_user_pos: int = 10
    min_item_pos: int = 10

    # How interactions are divided into train/val/test; see assign_splits() for the full reasoning.
    split: str = "user_temporal"
    # 80/10/10 per user. With 19-200 ratings per user this leaves 2-20 held-out books per user, which
    # keeps Recall/NDCG@20 meaningful. A separate validation split exists so early stopping and
    # hyper-parameter search never look at test (the reference LightGCN code has no validation
    # split and reports test metrics during training).
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42  # only used by split="random"; everything else is deterministic

    # Item-feature vocabularies. A category (tag, author, series, language) needs enough books to
    # learn a shared weight/embedding that generalises; below the threshold it goes to the shared
    # code 0 ("rare/unknown"). Code 0 also gives books unseen at training time somewhere to go,
    # which matters for PinSage's inductive use.
    min_books_per_tag: int = 50       # ~1,200 informative tags survive; below 50 they are mostly personal shelves
    min_books_per_author: int = 2     # a one-book author embedding would just duplicate that book's ID embedding
    min_books_per_series: int = 2     # same argument: a series needs 2+ books to link anything
    min_books_per_language: int = 10  # the 14 rarest language codes cover 1-7 books each (42 books in total)
    # Goodreads' own counters (ratings count, star histogram, review count) are strong popularity
    # signals, but they were scraped at the end of the timeline: they summarise ratings that happen
    # after any training cut-off, including the test users' ratings. With split="user_temporal" the
    # effect is small; for a strictly leakage-free evaluation (split="global_temporal") turn this off.
    use_popularity_features: bool = True

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if not (0 < self.val_ratio and 0 < self.test_ratio and self.val_ratio + self.test_ratio < 1):
            raise ValueError("need 0 < val_ratio, 0 < test_ratio and val_ratio + test_ratio < 1")
        if not 1 <= self.positive_threshold <= 5:
            raise ValueError("positive_threshold must be between 1 and 5")


_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


# ==============================================================================================
# 1. Download and load
# ==============================================================================================
def download_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        dest = raw_dir / name
        if dest.exists():
            continue
        log(f"downloading {name}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(GITHUB_RAW + name, tmp)
        tmp.rename(dest)  # rename only after success, so an interrupted download leaves no truncated CSV


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [f for f in RAW_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{missing} not found in {raw_dir} (run with --download)")
    # Narrow integer dtypes: 6M ratings take ~55 MB instead of ~145 MB with pandas' default int64,
    # and sorts/groupbys on small integers are faster.
    ratings = pd.read_csv(raw_dir / "ratings.csv",
                          dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.int8})
    to_read = pd.read_csv(raw_dir / "to_read.csv", dtype={"user_id": np.int32, "book_id": np.int32})
    # ISBNs are read as text: parsed as numbers they would lose even more digits than they already have.
    books = pd.read_csv(raw_dir / "books.csv", dtype={"isbn": str, "isbn13": str})
    book_tags = pd.read_csv(raw_dir / "book_tags.csv",
                            dtype={"goodreads_book_id": np.int64, "tag_id": np.int32, "count": np.int64})
    # keep_default_na=False: genuine tags are called "na" and "none", which pandas would turn into NaN.
    tags = pd.read_csv(raw_dir / "tags.csv", dtype={"tag_id": np.int32, "tag_name": str},
                       keep_default_na=False)
    return dict(ratings=ratings, to_read=to_read, books=books, book_tags=book_tags, tags=tags)


def _sorted_unique(x: np.ndarray) -> np.ndarray:
    """Sorted distinct values. pd.unique (hash-based, O(n)) and then sorting the few distinct values
    is ~50x faster than np.unique on millions of rows."""
    return np.sort(pd.unique(x))


def _lookup(sorted_ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised raw-id -> index lookup. Returns (index, found) for every value."""
    pos = np.minimum(np.searchsorted(sorted_ids, values), len(sorted_ids) - 1)
    return pos, sorted_ids[pos] == values


# ==============================================================================================
# 2. Ratings: cleaning, k-core filtering, re-indexing
# ==============================================================================================
def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    df = ratings.copy()
    # The file is chronological and has no timestamp column, so the row number *is* the timestamp.
    # It must be recorded before anything filters or reorders the rows.
    df["t"] = np.arange(len(df), dtype=np.int32)
    # Only 1-5 stars are valid ratings. In other Goodreads dumps a 0 means "shelved but not rated" and
    # must not be read as a very bad rating. This release has none; the check guards other copies.
    df = df[df["rating"].between(1, 5)]
    # One opinion per (user, book). This release has no duplicates; if a pair ever repeats, keeping
    # the latest row keeps the user's most recent opinion.
    df = df.drop_duplicates(["user_id", "book_id"], keep="last")
    return df


def kcore_filter(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, int]:
    """Iterative k-core on *positive* interactions.

    Removing a user lowers the counts of the books they rated (and vice versa), so the filter is
    repeated until nothing changes. Counting only positives makes the guarantee hold for the graph the
    models actually train on. Users and books that survive keep ALL their ratings, including low ones:
    those are not graph edges, but they record what a user has already read, which evaluation needs
    in order not to recommend a book the user rated 2 stars.
    """
    is_pos = (df["rating"] >= cfg.positive_threshold).to_numpy()
    users, items = df["user_id"].to_numpy(), df["book_id"].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for iteration in range(1, 101):
        p = keep & is_pos
        u_cnt = np.bincount(users[p], minlength=users.max() + 1)
        i_cnt = np.bincount(items[p], minlength=items.max() + 1)
        new_keep = keep & (u_cnt[users] >= cfg.min_user_pos) & (i_cnt[items] >= cfg.min_item_pos)
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return df[keep], iteration


def reindex(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Map raw ids to contiguous 0-based indices.

    Embedding tables (nn.Embedding) and sparse matrices need ids 0..n-1. The raw ids are already
    contiguous, but filtering can leave gaps, so the remap always happens. Indices follow sorted raw
    ids: the mapping is deterministic and item order keeps following book_id, which in this dataset
    is roughly the Goodreads popularity rank (handy when eyeballing results).
    """
    user_ids = _sorted_unique(df["user_id"].to_numpy())
    book_ids = _sorted_unique(df["book_id"].to_numpy())
    out = pd.DataFrame({
        "user": _lookup(user_ids, df["user_id"].to_numpy())[0].astype(np.int32),
        "item": _lookup(book_ids, df["book_id"].to_numpy())[0].astype(np.int32),
        "rating": df["rating"].to_numpy(),
        "t": df["t"].to_numpy(),
    })
    # Canonical order: by user, then time. Each user's history becomes one contiguous chronological
    # block, which makes per-user splitting simple and is exactly what sequential models
    # (SASRec, GRU4Rec, ...) expect if you add them to the comparison later.
    out = out.sort_values(["user", "t"], kind="stable", ignore_index=True)
    return out, user_ids, book_ids


# ==============================================================================================
# 3. Train / validation / test split
# ==============================================================================================
def _rank_within_user(user: np.ndarray, key: np.ndarray, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    """For every row: its 0-based rank inside the user's rows when sorted by key, and the user's row count.
    Pure numpy (one lexsort), so it takes about a second for 6M rows instead of minutes of Python loops."""
    order = np.lexsort((key, user))
    counts = np.bincount(user, minlength=n_users)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rank = np.empty(len(user), dtype=np.int64)
    rank[order] = np.arange(len(user)) - starts[user[order]]
    return rank, counts[user]


def assign_splits(df: pd.DataFrame, cfg: Config, n_users: int) -> np.ndarray:
    """Return a split code (0 train, 1 val, 2 test) for every row of df.

    The split is made on ALL ratings, independent of the positive threshold, so changing what counts as
    a positive later never moves an interaction between train and test.

    "user_temporal" (default): each user's oldest 80% of ratings -> train, next 10% -> val, newest 10%
      -> test. Why it is the default:
      * Every user has history in train, so the transductive LightGCN (one learned vector per user)
        and PinSage can be evaluated on exactly the same users and compared fairly.
      * Holding out each user's *latest* books asks the realistic question "what will they read next?".
        A random split trains on later books to predict earlier ones (e.g. learns from Harry Potter #5
        to "predict" #2), which inflates scores, especially for series.
      * Known limitation: users are cut at different absolute times, so the training graph contains
        some interactions that happened after other users' test interactions (cross-user leakage,
        e.g. how popular the 2017 releases eventually became).
    "global_temporal": a single cut on the shared timeline (oldest 80% of all ratings -> train, ...).
      It is the strictest and most realistic protocol, with no future information anywhere in
      training. The price on this data: 1/3 of test rows (1/2 of val rows) belong to users with no
      positive training history, and 12% of test rows to books with none (mostly new releases).
      LightGCN cannot score those cold-start rows, so mark_eval_targets() excludes them (43% of test
      positives, 53% of val positives) and the evaluation covers warm users and books only, about
      22k users. Use it for a final, leakage-free comparison.
    "random": per-user random 80/10/10 (the protocol of the LightGCN paper). Kept only for comparison
      with published numbers; it leaks the future within each user's own history. On this data even a
      most-popular baseline scores about twice as high under it (test Recall@20 0.083 vs 0.040).
    """
    t = df["t"].to_numpy()
    if cfg.split == "global_temporal":
        t_sorted = np.sort(t)
        cut_val = t_sorted[int(len(t) * (1 - cfg.val_ratio - cfg.test_ratio))]
        cut_test = t_sorted[int(len(t) * (1 - cfg.test_ratio))]
        return np.select([t >= cut_test, t >= cut_val], [2, 1], 0).astype(np.int8)

    user = df["user"].to_numpy()
    key = t if cfg.split == "user_temporal" else np.random.default_rng(cfg.seed).random(len(df))
    rank, n = _rank_within_user(user, key, n_users)

    def n_held_out(ratio: float) -> np.ndarray:
        # Round half up, and hold out at least 1 interaction per user so that every user can be
        # validated and tested (a user with 19 ratings still gets 2 val + 2 test).
        return np.maximum(1, np.floor(n * ratio + 0.5).astype(np.int64))

    n_test, n_val = n_held_out(cfg.test_ratio), n_held_out(cfg.val_ratio)
    if np.any(n - n_test - n_val < 1):
        raise ValueError("some users would have no training interactions; lower val/test ratios")
    return np.select([rank >= n - n_test, rank >= n - n_test - n_val], [2, 1], 0).astype(np.int8)


def mark_eval_targets(df: pd.DataFrame, split: np.ndarray, cfg: Config,
                      n_users: int, n_items: int) -> np.ndarray:
    """A val/test row is ground truth only if it is a positive AND both its user and its book have at
    least one positive in train. LightGCN cannot score cold users or books (their embeddings never get
    a gradient); keeping them would add noise to every model and unfairly penalise LightGCN against
    PinSage, which can embed unseen books from their features. stats.json counts the excluded rows,
    so a separate cold-start evaluation of PinSage remains possible."""
    is_pos = df["rating"].to_numpy() >= cfg.positive_threshold
    user, item = df["user"].to_numpy(), df["item"].to_numpy()
    train_pos = (split == 0) & is_pos
    warm_user = np.bincount(user[train_pos], minlength=n_users) > 0
    warm_item = np.bincount(item[train_pos], minlength=n_items) > 0
    return (split > 0) & is_pos & warm_user[user] & warm_item[item]


# ==============================================================================================
# 4. Book metadata
# ==============================================================================================
# Publication years run from -1750 to 2017 but most books are from 1990-2017, so a single linear
# feature would be dominated by a few ancient texts. Buckets let the model learn non-linear era effects
# (classics, the 2000s YA boom, recent releases). Bins are narrower where books are dense so that every
# bucket holds a few hundred books (from 125 for <1800 to ~1,100 per recent bucket).
YEAR_EDGES = [1800, 1900, 1950, 1970, 1980, 1990, 1995, 2000, 2003, 2006, 2009, 2011, 2013, 2015]
YEAR_LABELS = ([f"<{YEAR_EDGES[0]}"] + [f"{a}-{b - 1}" for a, b in zip(YEAR_EDGES[:-1], YEAR_EDGES[1:])]
               + [f"{YEAR_EDGES[-1]}+"])
# A trailing "(Series Name, #2)" in a title. Only numbered series are parsed: about 120 titles contain
# un-numbered parentheses such as "(A Little Golden Book)" or "Ten Things We Did (and Probably
# Shouldn't Have)", where an imprint cannot be told apart from part of the title.
SERIES_SUFFIX = r"\s*\(([^()]*#[^()]*)\)\s*$"


def _year_bucket(year: pd.Series) -> pd.Series:
    edges = [-np.inf, *YEAR_EDGES, np.inf]
    bucket = pd.cut(year, edges, right=False, labels=YEAR_LABELS).astype(object)
    # Missing years get their own bucket instead of an imputed value: an invented year would put
    # these books into an era they may not belong to.
    return bucket.where(year.notna(), "unknown").astype(str)


def clean_books(books: pd.DataFrame, book_ids: np.ndarray, cfg: Config) -> pd.DataFrame:
    # Keep only books that survived filtering, in item-index order, so that row i == item i.
    b = books.set_index("book_id").loc[book_ids].reset_index()
    b.insert(0, "item", np.arange(len(b), dtype=np.int32))

    # 74 titles and 168 author strings contain repeated spaces; collapsing them prevents the same
    # author or series from turning into two different categories.
    for col in ("title", "original_title", "authors"):
        b[col] = b[col].str.replace(r"\s+", " ", regex=True).str.strip()

    # Series: books of one series are read by the same people in order, which is one of the strongest
    # signals available, but it exists only inside titles. Some titles list several series, e.g.
    # "Mort (Death, #1; Discworld, #4)". The one with the most books in the catalogue is kept
    # (Discworld here), because a series feature is only useful through the books it links together.
    inside = b["title"].str.extract(SERIES_SUFFIX, expand=False)
    cand = inside.str.split(";").explode().str.extract(r"^\s*(?P<name>.*?),?\s*#\s*(?P<num>\d+(?:\.\d+)?)")
    cand["name"] = cand["name"].str.strip()
    cand = cand[cand["name"].str.len() > 0].rename_axis("row").reset_index()
    cand["size"] = cand.groupby("name")["row"].transform("nunique")
    best = cand.sort_values(["row", "size"], ascending=[True, False], kind="stable").drop_duplicates("row")
    best = best.set_index("row").reindex(b.index)               # books without a series -> NaN
    b["series_name"] = best["name"]
    b["series_number"] = pd.to_numeric(best["num"], errors="coerce")  # "#1-3" omnibus -> 1
    # Title without the series suffix: cleaner text for text embeddings, and it stops the series
    # name from being counted twice when title and series are both used.
    b["title_clean"] = b["title"].str.replace(SERIES_SUFFIX, "", regex=True).str.strip()

    # Primary author = first listed name. Later names are mostly illustrators, translators or editors
    # ("J.K. Rowling, Mary GrandPré"); using the full string would split one writer into many categories.
    b["primary_author"] = b["authors"].str.split(",").str[0].str.strip()
    b["n_authors"] = b["authors"].str.count(",") + 1

    # Language: merge the five English codes (8,700 books) into one, and keep missing codes as their
    # own "unknown" value rather than guessing English, since missingness may correlate with the kind
    # of edition. Languages with too few books share "other".
    lang = b["language_code"].fillna("unknown")
    lang = lang.where(~lang.str.startswith("en"), "eng")       # eng, en-US, en-GB, en-CA, en
    counts = lang.value_counts()
    rare = set(counts.index[counts < cfg.min_books_per_language]) - {"unknown"}
    b["language"] = lang.where(~lang.isin(rare), "other")

    b["year_bucket"] = _year_bucket(b["original_publication_year"])

    # isbn was stored as a number upstream, so ISBN-10s lost their leading zeros ("61120081" is
    # really "0061120081"); zero-padding restores them. isbn13 went through a float and cannot be
    # repaired, so it is dropped. Neither is used for modelling; isbn10 is kept only for joining
    # external metadata (descriptions, covers) later.
    b["isbn10"] = b["isbn"].str.zfill(10)
    b = b.drop(columns=["isbn", "isbn13"])

    # A third of the cover URLs are Goodreads' "nophoto" placeholder. Flagged so that image features,
    # if you add them, are not learned from 3,000 copies of the same grey picture.
    b["has_cover"] = ~b["image_url"].str.contains("nophoto", na=True)
    return b


# ==============================================================================================
# 5. Tags -> weighted multi-hot "genre" features
# ==============================================================================================
# Shelves that describe the reader's relationship with the book (status, intent, ownership, format,
# personal verdict, reading challenges, dates) instead of the book's content. They are among the most
# frequent tags, so without this filter the "genre" features would mostly encode popularity
# ("to-read" is the top tag of 88% of books). Patterns match whole hyphen-separated words, so
# "read" removes "to-read" and "read-in-2015" but leaves "ready-player-one" alone. The list was built
# by inspecting the ~450 most widespread tags; it errs on the side of removal, at the cost of a few
# mildly descriptive shelves such as "beach-reads".
_NON_CONTENT_WORDS = [
    r"read", r"reading", r"reads", r"reread", r"re-read", r"tbr", r"unread", r"dnf", r"finish",
    r"finished", r"unfinished", r"abandoned", r"did-not-finish", r"didn-t-finish", r"couldn-t-finish",
    r"could-not-finish", r"gave-up", r"on-hold", r"not-interested", r"want", r"wish-?list", r"to-buy",
    r"to-get", r"maybe", r"someday", r"listened(?:-to)?", r"reviewed", r"reviews?",
    r"own", r"owned", r"mine", r"have", r"borrowed", r"library", r"libraries", r"shelf", r"shelves",
    r"bookshelf", r"kindle", r"nook", r"kobo", r"ebooks?", r"e-books?", r"audio", r"audiobooks?",
    r"audio-books?", r"audible", r"paperback", r"hardcover", r"hardback", r"physical", r"pdf", r"epub",
    r"calibre", r"overdrive", r"scanned", r"signed", r"netgalley", r"arc", r"first-reads", r"giveaways?",
    r"bought", r"purchased", r"favs?", r"faves?", r"favou?rites?", r"loved", r"best-books", r"best-of",
    r"recommended", r"must-read", r"meh", r"\d+-stars?", r"(?:one|two|three|four|five)-stars?",
    r"my", r"book-club", r"bookclub", r"book-group", r"challenge", r"goodreads", r"shelfari", r"amazon",
    r"(?:19|20)\d\d",  # dates ("read-in-2015", "2016-books"); decades like "1920s" stay, they are content
]
NON_CONTENT_RE = re.compile(r"(?:^|-)(?:" + "|".join(_NON_CONTENT_WORDS) + r")(?:-|$)")
NON_CONTENT_EXACT = {"books", "book", "default", "general", "other", "misc", "home", "collection",
                     "english", "in-english", "done", "next", "need", "paper", "work", "and", "cover-love"}
# Abbreviations and true synonyms that the automatic normalisation in _tag_key() cannot catch.
# Hyphenation and singular/plural variants ("graphic-novel" / "graphicnovels") are handled there.
TAG_SYNONYMS = {
    "ya": "young-adult", "ya-fiction": "young-adult", "ya-books": "young-adult", "ya-lit": "young-adult",
    "young-adult-fiction": "young-adult", "teen-fiction": "young-adult",
    "sci-fi": "science-fiction", "scifi": "science-fiction", "sf": "science-fiction",
    "sff": "sci-fi-fantasy", "fantasy-sci-fi": "sci-fi-fantasy", "fantasy-scifi": "sci-fi-fantasy",
    "scifi-fantasy": "sci-fi-fantasy", "sci-fi-and-fantasy": "sci-fi-fantasy",
    "science-fiction-fantasy": "sci-fi-fantasy", "sf-fantasy": "sci-fi-fantasy", "sf-f": "sci-fi-fantasy",
    "non-fic": "non-fiction", "nonfic": "non-fiction", "bio": "biography", "biographies": "biography",
    "kids": "childrens", "children": "childrens", "children-s-books": "childrens",
    "childrens-books": "childrens", "kids-books": "childrens", "children-s-literature": "childrens",
    "childrens-literature": "childrens", "kid-lit": "childrens",
    "chicklit": "chick-lit", "chic-lit": "chick-lit",
    "humour": "humor", "humorous": "humor", "mysteries": "mystery", "dystopia": "dystopian",
    "literary": "literary-fiction", "pnr": "paranormal-romance", "na": "new-adult",
    "vamps": "vampires", "comix": "comics", "picture": "picture-books",
}


def _tag_key(name: pd.Series) -> pd.Series:
    """Spelling-insensitive key: drop punctuation and one trailing plural 's', so "graphic-novel",
    "graphic-novels" and "graphicnovels" become one feature. [\\W_] is Unicode-aware, so Arabic or
    Spanish tags keep their letters instead of collapsing into an empty key."""
    key = name.str.replace(r"[\W_]+", "", regex=True)
    plural = (key.str.len() > 3) & key.str.endswith("s")
    return key.where(~plural, key.str[:-1])


def build_tag_features(book_tags: pd.DataFrame, tags: pd.DataFrame, books: pd.DataFrame,
                       cfg: Config) -> tuple[sp.csr_matrix, pd.DataFrame, np.ndarray]:
    # Name cleaning runs on the 34k-row vocabulary, not the 1M-row table: same result, much faster.
    name = (tags["tag_name"].str.lower().str.strip()
            .str.replace(r"[\s_]+", "-", regex=True).str.replace(r"-{2,}", "-", regex=True).str.strip("-"))
    # Single characters ("g", "-") carry no meaning; two-letter tags stay ("uk", and "ya"/"na" are
    # remapped below to young-adult / new-adult).
    is_content = ~(name.str.contains(NON_CONTENT_RE) | name.isin(NON_CONTENT_EXACT) | (name.str.len() <= 1))
    name = name.replace(TAG_SYNONYMS)
    tag_table = pd.DataFrame({"tag_id": tags["tag_id"], "name": name, "key": _tag_key(name)})[is_content]

    # Negative counts (6 rows, all -1) are scraping errors. The 8 duplicated (book, tag) pairs have
    # near-identical counts (6066 vs 6065), i.e. the same shelf captured twice, so the max is taken;
    # summing would double-count those users.
    bt = book_tags[book_tags["count"] > 0]
    bt = bt.groupby(["goodreads_book_id", "tag_id"], as_index=False)["count"].max()
    gr_to_item = pd.Series(books["item"].to_numpy(), index=books["goodreads_book_id"].to_numpy())
    bt = bt.assign(item=bt["goodreads_book_id"].map(gr_to_item))
    bt = bt.dropna(subset=["item"]).merge(tag_table, on="tag_id", how="inner")  # inner join drops non-content
    bt["item"] = bt["item"].astype(np.int64)

    # Each merged feature is displayed under its most widespread spelling.
    display = bt.groupby(["key", "name"]).size().reset_index(name="n").sort_values(
        ["key", "n", "name"], ascending=[True, False, True]).drop_duplicates("key")
    display = dict(zip(display["key"], display["name"]))
    # Different spellings of one concept were typically chosen by different users, so their counts
    # are summed: the result estimates how many users associate the concept with the book.
    bt = bt.groupby(["item", "key"], as_index=False)["count"].sum()

    n_books = bt["key"].value_counts()
    vocab = n_books[n_books >= cfg.min_books_per_tag]
    vocab = vocab.sort_index().sort_values(ascending=False, kind="stable")  # deterministic order
    bt = bt[bt["key"].isin(vocab.index)].copy()

    # Weight = count / count of the book's strongest kept tag, in (0, 1]. Dividing by the book's own
    # maximum removes popularity (the busiest book has 10,000x more taggers than the quietest, but it
    # should not get 10,000x larger features) while keeping the relative strength of its tags.
    # Linear rather than log scaling, because per-book counts are long-tailed: with log weights a
    # third of each book's ~36 tags would score above 0.5, making the vector near-binary and noisy;
    # with linear weights only the 2-3% strongest do, so the input is dominated by the few tags that
    # actually define the book. (Tag columns are not z-scored; give them their own input projection,
    # as DGL's PinSage example does per feature, so their scale is learned.)
    bt["weight"] = bt["count"] / bt.groupby("item")["count"].transform("max")
    col = pd.Series(np.arange(len(vocab)), index=vocab.index)
    mat = sp.csr_matrix(
        (bt["weight"].to_numpy(np.float32), (bt["item"].to_numpy(), bt["key"].map(col).to_numpy())),
        shape=(len(books), len(vocab)))

    # Top-10 tag names per book as plain text, for inspection and for text embeddings later.
    bt["tag"] = bt["key"].map(display)
    top = (bt.sort_values(["item", "count"], ascending=[True, False]).groupby("item").head(10)
           .groupby("item")["tag"].agg("|".join))
    top_tags = top.reindex(np.arange(len(books)), fill_value="").to_numpy()
    vocab_df = pd.DataFrame({"tag_idx": np.arange(len(vocab)), "tag": [display[k] for k in vocab.index],
                             "n_books": vocab.to_numpy()})
    return mat, vocab_df, top_tags


# ==============================================================================================
# 6. Item feature matrix for PinSage
# ==============================================================================================
def _encode_category(values: pd.Series, min_books: int,
                     order: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Integer-encode a categorical column. Code 0 = rare/unknown; codes 1.. follow `order` if given,
    otherwise decreasing frequency."""
    counts = values.value_counts()  # NaN (e.g. "not in a series") is excluded and therefore gets 0
    keep = counts[counts >= min_books].sort_index().sort_values(ascending=False, kind="stable")
    if order is not None:
        keep = keep.reindex([v for v in order if v in keep.index])
    codes = pd.Series(np.arange(1, len(keep) + 1), index=keep.index)
    encoded = values.map(codes).fillna(0).to_numpy(np.int64)
    vocab = np.asarray(["<rare/unknown>"] + [str(v) for v in keep.index], dtype=str)
    return encoded, vocab


def build_item_features(b: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    num: dict[str, pd.Series] = {}
    if cfg.use_popularity_features:
        # Counts span several orders of magnitude (8 to 4.9M ratings); log1p makes them usable
        # by a linear input layer instead of letting a handful of bestsellers dominate.
        num["log_ratings"] = np.log1p(b["work_ratings_count"])
        num["log_text_reviews"] = np.log1p(b["work_text_reviews_count"])
        # The star histogram as fractions captures polarisation that a mean hides (a 3.9 book can be
        # loved-or-hated or uniformly fine). average_rating is not added separately: it equals the
        # histogram's mean (max difference 0.005), so a linear layer can already recover it.
        total = b["work_ratings_count"].clip(lower=1)
        for k in range(1, 6):
            num[f"frac_{k}_star"] = b[f"ratings_{k}"] / total
    # Number of editions is publisher-side, not user activity: long-lived classics have hundreds.
    num["log_editions"] = np.log1p(b["books_count"])
    # Position in the series: first volumes are entry points, high volumes reach only existing fans.
    num["series_position"] = np.log1p(b["series_number"].fillna(0))
    X = pd.DataFrame(num).to_numpy(np.float64)
    # z-scoring puts all inputs on one scale, which keeps the first linear layer well-conditioned.
    # Statistics are computed over the whole catalogue: they are book metadata, not interaction labels,
    # so using every book does not leak anything about the held-out interactions.
    std = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)

    feats: dict[str, np.ndarray] = {"numeric": X.astype(np.float32),
                                    "numeric_names": np.asarray(list(num), dtype=str)}
    # Categorical features are stored as integer codes for nn.Embedding layers (this is how DGL's
    # PinSage example consumes them) rather than one-hot: 1,400 authors one-hot would be wide and sparse.
    # Year codes follow chronology (1 = oldest), so they can also be used as an ordinal feature.
    for name, column, min_books, order in [
            ("author", b["primary_author"], cfg.min_books_per_author, None),
            ("series", b["series_name"], cfg.min_books_per_series, None),
            ("language", b["language"], 1, None),                       # already grouped in clean_books
            ("year_bucket", b["year_bucket"], 1, YEAR_LABELS + ["unknown"])]:  # buckets are already coarse
        feats[name], feats[f"vocab_{name}"] = _encode_category(column, min_books, order)
    return feats


# ==============================================================================================
# 7. to_read: optional auxiliary edges
# ==============================================================================================
def process_to_read(to_read: pd.DataFrame, user_ids: np.ndarray, book_ids: np.ndarray,
                    df: pd.DataFrame, n_items: int) -> tuple[dict[str, np.ndarray], dict]:
    """Optional "wants to read" edges (913k). Saved separately and NOT used in the default training graph:
    * They are a snapshot taken when the data was collected, i.e. at the end of the timeline, and
      cannot be aligned with the ratings' time order (to_read.csv has its own ordering). Under a
      temporal split they partly reveal the future: someone who loved a test-period book often
      shelves its sequel as to-read.
    * Intent is a weaker, different signal from a rating. If you use them, add them as a separate edge
      type (e.g. ("user", "wants", "book") in a PinSage heterograph), not as extra positives.
    Pairs the user also rated (about 2,200, 0.24%) are removed: they contradict or duplicate the rating,
    and the ones that fall into val/test would hand the model the answer.
    """
    u, u_ok = _lookup(user_ids, to_read["user_id"].to_numpy())
    i, i_ok = _lookup(book_ids, to_read["book_id"].to_numpy())
    t = np.arange(len(to_read), dtype=np.int32)  # its own time order, not comparable with ratings' t
    ok = u_ok & i_ok  # drop users/books removed by the k-core filter
    u, i, t = u[ok].astype(np.int32), i[ok].astype(np.int32), t[ok]
    rated = np.sort(df["user"].to_numpy().astype(np.int64) * n_items + df["item"].to_numpy())
    overlap = _lookup(rated, u.astype(np.int64) * n_items + i)[1]
    edges = {"user": u[~overlap], "item": i[~overlap], "t": t[~overlap]}
    return edges, {"raw": int(len(to_read)), "removed_by_filtering": int((~ok).sum()),
                   "removed_overlap_with_ratings": int(overlap.sum()), "kept": int((~overlap).sum())}


# ==============================================================================================
# 8. Writing outputs and integrity checks
# ==============================================================================================
def write_adjacency_list(path: Path, user: np.ndarray, item: np.ndarray, t: np.ndarray) -> int:
    """One line per user, "u i1 i2 ...", 0-based ids, items in chronological order. Users without items
    are skipped instead of written as a bare id: the reference LightGCN loader calls max() on each
    line's items and crashes on an empty list."""
    order = np.lexsort((t, user))
    user, item = user[order], item[order]
    with open(path, "w") as f:
        if len(user) == 0:
            return 0
        starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
        ends = np.r_[starts[1:], len(user)]
        for s, e in zip(starts, ends):
            f.write(f"{user[s]} {' '.join(map(str, item[s:e].tolist()))}\n")
    return len(starts)


def sanity_checks(df: pd.DataFrame, split: np.ndarray, is_target: np.ndarray, cfg: Config,
                  n_users: int, n_items: int, feats: dict, tag_mat: sp.csr_matrix,
                  books: pd.DataFrame) -> None:
    """Fail loudly instead of training on silently broken data."""
    u, i, t = df["user"].to_numpy(), df["item"].to_numpy(), df["t"].to_numpy()
    key = np.sort(u.astype(np.int64) * n_items + i)
    assert not (np.diff(key) == 0).any(), "duplicate (user, item) pairs"
    assert u.max() + 1 == n_users and np.bincount(u).min() > 0, "user ids are not contiguous"
    assert i.max() + 1 == n_items and np.bincount(i).min() > 0, "item ids are not contiguous"
    if cfg.split == "global_temporal":
        # Every training rating is older than every validation rating, which is older than every test one.
        assert t[split == 0].max() < t[split == 1].min() and t[split == 1].max() < t[split == 2].min()
    else:
        assert np.bincount(u[split == 0], minlength=n_users).min() >= 1, "a user has no training data"
    if cfg.split == "user_temporal":
        # Per user, training ends before validation starts and validation ends before test starts.
        g = pd.DataFrame({"u": u, "s": split, "t": t}).groupby(["u", "s"])["t"].agg(["min", "max"])
        lo, hi = g["min"].unstack(), g["max"].unstack()
        assert (hi[0] < lo[1]).all() and (hi[1] < lo[2]).all(), "temporal order violated inside a user"
    train_pos = (split == 0) & (df["rating"].to_numpy() >= cfg.positive_threshold)
    warm_u = np.bincount(u[train_pos], minlength=n_users) > 0
    warm_i = np.bincount(i[train_pos], minlength=n_items) > 0
    assert warm_u[u[is_target]].all() and warm_i[i[is_target]].all(), "a target is cold-start"
    assert not is_target[split == 0].any(), "train rows cannot be evaluation targets"
    assert feats["numeric"].shape[0] == n_items and np.isfinite(feats["numeric"]).all()
    assert all(feats[c].shape == (n_items,) for c in ("author", "series", "language", "year_bucket"))
    assert tag_mat.shape[0] == n_items and (books["item"].to_numpy() == np.arange(n_items)).all()


def _n_distinct(idx: np.ndarray, n: int) -> int:
    return int(np.count_nonzero(np.bincount(idx, minlength=n)))


def main(cfg: Config) -> None:
    raw_dir, out = Path(cfg.raw_dir), Path(cfg.out_dir)
    if cfg.download:
        download_raw(raw_dir)
    raw = load_raw(raw_dir)
    r0 = raw["ratings"]
    stats: dict = {"config": asdict(cfg), "split_codes": SPLIT_CODES,
                   "raw": {"ratings": len(r0), "users": int(r0["user_id"].nunique()),
                           "books": int(r0["book_id"].nunique()), "to_read": len(raw["to_read"])}}
    log(f"loaded {len(r0):,} ratings")

    # Ratings -> filtered, re-indexed, split interactions
    r = clean_ratings(r0)
    r, iterations = kcore_filter(r, cfg)
    df, user_ids, book_ids = reindex(r)
    n_users, n_items = len(user_ids), len(book_ids)
    stats["after_kcore"] = {"ratings": len(df), "users": n_users, "books": n_items, "iterations": iterations,
                            "removed_users": stats["raw"]["users"] - n_users,
                            "removed_books": stats["raw"]["books"] - n_items}
    log(f"k-core: {n_users:,} users, {n_items:,} books, {len(df):,} ratings ({iterations} iterations)")

    split = assign_splits(df, cfg, n_users)
    is_target = mark_eval_targets(df, split, cfg, n_users, n_items)
    is_pos = df["rating"].to_numpy() >= cfg.positive_threshold
    train_pos = (split == 0) & is_pos
    stats["splits"] = {}
    for name, code in SPLIT_CODES.items():
        m = split == code
        stats["splits"][name] = {
            "rows": int(m.sum()), "positives": int((m & is_pos).sum()),
            "users_with_rows": _n_distinct(df["user"].to_numpy()[m], n_users)}
        if code > 0:
            stats["splits"][name].update({
                "targets": int((m & is_target).sum()),
                "users_with_targets": _n_distinct(df["user"].to_numpy()[m & is_target], n_users),
                "cold_positives_excluded": int((m & is_pos & ~is_target).sum())})
    item_train_deg = np.bincount(df["item"].to_numpy()[train_pos], minlength=n_items)
    stats["train_graph"] = {"edges": int(train_pos.sum()),
                            "density": float(train_pos.sum() / (n_users * n_items)),
                            "books_without_train_edges": int((item_train_deg == 0).sum())}
    log(f"split '{cfg.split}': " + ", ".join(f"{k}={v['rows']:,}" for k, v in stats["splits"].items()))

    # Books, tags, features
    books = clean_books(raw["books"], book_ids, cfg)
    tag_mat, tag_vocab, top_tags = build_tag_features(raw["book_tags"], raw["tags"], books, cfg)
    books["top_tags"] = top_tags
    # Training popularity (positives in train only): what a MostPop baseline or a popularity-bias
    # analysis must use. Goodreads' own counts would leak the held-out period.
    books["train_pos_count"] = item_train_deg
    feats = build_item_features(books, cfg)
    stats["features"] = {"numeric": feats["numeric_names"].tolist(),
                         "categorical_vocab_sizes": {c: int(len(feats[f"vocab_{c}"]))
                                                     for c in ("author", "series", "language", "year_bucket")},
                         "n_tags": int(tag_mat.shape[1]),
                         "avg_tags_per_book": float(np.diff(tag_mat.indptr).mean()),
                         "books_without_tags": int((np.diff(tag_mat.indptr) == 0).sum())}
    log(f"features: {feats['numeric'].shape[1]} numeric, {tag_mat.shape[1]} tags, "
        f"{stats['features']['categorical_vocab_sizes']}")

    to_read, stats["to_read"] = process_to_read(raw["to_read"], user_ids, book_ids, df, n_items)

    sanity_checks(df, split, is_target, cfg, n_users, n_items, feats, tag_mat, books)
    log("sanity checks passed")

    # Write everything
    (out / "lightgcn").mkdir(parents=True, exist_ok=True)
    # Uncompressed .npz: loads in well under a second; compression would save ~60 MB of disk at the
    # cost of several seconds on every load.
    np.savez(out / "interactions.npz", user=df["user"].to_numpy(), item=df["item"].to_numpy(),
             rating=df["rating"].to_numpy(), t=df["t"].to_numpy(), split=split, is_target=is_target)
    u, i, t = df["user"].to_numpy(), df["item"].to_numpy(), df["t"].to_numpy()
    for name, mask in [("train", train_pos), ("val", is_target & (split == 1)),
                       ("test", is_target & (split == 2))]:
        write_adjacency_list(out / "lightgcn" / f"{name}.txt", u[mask], i[mask], t[mask])
    np.savez(out / "item_features.npz", **feats)
    sp.save_npz(out / "item_tags.npz", tag_mat)
    tag_vocab.to_csv(out / "tag_vocab.csv", index=False)
    books.to_csv(out / "books.csv", index=False)
    pd.DataFrame({"user": np.arange(n_users), "user_id": user_ids,
                  "n_ratings": np.bincount(u, minlength=n_users),
                  "n_train_pos": np.bincount(u[train_pos], minlength=n_users)}).to_csv(out / "users.csv", index=False)
    np.savez(out / "to_read.npz", **to_read)
    stats["seconds"] = round(time.time() - _T0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    log(f"done, outputs in {out.resolve()}")


# ==============================================================================================
# Loading helper for training / evaluation scripts
# ==============================================================================================
def load_processed(out_dir: str | Path) -> dict:
    """Load the processed data as [n_users x n_items] CSR matrices that encode the evaluation protocol.

    train              positive train interactions -> graph edges / BPR positives for every model
    val, test          ground truth: warm positives of the held-out periods
    seen_before_val    everything rated in train (any stars): mask these out when ranking for val
    seen_before_test   everything rated in train or val: mask these out when ranking for test
    train_edges        the same positive train edges as arrays (user, item, rating, t), for code that
                       needs time, e.g. PinSage evaluation that queries with a user's latest book
    Masks use all ratings, not just positives: a book the user already rated (even 2 stars) can never
    be a hit, and recommending it again would be a wasted slot in a real system.
    Typical protocol: train on "train", early-stop / tune on "val", report "test" once at the end.
    """
    out = Path(out_dir)
    z = np.load(out / "interactions.npz")
    stats = json.loads((out / "stats.json").read_text())
    n_u, n_i = stats["after_kcore"]["users"], stats["after_kcore"]["books"]
    thr = stats["config"]["positive_threshold"]
    u, i, s, tgt = z["user"], z["item"], z["split"], z["is_target"]

    def csr(mask: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix((np.ones(int(mask.sum()), np.float32), (u[mask], i[mask])), shape=(n_u, n_i))

    train_pos = (s == 0) & (z["rating"] >= thr)
    return {"n_users": n_u, "n_items": n_i,
            "train": csr(train_pos),
            "train_edges": {k: z[k][train_pos] for k in ("user", "item", "rating", "t")},
            "val": csr((s == 1) & tgt), "test": csr((s == 2) & tgt),
            "seen_before_val": csr(s == 0), "seen_before_test": csr(s <= 1)}


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Preprocess goodbooks-10k for LightGCN / PinSage. "
                                            "Every option is documented on the Config class.")
    for f in fields(Config):  # one CLI flag per Config field, so defaults live in exactly one place
        flag, help_ = "--" + f.name.replace("_", "-"), "default: %(default)s"
        if isinstance(f.default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default, help=help_)
        elif f.name == "split":
            p.add_argument(flag, choices=SPLITS, default=f.default, help=help_)
        else:
            p.add_argument(flag, type=type(f.default), default=f.default, help=help_)
    return Config(**vars(p.parse_args()))


if __name__ == "__main__":
    main(parse_args())
