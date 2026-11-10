# BookRecommendation

Graph ML book recommender (Sacha Gregoire, Bryan Kwan, Derek Shi). Users and books form a bipartite graph. Three graph experts (LightGCN, PinSAGE, graph transformer) each rank books for a user, and a Mixture-of-Experts (MoE) router combines them. Recommendations are judged on held-out future ratings (Recall/NDCG@K), and star predictions on RMSE/accuracy.

## Dataset: goodbooks-10k

Source: https://github.com/zygmuntz/goodbooks-10k (Zygmunt Zając), licensed [CC BY-SA 4.0](data/LICENSE).

| File | Contents |
|---|---|
| `data/ratings.csv` | `user_id,book_id,rating`: 5,976,479 ratings, 53,424 users, 10,000 books, **sorted by time** (no timestamps) |
| `data/to_read.csv` | `user_id,book_id`: ~1M "to read" marks (a snapshot from the end of the timeline) |
| `data/books.csv` | book metadata (title, authors, year, avg rating, goodreads ids, ...) |
| `data/book_tags.csv` | `goodreads_book_id,tag_id,count`: user shelves/genres |
| `data/tags.csv` | `tag_id,tag_name` |
| `data/samples/` | small snippets of each file |

- `ratings.csv` rows are interleaved across users in global time order, so the row number is used as the clock.
- Every user has ≥19 ratings (median 111, max 200); books have 8–22,806 ratings. Stars: 1★ 2.1%, 2★ 6.0%, 3★ 22.9%, 4★ 35.8%, 5★ 33.2%.
- `book_tags.csv` uses `goodreads_book_id`; join through `books.csv`. The raw Goodreads XML (`books_xml/`) is not included.

## Quick start

```bash
pip install -r requirements.txt
python -m common.process            # raw CSVs -> data/processed/ (~15s, deterministic, gitignored)
python -m pytest -q tests           # shared-pipeline invariants + graph-transformer tests (synthetic data)
python -m graphtransformer.train_graph_transformer --device cuda --out runs/gt_full
```

`python -m common.process` prints a **fingerprint** (currently `54afd23cf75a` with the default config). If your fingerprint differs, you're not on the shared split.

## Repository layout

| Path | Owner | What |
|---|---|---|
| `common/process.py` | shared (Sacha) | the **only** place the data is cleaned, split, masked and featurised |
| `common/metrics.py` | shared (adapted from Bryan's `lightgcn/metrics.py`) | full-ranking Recall/Precision/NDCG/Hit@K, rating metrics, baselines |
| `common/expert.py` | shared | the expert interface and per-user score standardisation for the router |
| `graphtransformer/` | Derek | graph-transformer expert |
| `lightgcn/` | Bryan (`light_gcn` branch) | LightGCN expert; to be ported onto `common/` |
| `pinsage.py` | Sacha | PinSAGE expert (placeholder) |
| `moe/` | team | router (to do) |

## Shared data pipeline (`common/`)

`python -m common.process` (defaults: `--raw-dir data --out-dir data/processed --split user_temporal`):

1. **Cleans ratings.** It keeps users and books with ≥10 positive (≥4★) ratings (10-core), which removes 135 users and 4 books. It then renumbers ids to `0..n-1`; `users.csv` / `books.csv` map them back to the raw ids. Result: 53,289 users × 9,996 books × 5.96M ratings.
2. **Splits each user's ratings in time order, using all star values:** oldest 80% → train, next 10% → val, newest 10% → test. That gives 4.77M / 599k / 599k.
3. **Ground truth** (`val`, `test`): held-out ratings of ≥4★ whose user and book both have a train positive.
4. **Masks:** `seen_before_val` = everything rated in train (any stars); `seen_before_test` = everything rated in train **or val**.
5. **Book features:**
   - 1,213 content tags (reader-status, ownership, date and rating shelves removed; synonyms merged);
   - categorical codes: author, series, language, year bucket;
   - numeric: `log_editions`, `series_position`.

   Goodreads' popularity and rating counters are **off by default** because they leak the held-out period.

Every model loads it the same way:

```python
from common.process import load_processed, load_item_features
data  = load_processed()           # dict of CSRs / arrays, see the docstring
feats = load_item_features()       # PinSAGE + graph transformer

data["train"]         # CSR [U x I] positive train edges -> LightGCN / PinSAGE graph, BPR positives
data["train_all"]     # all train ratings with stars (user, item, rating, t) -> graph transformer
data["val"], data["test"]                       # ground-truth CSRs
data["seen_before_val"], data["seen_before_test"]  # masks
data["fingerprint"]   # write this into results.json
```

- **LightGCN:** build the normalised adjacency from `data["train"]` only. Users are nodes `0..U-1` and books `U..U+I-1`.
- **PinSAGE:** the bipartite graph comes from `data["train"]`. Book node features come from `feats` (`tags` CSR, `numeric`, and categorical codes fed to `nn.Embedding`). Random walks run on the train graph only.
- **Graph transformer:** `graphtransformer.data.load_dataset()` wraps `data["train_all"]` into per-user histories.
- **Evaluation (all experts):**
  ```python
  from common.metrics import ranking_metrics
  val  = ranking_metrics(score_fn, data["val"],  data["seen_before_val"])
  test = ranking_metrics(score_fn, data["test"], data["seen_before_test"])
  ```
  `score_fn(user_ids) -> [B, n_items]` scores every book.

## Invariants (what we agreed on; every expert and the MoE must follow these)

**Data**
1. **One split.** Everyone loads `common.process` output with the default config, and every `results.json` records `fingerprint`. Results with different fingerprints are not compared or combined.
2. **One id space.** Use the processed indices only. Never mix in raw `user_id − 1` / `book_id − 1`, or anything from another branch's own split code.
3. **Inputs are train-only.** Graph edges, message passing, random walks, co-rating neighbours, sampled negatives, popularity counts and user histories are all built from train rows. Val/test rows are never inputs, including at evaluation time: **the input history for both val and test evaluation is the train period only.**
4. **Leak-free features.** Use the `common.process` feature set as-is: no `average_rating`, rating counts, star histograms or rating/sentiment shelves. Turning `--use-popularity-features` on is a labelled ablation, never a headline number.
5. **No `to_read` as positives.** It's an end-of-timeline snapshot and leaks the future. It may only appear in a labelled ablation, or as a separate edge type with its own result.

**Objective**

6. **Positive = ≥4★** (the processed `positive_threshold`).
7. **Same loss for every expert:** `L = L_BPR + 0.5·L_graded + 0.5·L_rating`.
   - BPR: per-user mean. Negatives are random unrated books, PinSAGE-style hard negatives (co-rated neighbours ranked 10–100, one more per epoch up to 8), and the user's own 1–2★ books.
   - Graded: within-user pairs with a rating gap ≥ 2.
   - Rating: 5-class cross-entropy.
   - Experts without a native rating head (LightGCN, PinSAGE) add a small MLP on `[z_u, z_v, z_u⊙z_v]` that outputs 5 star logits.
   - A plain-BPR version is kept as a *baseline*, not as the MoE expert.
8. **Hide targets during training.** An expert must not see the edges it's being trained to predict. The graph transformer hides each step's targets from the context. For LightGCN/PinSAGE, either drop the batch's target edges from propagation, or state in results that full-graph propagation was used. This matters for interpreting results, and it's why the router is trained on val.

**Evaluation**

9. **Rank the full catalogue and mask with the shared masks.** Val masks books rated in train; **test masks books rated in train and val** (any stars). Ground truth is `data["val"]` / `data["test"]`. Metrics come from `common.metrics.ranking_metrics` only.
10. **Report** Recall/NDCG/Precision/Hit@{10, 20, 50} plus RMSE/accuracy on all held-out ratings, next to the MostPop and constant-rating baselines from `common.metrics`.
11. **Model selection uses val Recall@20.** Test is evaluated **once**, on the selected checkpoint. Seeds are recorded.
12. **Same users.** Headline numbers use all users with targets (`--test-users 0`). Per-epoch val may use a fixed random sample of users.

## MoE pipeline

```
               common.process (one split, masks, features, fingerprint)
                                   │
        ┌──────────────────────────┼──────────────────────────┐
     LightGCN                   PinSAGE              Graph transformer
   (IDs + graph)       (content + sampled graph)   (rating-aware attention)
        └──────── same loss, trained on TRAIN, frozen ────────┘
                                   │ score_all(users) / rating_logits
                        per-user standardised scores
                                   │
              router g(u, v) = softmax(MLP(features)) over 3 experts
                     trained on VAL users (80/20 by user)
                                   │
                          report once on TEST
```

1. **Train each expert** on train with the shared loss (invariants 6–8). Select its checkpoint on val Recall@20 and freeze it. Each implements `common.expert.Expert`: `score_all(users) -> [B, I]`, plus `rating_logits` if it has a rating head.
2. **Build the router's training data from val, not train.** Experts have fit the train edges (LightGCN propagates over all of them), so a router trained on train would learn to always trust the most overfit expert.
   - For each val user, take a fixed candidate set: val targets + ~100 negatives (random, hard, and the user's 1–2★ books).
   - Precompute every expert's score and star probabilities for it: ~53k users × ~120 candidates × 3 experts ≈ 80 MB, so router epochs take seconds.
   - Split val **users** 80/20 (fixed seed): 80% to fit the router, 20% to tune it.
3. **Standardise scores per user** (`common.expert.standardize_per_user`) before mixing. BPR scores only matter within a user, so raw scales differ between experts.
4. **Router** `g(u, v) = softmax(MLP(x))` over the experts. Its inputs are computed from train only:
   - log(user's train ratings), log(book's train positives);
   - user rating mean/std and genre (tag) entropy, book year bucket;
   - the standardised expert scores, their disagreement, and an "expert available" flag (LightGCN can't score a brand-new user).
   - Mixed score = `Σ g_i · z_i`; mixed stars = `Σ g_i · p_i(stars)`.
   - Trained with the same loss (invariant 7) plus a small load-balancing penalty (e.g. `0.01·3·Σ_i mean(g_i)²`) and expert dropout, so it doesn't collapse onto one expert.
5. **Evaluate on test once**, with the shared metrics and masks (input history = train only). Report:
   - each expert alone;
   - a uniform average of the standardised scores (the MoE must beat this);
   - a router that only sees rating counts;
   - the MoE;
   - an oracle "best expert per user" upper bound.
6. **Analysis:** metrics broken down by user and book rating-count buckets, and the router's weights plotted against those counts. The expected pattern is LightGCN on popular books, PinSAGE on rarely rated books and light users, and attention on heavy, varied readers.

Optional: joint fine-tuning of experts + router at a low learning rate. Keep it only if it beats step 5 on the held-out 20% of val users.

## Experts

### Graph transformer (`graphtransformer/`)

Candidate-aware attention over each user's ego-graph.
- **Tokens:** user u is a `[USER]` token plus one token per train-rated book. Each token is `book embedding + star (edge-type) embedding + recency embedding`.
- **Book embeddings:** an ID embedding (randomly dropped in training to simulate rarely rated books) plus the shared content features (tags, author, series, language, year, numeric), with one round of smoothing over the book–book co-rating graph (train edges only).
- **Scoring:** a transformer encoder runs over the ego-graph. Each candidate book cross-attends over the encoded library, and an MLP outputs a ranking score (BPR) and P(1..5 stars).
- **New users:** a user is defined by their library, so they need no retraining.

| Module | What |
|---|---|
| `data.py` | wraps `load_processed()` into per-user histories (all stars), targets, masks |
| `features.py` | wraps `load_item_features()` |
| `sampling.py` | per-step context/target masking, random + hard negatives, curriculum |
| `models/graph_transformer.py` | the model (~1.0M params at dim 64, 2 layers) |
| `losses.py` | the shared objective (invariant 7) |
| `evaluate.py` | shared-protocol evaluation (invariants 3, 9) |
| `train_graph_transformer.py` | training script |

```bash
python -m graphtransformer.train_graph_transformer --device cuda --out runs/gt_d64_L2
python -m graphtransformer.train_graph_transformer --dim 128 --layers 2 --out runs/gt_d128_L2
python -m graphtransformer.train_graph_transformer --max-users 4000 --epochs 2 --eval-users 500 --test-users 500   # smoke test
```
- **Epoch size:** an epoch visits each user once and hides ~20% of their train ratings, so ~956k targets per epoch. 25 epochs is the default.
- **GPU use:** batch sampling runs on the CPU. If the GPU is underused, raise `--batch-size` and the learning rate.
- **Output:** `runs/<name>/results.json` holds the args, fingerprint, per-epoch val history, test metrics and baselines.

### LightGCN (`light_gcn` branch, Bryan)

Implemented with BPR and full-ranking evaluation. To join the MoE:
- load `common.process.load_processed()` instead of `lightgcn/data.py`;
- evaluate with `common.metrics.ranking_metrics(..., data["seen_before_test"])`;
- add the rating head and the shared loss (invariant 7), keeping plain BPR as a baseline;
- record `fingerprint`.

### PinSAGE (`pinsage.py`, Sacha)

To do. Inputs: the train positive graph `data["train"]`, book features from `load_item_features()`, random walks on train only, the shared loss and metrics.

## Running on Colab / a GPU cluster

```bash
git clone https://github.com/Bkwan27/BookRecommendation.git && cd BookRecommendation
pip install -r requirements.txt
python -m common.process                  # check the printed fingerprint
python -m graphtransformer.train_graph_transformer --device cuda --out runs/gt_full
```
`--device auto` picks CUDA, then MPS, then CPU. The co-rating neighbour cache is written to `data/processed/cache/`.
