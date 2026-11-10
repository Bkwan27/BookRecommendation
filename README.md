# BookRecommendation

Graph ML book recommender (Sacha Gregoire, Bryan Kwan, Derek Shi). We model users and books as a bipartite graph and predict held-out user→book ratings (1–5) with graph models (GC-MC, heterogeneous GraphSAGE, GAT, graph transformers, MoE routing).

## Dataset: goodbooks-10k

Source: https://github.com/zygmuntz/goodbooks-10k (Zygmunt Zając), licensed [CC BY-SA 4.0](data/LICENSE).

| File | Contents |
|---|---|
| `data/ratings.csv` | `user_id,book_id,rating`: 5,976,479 ratings, 53,424 users, 10,000 books, **sorted by time** (no timestamps) |
| `data/to_read.csv` | `user_id,book_id`: ~1M "to read" marks, sorted by time |
| `data/books.csv` | book metadata (title, authors, year, avg rating, goodreads ids, ...) |
| `data/book_tags.csv` | `goodreads_book_id,tag_id,count`: user shelves/genres |
| `data/tags.csv` | `tag_id,tag_name` |
| `data/samples/` | small snippets of each file |

Notes:
- `book_id` in `ratings.csv` / `to_read.csv` is 1–10000; join to `books.csv` on `book_id`. `book_tags.csv` uses `goodreads_book_id`, so map through `books.csv`.
- Every user has ≥19 ratings (median 111, max 200); books have 8–22,806 ratings.
- Rating distribution: 1★ 2.1%, 2★ 6.0%, 3★ 22.9%, 4★ 35.8%, 5★ 33.2%.
- The raw goodreads XML (`books_xml/`) from the original repo is not included.
- `ratings.csv` rows are interleaved across users in global time order, so the row index is used as a clock.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests          # fast synthetic tests
```

## Graph-transformer expert (`bookrec/`)

One of the MoE experts: candidate-aware attention over each user's ego-graph.

| Module | What it does |
|---|---|
| `bookrec/data.py` | loading, frozen splits (`splits/`, regenerated deterministically), per-user histories, leak-filtered `to_read` |
| `bookrec/features.py` | book features: tf-idf of content tags, author, language, year. Rating-derived columns and rating/sentiment shelves (`5-stars`, `favorites`, ...) are excluded because they leak labels |
| `bookrec/sampling.py` | per-step context/target masking, random and PinSAGE-style hard negatives (co-rated neighbours ranked 10–100), curriculum |
| `bookrec/models/graph_transformer.py` | the model |
| `bookrec/losses.py` | `L = L_BPR + w_graded·L_graded + λ·L_rating` |
| `bookrec/evaluate.py` | full-catalogue Recall@K / NDCG@K, RMSE / accuracy, popularity and constant baselines |
| `bookrec/train_graph_transformer.py` | training script |

**Model.** User u is the `[USER]` token plus one token per book in their library. Each token is `book embedding + rating (edge-type) embedding + recency embedding`. Book embeddings combine an ID embedding (randomly dropped during training, which simulates rarely-rated books) with the content features. They also get one round of message passing over the book–book co-rating graph (2-hop structure, train edges only). A transformer encoder runs over the ego-graph. Each candidate book then cross-attends over the encoded library, and an MLP outputs a **ranking score** (for BPR) and **P(1..5 stars)** (auxiliary). A user is defined purely by their library, so new users need no retraining.

**Training step.** For each user, ~20% of their *train* ratings become hidden targets and the rest are the context the model sees. This mirrors serving, where the library is known and candidates are unseen.

**Loss.**
- **BPR (per-user mean).**
  - Positives: targets rated 4–5★, plus `to_read` books.
  - Negatives: random unrated books, hard negatives (added at `hard_step` per epoch, up to `max_hard_neg`), and targets rated 1–2★ (read and disliked).
- **Graded pairwise:** within-user target pairs with a rating gap ≥ 2 (e.g. 5★ > 3★).
- **Rating:** 5-class cross-entropy (or `--rating-loss mse` on the expected rating) on targets.

**Evaluation.** The temporal split (default) holds out each user's last 10% of ratings for test and the 10% before that for val. Every book is ranked, books the user rated in train are excluded, and positives are held-out 4–5★ ratings. The checkpoint with the best val Recall@20 is the one tested.

```bash
python -m bookrec.train_graph_transformer                                     # full run
python -m bookrec.train_graph_transformer --max-users 4000 --epochs 2 --eval-users 500 --test-users 500  # smoke test
python -m bookrec.train_graph_transformer --split random                      # random 80/10/10
```
Results go to `runs/<name>/results.json` (args, per-epoch history, test metrics, baselines).

### Running on Colab / a GPU cluster

```bash
git clone -b derek/graph-transformer https://github.com/Bkwan27/BookRecommendation.git
cd BookRecommendation && pip install -r requirements.txt
python -m bookrec.train_graph_transformer --device cuda --out runs/gt_full
```
- `--device auto` (the default) picks CUDA, then MPS, then CPU.
- The first run writes `splits/temporal_seed0.npz` and the co-rating neighbour cache (~10s). Copy `splits/` between machines so everyone uses identical splits.
- Batch sampling runs on the CPU in the main process. If the GPU is underused, raise `--batch-size` (e.g. 1024, and scale `--lr` up).
- `--eval-users 0 --test-users 0` evaluates every user (the default tests on all users and validates on 2,000).
