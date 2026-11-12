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

## LightGCN top-K recommendation (`light_gcn` branch)

Implicit-feedback formulation: ratings ≥ 4 are positive interactions (`--min_rating`). LightGCN (He et al., 2020) is trained with BPR loss and uniform negative sampling, and evaluated by full ranking over all 10k books.

**Data**: the shared preprocessing in `process.py` (run once: `python process.py --raw-dir data --out-dir data/processed`), loaded via `process.load_processed`. It applies a 10-core filter, then splits each user's ratings (all stars) chronologically 80/10/10. Graph edges are the >=4 star train ratings, and val/test ground truth are warm >=4 star held-out ratings. Result: 53,289 users, 9,996 books, 3.30M train edges, 407K val / 414K test targets.

**Evaluation** (`lightgcn/metrics.py`): Recall/Precision/NDCG/HitRate @ {10, 20, 50}. Val masks every book rated in train; test masks every book rated in train or val (any stars). Early stopping on val Recall@20; the best checkpoint is scored on test.

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install numpy pandas scipy
.venv/bin/python process.py --raw-dir data --out-dir data/processed   # shared split
.venv/bin/python -m lightgcn.data                   # print split stats
.venv/bin/python -m lightgcn.train --popularity     # most-popular baseline
.venv/bin/python -m lightgcn.train --layers 0       # MF-BPR baseline (no propagation)
# GPU (FarmShare `gpu` partition, L40S); needs a CUDA torch venv in .venv-gpu (cu126 wheel)
sbatch slurm/train_gpu.sbatch --layers 3 --dim 64
```

CPU training is too slow to be practical (more than 15 minutes per epoch), so use the GPU script. Results go to `runs/<name>/results.json`.

| Model | val R@20 | val N@20 | test R@20 | test N@20 |
|---|---|---|---|---|
| Popularity | 0.0483 | 0.0383 | 0.0399 | 0.0313 |
