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
