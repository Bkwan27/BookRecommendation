"""Train the graph-transformer expert with BPR + hard negatives + auxiliary rating loss.

    python -m bookrec.train_graph_transformer                       # full run
    python -m bookrec.train_graph_transformer --max-users 3000 --epochs 2 --eval-users 500  # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import SPLIT_DIR, build_dataset, load_or_make_split, load_ratings, load_to_read
from .evaluate import constant_baselines, evaluate, popularity_recall
from .features import load_book_features
from .losses import LossConfig, objective
from .models.graph_transformer import GraphTransformerRecommender
from .sampling import TrainSampler, corated_neighbors


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--split", default="temporal", choices=["temporal", "random"])
    a("--seed", type=int, default=0)
    a("--epochs", type=int, default=10)
    a("--batch-size", type=int, default=256)
    a("--lr", type=float, default=1e-3)
    a("--weight-decay", type=float, default=1e-4)
    a("--dim", type=int, default=64)
    a("--heads", type=int, default=4)
    a("--layers", type=int, default=2)
    a("--dropout", type=float, default=0.1)
    a("--id-dropout", type=float, default=0.1, help="prob. of dropping a book's ID embedding")
    a("--graph-nbrs", type=int, default=20, help="co-rated neighbours for book smoothing (0 = off)")
    a("--max-context", type=int, default=128)
    a("--max-targets", type=int, default=32)
    a("--target-frac", type=float, default=0.2)
    a("--rand-neg", type=int, default=16)
    a("--max-hard-neg", type=int, default=8)
    a("--hard-step", type=int, default=2, help="hard negatives added per epoch (PinSAGE curriculum)")
    a("--hard-rank", type=int, nargs=2, default=[10, 100])
    a("--no-to-read", action="store_true", help="don't use to_read.csv as positives")
    a("--to-read-per-user", type=int, default=4)
    # loss
    a("--pos-threshold", type=int, default=4)
    a("--dislike-threshold", type=int, default=2)
    a("--w-dislike", type=float, default=1.0)
    a("--w-graded", type=float, default=0.5)
    a("--graded-min-gap", type=int, default=2)
    a("--rating-weight", type=float, default=0.5, help="lambda for the auxiliary rating loss")
    a("--rating-loss", default="ce", choices=["ce", "mse"])
    # run control
    a("--max-users", type=int, default=0, help="train on a random subset of users (debug)")
    a("--eval-users", type=int, default=2000, help="val users per epoch (0 = all)")
    a("--test-users", type=int, default=0, help="test users at the end (0 = all)")
    a("--device", default="auto")
    a("--out", default="runs/graph_transformer")
    a("--log-every", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    t0 = time.time()
    r = load_ratings()
    split = load_or_make_split(r, mode=args.split, seed=args.seed)
    ds = build_dataset(r, split, None if args.no_to_read else load_to_read())
    feats = load_book_features()
    need_k = max(args.hard_rank[1], args.graph_nbrs)
    nbrs = corated_neighbors(ds.train, ds.n_books, k=need_k,
                             cache=SPLIT_DIR / f"{args.split}_seed{args.seed}_nbrs{need_k}.npz")
    print(f"data ready in {time.time() - t0:.0f}s: train={len(ds.train.book):,} "
          f"val={len(ds.val.book):,} test={len(ds.test.book):,}")

    model = GraphTransformerRecommender(
        feats, ds.n_books, d=args.dim, heads=args.heads, layers=args.layers,
        dropout=args.dropout, id_dropout=args.id_dropout,
        nbrs=nbrs if args.graph_nbrs > 0 else None, n_graph_nbrs=max(args.graph_nbrs, 1),
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_cfg = LossConfig(args.pos_threshold, args.dislike_threshold, args.w_dislike,
                          args.w_graded, args.graded_min_gap, args.rating_weight, args.rating_loss)
    sampler = TrainSampler(ds, nbrs, max_context=args.max_context, max_targets=args.max_targets,
                           target_frac=args.target_frac, n_rand_neg=args.rand_neg,
                           hard_rank=tuple(args.hard_rank), n_to_read=args.to_read_per_user,
                           seed=args.seed)

    train_users = sampler.trainable_users()
    if args.max_users:
        train_users = rng.choice(train_users, min(args.max_users, len(train_users)), replace=False)
    all_users = np.arange(ds.n_users)
    val_users = all_users if not args.eval_users else rng.choice(all_users, args.eval_users, replace=False)
    if args.max_users:   # debug runs: evaluate the users we trained on
        val_users = train_users[:args.eval_users or None]

    baselines = {**constant_baselines(ds, ds.val), **popularity_recall(ds, ds.val, val_users)}
    print("val baselines:", json.dumps(baselines))

    steps_per_epoch = int(np.ceil(len(train_users) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, pct_start=0.05,
                                                total_steps=steps_per_epoch * args.epochs)
    history, best, best_path = [], -1.0, out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        sampler.n_hard = min(args.max_hard_neg, (epoch - 1) * args.hard_step)
        perm = rng.permutation(train_users)
        model.train()
        running, t_ep = {}, time.time()
        for step in range(steps_per_epoch):
            batch = sampler.sample(perm[step * args.batch_size:(step + 1) * args.batch_size]).to(device)
            score, logits = model(batch)
            loss, parts = objective(score, logits, batch, loss_cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            if (step + 1) % args.log_every == 0 or step + 1 == steps_per_epoch:
                n = step + 1
                print(f"ep {epoch} step {n}/{steps_per_epoch} hard={sampler.n_hard} "
                      + " ".join(f"{k}={v / n:.4f}" for k, v in running.items())
                      + f" ({time.time() - t_ep:.0f}s)", flush=True)
        val = evaluate(model, ds, ds.val, val_users, device=device, max_context=args.max_context,
                       pos_threshold=args.pos_threshold)
        rec = {"epoch": epoch, **{f"train_{k}": v / steps_per_epoch for k, v in running.items()},
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        print("val:", json.dumps(val), flush=True)
        if val["recall@20"] > best:
            best = val["recall@20"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_users = all_users if not args.test_users else rng.choice(all_users, args.test_users, replace=False)
    if args.max_users:
        test_users = train_users[:args.test_users or None]
    test = evaluate(model, ds, ds.test, test_users, device=device, max_context=args.max_context,
                    pos_threshold=args.pos_threshold)
    test_base = {**constant_baselines(ds, ds.test), **popularity_recall(ds, ds.test, test_users)}
    print("test:", json.dumps(test))
    print("test baselines:", json.dumps(test_base))
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "loss_config": asdict(loss_cfg), "history": history,
         "val_baselines": baselines, "test": test, "test_baselines": test_base}, indent=2))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
