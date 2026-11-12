"""Train the graph-transformer expert with BPR + hard negatives + auxiliary rating loss.

Prerequisite (once per machine, ~15s): python -m common.process

    python -m graphtransformer.train_graph_transformer                                  # full run
    python -m graphtransformer.train_graph_transformer --max-users 4000 --epochs 2 \
        --eval-users 500 --test-users 500                                               # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from common.process import DEFAULT_PROCESSED
from .data import load_dataset
from .evaluate import baselines, evaluate
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
    a("--processed-dir", default=str(DEFAULT_PROCESSED), help="output of `python -m common.process`")
    a("--seed", type=int, default=0)
    a("--epochs", type=int, default=25)
    a("--batch-size", type=int, default=256)
    a("--lr", type=float, default=1e-3)
    a("--weight-decay", type=float, default=1e-4)
    a("--dim", type=int, default=64)
    a("--heads", type=int, default=4)
    a("--layers", type=int, default=2)
    a("--dropout", type=float, default=0.1)
    a("--id-dropout", type=float, default=0.1, help="prob. of dropping a book's ID embedding")
    a("--graph-nbrs", type=int, default=20, help="co-rated neighbours for book smoothing (0 = off)")
    a("--max-context", type=int, default=160, help="160 = every user's full train history")
    a("--max-targets", type=int, default=32)
    a("--target-frac", type=float, default=0.2)
    a("--rand-neg", type=int, default=16)
    a("--max-hard-neg", type=int, default=8)
    a("--hard-step", type=int, default=1, help="hard negatives added per epoch (PinSAGE curriculum)")
    a("--hard-rank", type=int, nargs=2, default=[10, 100])
    a("--to-read", action="store_true",
      help="ABLATION ONLY: use to_read.csv as extra positives (off in the shared protocol: it is an "
           "end-of-timeline snapshot and leaks the future)")
    a("--to-read-per-user", type=int, default=4)
    # loss
    a("--pos-threshold", type=int, default=4, help="must equal the threshold common.process used")
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
    ds = load_dataset(args.processed_dir, use_to_read=args.to_read)
    if args.pos_threshold != ds.positive_threshold:
        raise SystemExit(f"--pos-threshold {args.pos_threshold} != processed threshold {ds.positive_threshold}")
    feats = load_book_features(args.processed_dir)
    need_k = max(args.hard_rank[1], args.graph_nbrs)
    nbrs = corated_neighbors(ds.train, ds.n_items, k=need_k,
                             cache=Path(args.processed_dir) / "cache" / f"corated_{ds.fingerprint}_k{need_k}.npz")
    print(f"data ready in {time.time() - t0:.0f}s (fingerprint {ds.fingerprint}): "
          f"users={ds.n_users:,} books={ds.n_items:,} train={len(ds.train.book):,} "
          f"val={len(ds.val.book):,} test={len(ds.test.book):,}")

    model = GraphTransformerRecommender(
        feats, ds.n_items, d=args.dim, heads=args.heads, layers=args.layers,
        dropout=args.dropout, id_dropout=args.id_dropout,
        nbrs=nbrs if args.graph_nbrs > 0 else None, n_graph_nbrs=max(args.graph_nbrs, 1),
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_cfg = LossConfig(args.pos_threshold, args.dislike_threshold, args.w_dislike,
                          args.w_graded, args.graded_min_gap, args.rating_weight, args.rating_loss)
    sampler = TrainSampler(ds, nbrs, max_context=args.max_context, max_targets=args.max_targets,
                           target_frac=args.target_frac, n_rand_neg=args.rand_neg,
                           hard_rank=tuple(args.hard_rank),
                           n_to_read=args.to_read_per_user if args.to_read else 0, seed=args.seed)

    train_users = sampler.trainable_users()
    if args.max_users:
        train_users = rng.choice(train_users, min(args.max_users, len(train_users)), replace=False)
    all_users = np.arange(ds.n_users)
    val_users = None if not args.eval_users else rng.choice(all_users, args.eval_users, replace=False)
    test_users = None if not args.test_users else rng.choice(all_users, args.test_users, replace=False)
    if args.max_users:   # debug runs: evaluate users we trained on
        val_users = train_users[:args.eval_users or None]
        test_users = train_users[:args.test_users or None]

    val_base = baselines(ds, "val", val_users)
    print("val baselines:", json.dumps(val_base))

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
        val = evaluate(model, ds, "val", val_users, device=device, max_context=args.max_context)
        history.append({"epoch": epoch, **{f"train_{k}": v / steps_per_epoch for k, v in running.items()},
                        **{f"val_{k}": v for k, v in val.items()}})
        print("val:", json.dumps(val), flush=True)
        if val["recall@20"] > best:      # model selection on val Recall@20 (team invariant)
            best = val["recall@20"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "fingerprint": ds.fingerprint}, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test = evaluate(model, ds, "test", test_users, device=device, max_context=args.max_context)
    test_base = baselines(ds, "test", test_users)
    print("test:", json.dumps(test))
    print("test baselines:", json.dumps(test_base))
    (out / "results.json").write_text(json.dumps(
        {"expert": "graph_transformer", "fingerprint": ds.fingerprint, "args": vars(args),
         "loss_config": asdict(loss_cfg), "history": history, "val_baselines": val_base,
         "test": test, "test_baselines": test_base}, indent=2))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
