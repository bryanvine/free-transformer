#!/usr/bin/env python3
"""Tokenize a FineWeb-Edu sample into data/fineweb_edu/{train,val}.bin.

Defaults to the 10BT sample (matching mla-gpt's headline corpus); use
--limit-docs for a smaller slice. The last `--val-docs` documents form the
held-out split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from free_transformer.data import tokenize_split_to_bin, write_meta

OUT = Path(__file__).resolve().parents[1] / "data" / "fineweb_edu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="sample-10BT")
    ap.add_argument("--limit-docs", type=int, default=0, help="0 = all docs in the sample")
    ap.add_argument("--val-docs", type=int, default=48_000)
    ap.add_argument("--num-proc", type=int, default=8)
    args = ap.parse_args()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.sample, split="train")
    if args.limit_docs:
        ds = ds.select(range(min(args.limit_docs, len(ds))))
    n = len(ds)
    train_ds = ds.select(range(n - args.val_docs))
    val_ds = ds.select(range(n - args.val_docs, n))

    n_train = tokenize_split_to_bin(train_ds, "text", OUT / "train.bin", args.num_proc)
    n_val = tokenize_split_to_bin(val_ds, "text", OUT / "val.bin", args.num_proc)
    write_meta(OUT, dataset=f"HuggingFaceFW/fineweb-edu:{args.sample}",
               train_tokens=n_train, val_tokens=n_val, val_docs=args.val_docs)
    print(f"train: {n_train:,} tokens | val: {n_val:,} tokens -> {OUT}")


if __name__ == "__main__":
    main()
