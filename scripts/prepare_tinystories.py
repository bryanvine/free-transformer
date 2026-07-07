#!/usr/bin/env python3
"""Download and tokenize TinyStories into data/tinystories/{train,val}.bin (~530M train tokens)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from free_transformer.data import tokenize_split_to_bin, write_meta

OUT = Path(__file__).resolve().parents[1] / "data" / "tinystories"


def main() -> None:
    ds = load_dataset("roneneldan/TinyStories")
    n_train = tokenize_split_to_bin(ds["train"], "text", OUT / "train.bin")
    n_val = tokenize_split_to_bin(ds["validation"], "text", OUT / "val.bin")
    write_meta(OUT, dataset="roneneldan/TinyStories", train_tokens=n_train, val_tokens=n_val)
    print(f"train: {n_train:,} tokens | val: {n_val:,} tokens -> {OUT}")


if __name__ == "__main__":
    main()
