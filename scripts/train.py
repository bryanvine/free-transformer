#!/usr/bin/env python3
"""Train one run from a YAML config, with dotted CLI overrides.

Usage:
    python scripts/train.py configs/dev_tinystories.yaml \
        --set model.model_type=free train.seed=2 train.out_dir=runs/dev_free_s2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from free_transformer.config import FTConfig
from free_transformer.training import TrainConfig, run_training


def _coerce(value: str, target_type):
    if target_type is bool:
        return value.lower() in ("1", "true", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="YAML file with `model:` and `train:` sections")
    ap.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                    help="overrides, e.g. model.latent_bits=8 train.seed=2")
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    model_kw = dict(raw.get("model") or {})
    train_kw = dict(raw.get("train") or {})

    types = {("model", f.name): f.type for f in fields(FTConfig)}
    types |= {("train", f.name): f.type for f in fields(TrainConfig)}
    py_types = {"int": int, "float": float, "bool": bool, "str": str}
    for ov in args.set:
        key, _, value = ov.partition("=")
        section, _, name = key.partition(".")
        if section not in ("model", "train") or not value:
            raise SystemExit(f"bad override {ov!r}; expected model.key=value or train.key=value")
        tname = str(types.get((section, name), "str"))
        target = next((t for n, t in py_types.items() if n in tname), str)
        (model_kw if section == "model" else train_kw)[name] = _coerce(value, target)

    model_cfg = FTConfig(**model_kw)
    tc = TrainConfig(**train_kw)
    run_training(model_cfg, tc)


if __name__ == "__main__":
    main()
