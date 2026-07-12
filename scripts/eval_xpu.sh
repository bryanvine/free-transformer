#!/usr/bin/env bash
# Honest-eval sweep over the B70-resident dev checkpoints (run inside the
# arc_window container): posterior-CE, prior-CE, KL, and the tight
# posterior-proposal IWAE bound for every runs/dev_*_xpu checkpoint.
set -u
python3 scripts/eval_prior.py --runs-glob "runs/dev_*_xpu" \
    --data-dir data/tinystories --batch-size 16 \
    --iwae-k 16 --iwae-batches 40 --iwae-posterior \
    --out runs/eval_xpu_dev.json
