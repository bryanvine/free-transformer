#!/usr/bin/env bash
# Single hardened 124M run for the Arc Pro B70 (inside the training container,
# invoked via arc_window.sh -> b70_sweep_entry.sh passthrough).
#
# Usage: run_xpu_124m.sh <baseline|free> <kappa_bits> <seed> [extra --set args...]
# Saturation posture matches sweep_dev.sh xpu: eager mode, moderate batch,
# 50ms/iter breather (arc_window.sh additionally caps clocks). 32GB VRAM ->
# no chunked CE needed (loss-exact either way; unchunked is faster).
set -u
ARM="${1:?usage: run_xpu_124m.sh <baseline|free> <kappa> <seed>}"
KAPPA="${2:?kappa}"
SEED="${3:?seed}"
shift 3

if [ "$ARM" = baseline ]; then
    NAME="ft124m_baseline_s${SEED}_xpu"
    EXTRA="model.model_type=baseline"
else
    NAME="ft124m_free_k${KAPPA}_s${SEED}_xpu"
    EXTRA="model.model_type=free model.kappa_bits=${KAPPA}"
fi
if [ -f "runs/${NAME}/summary.json" ]; then
    echo "[skip] ${NAME} (already complete)"
    exit 0
fi
echo "[run] ${NAME} ($(date +%H:%M))"
python3 scripts/train.py configs/ft124m_fineweb.yaml --set $EXTRA \
    model.chunked_ce=false \
    train.seed="$SEED" train.out_dir="runs/${NAME}" \
    train.compile=false train.batch_size=16 train.grad_accum=30 \
    train.iter_sleep_s=0.05 train.max_iters=5000 "$@" \
    > "runs/${NAME}.log" 2>&1 &
CHILD=$!
trap 'kill -TERM $CHILD 2>/dev/null; wait $CHILD; exit 143' TERM INT
wait "$CHILD"
RC=$?
printf '%s\t%s\t%s\n' "$NAME" "$RC" "$(date -Is)" >> runs/sweep_status_xpu.tsv
echo "[exit ${RC}] ${NAME}"
exit "$RC"
