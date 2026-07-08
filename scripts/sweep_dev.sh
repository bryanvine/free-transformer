#!/usr/bin/env bash
# Phase 1 dev sweep (51M, TinyStories, 131M tokens/run):
#   {baseline, free x kappa in {0.125, 0.5, 1, 2, 4}} x seeds {1,2,3}
# split across hosts. Usage: scripts/sweep_dev.sh <cuda|xpu> [python]
#
# cuda host (RTX 5060 8GB): baseline + kappa {0.125, 0.5, 4}, compile on,
#   batch 16 x accum 4 (measured peak 4.6 GiB beside a 1.5 GiB llama-server).
# xpu host (Arc Pro B70 32GB): kappa {1, 2} + a duplicate baseline seed-1 as
#   the cross-backend anchor; eager mode (Triton-XPU autotune is the known
#   flaky part of the stack — throughput matters less than a clean overnight).
#
# Runs are sequential; a run whose summary.json exists is skipped, so the
# script is safe to re-launch after a crash (train.py also self-resumes from
# its checkpoint). Per-run exit codes -> runs/sweep_status_<host>.tsv.
set -u
HOST_KIND="${1:?usage: sweep_dev.sh <cuda|xpu> [python]}"
PY="${2:-.venv/bin/python}"
CFG=configs/dev_tinystories.yaml
STATUS="runs/sweep_status_${HOST_KIND}.tsv"
mkdir -p runs

EXTRA_TRAIN=""
if [ "$HOST_KIND" = cuda ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    COMPILE=true;  BS=16; GA=4
    RUNS=(
        "baseline 0 1" "baseline 0 2" "baseline 0 3"
        "free 0.5 1"   "free 0.5 2"   "free 0.5 3"
        "free 0.125 1" "free 0.125 2" "free 0.125 3"
        "free 4 1"     "free 4 2"     "free 4 3"
        "free 1 1"     "free 1 2"     "free 1 3"
        "free 2 1"     "free 2 2"     "free 2 3"
    )
else
    # Dedicated-window share (vLLM paused; see scripts/arc_window.sh): the
    # cross-backend anchors — one arm each, seed 1, mirroring cuda runs.
    # kappa {1,2} moved to cuda after the contention descope (RESEARCH_LOG).
    # Saturation hardening (the B70 crashes under sustained peak load, cf.
    # the c>14 serving crashes): halved instantaneous batch (same tokens/
    # iter) + 50ms/iter breather; arc_window.sh additionally caps clocks.
    COMPILE=false; BS=16; GA=4
    EXTRA_TRAIN="train.iter_sleep_s=0.05"
    RUNS=(
        "baseline 0 1"
        "free 0.5 1"
    )
fi

for spec in "${RUNS[@]}"; do
    read -r ARM KAPPA SEED <<<"$spec"
    if [ "$ARM" = baseline ]; then
        NAME="dev_baseline_s${SEED}"
        EXTRA="model.model_type=baseline"
    else
        NAME="dev_free_k${KAPPA}_s${SEED}"
        EXTRA="model.model_type=free model.kappa_bits=${KAPPA}"
    fi
    [ "$HOST_KIND" = xpu ] && NAME="${NAME}_xpu"
    if [ -f "runs/${NAME}/summary.json" ]; then
        echo "[skip] ${NAME} (already complete)"
        continue
    fi
    echo "[run] ${NAME} ($(date +%H:%M))"
    # Run in background + wait, with SIGTERM forwarded to the trainer so
    # `docker stop` reaches python's checkpoint-and-exit handler (bash as
    # PID 1 would otherwise swallow the signal and the grace period would
    # end in SIGKILL mid-GPU-op — which wedges the B70).
    "$PY" scripts/train.py "$CFG" --set $EXTRA $EXTRA_TRAIN \
        train.seed="$SEED" train.out_dir="runs/${NAME}" \
        train.compile="$COMPILE" train.batch_size="$BS" train.grad_accum="$GA" \
        > "runs/${NAME}.log" 2>&1 &
    CHILD=$!
    trap 'kill -TERM $CHILD 2>/dev/null; wait $CHILD; exit 143' TERM INT
    wait "$CHILD"
    RC=$?
    trap - TERM INT
    printf '%s\t%s\t%s\n' "$NAME" "$RC" "$(date -Is)" >> "$STATUS"
    echo "[exit ${RC}] ${NAME}"
done
echo "[sweep ${HOST_KIND} done] $(date)"
