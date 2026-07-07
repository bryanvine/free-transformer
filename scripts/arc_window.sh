#!/usr/bin/env bash
# Dedicated Arc training window for the B70 host.
#
# Stops the resident vLLM serving container (which pre-allocates most of the
# 32GB and starves training — see RESEARCH_LOG 2026-07-07), runs the xpu
# sweep share in a container, and ALWAYS restarts vLLM afterward: the trap
# fires on success, crash, or kill, and the script is designed to be run
# under nohup on the b70 host so it survives SSH disconnects.
#
# Usage (on the b70): nohup bash scripts/arc_window.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG=runs/arc_window.log
mkdir -p runs

restore() {
    docker start vllm-xpu >/dev/null 2>&1
    echo "[restore] vllm-xpu start issued $(date -Is)" >> "$LOG"
}
trap restore EXIT

echo "[window] stopping vllm-xpu $(date -Is)" >> "$LOG"
docker stop vllm-xpu >> "$LOG" 2>&1
docker rm -f ft-sweep >/dev/null 2>&1
echo "[window] training start $(date -Is)" >> "$LOG"
docker run --name ft-sweep --device /dev/dri --group-add 992 --group-add 44 \
    -v "$(pwd)":/work -w /work intel/vllm:latest \
    bash scripts/b70_sweep_entry.sh >> "$LOG" 2>&1
echo "[window] training container exited (rc=$?) $(date -Is)" >> "$LOG"
docker rm -f ft-sweep >/dev/null 2>&1
