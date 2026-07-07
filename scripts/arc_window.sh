#!/usr/bin/env bash
# Dedicated Arc training window for the B70 host.
#
# Stops the resident vLLM serving container (which pre-allocates most of the
# 32GB and starves training — see RESEARCH_LOG 2026-07-07), runs the xpu
# sweep share in a container, and ALWAYS restarts vLLM afterward: the trap
# fires on success, crash, or kill, and the script is designed to be run
# under nohup on the b70 host so it survives SSH disconnects.
#
# HARD SAFETY RULE (learned 2026-07-07): never `docker rm -f` a container
# with in-flight XPU work — a force-kill mid-Level-Zero-operation wedged the
# GPU (forcewake MMIO reads of 0xFFFFFFFF) and only a reboot recovered it.
# Always `docker stop -t 60` first so training gets SIGTERM and exits.
#
# Usage (on the b70): nohup bash scripts/arc_window.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG="$PWD/arc_window.log"   # repo root: bryan-writable (runs/ may be root-owned)

restore() {
    docker stop -t 60 ft-sweep >/dev/null 2>&1   # graceful if still running
    docker rm ft-sweep >/dev/null 2>&1
    docker start vllm-xpu >/dev/null 2>&1
    echo "[restore] vllm-xpu start issued $(date -Is)" >> "$LOG"
}
trap restore EXIT

echo "[window] open $(date -Is)" >> "$LOG"
# repair runs/ ownership (containers write as root; host-side tools need access)
docker run --rm -v "$PWD":/work intel/vllm:latest chown -R 1000:1000 /work/runs /work/arc_window.log 2>/dev/null
docker stop -t 120 vllm-xpu >> "$LOG" 2>&1
docker rm ft-sweep >/dev/null 2>&1
echo "[window] training start $(date -Is)" >> "$LOG"
docker run --name ft-sweep --device /dev/dri --group-add 992 --group-add 44 \
    -v "$PWD":/work -w /work intel/vllm:latest \
    bash scripts/b70_sweep_entry.sh >> "$LOG" 2>&1
echo "[window] training container exited (rc=$?) $(date -Is)" >> "$LOG"
