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
# Usage (on the b70): nohup bash scripts/arc_window.sh [inner cmd...] >/dev/null 2>&1 &
# Extra args are forwarded to b70_sweep_entry.sh (default: the dev sweep),
# e.g.: nohup bash scripts/arc_window.sh scripts/run_xpu_124m.sh free 1 1 &
set -u
cd "$(dirname "$0")/.."
LOG="$PWD/arc_window.log"   # repo root: bryan-writable (runs/ may be root-owned)

GT=/sys/class/drm/card0/device/tile0/gt0/freq0
FREQ_CAP=2000          # MHz during training (stock max 2800): the B70 crashes
                       # under sustained saturation; ~30% clock cut buys a
                       # large power/stability margin for ~20% throughput.

restore() {
    sudo pkill -f "dmesg --follow-new" 2>/dev/null
    docker stop -t 90 ft-sweep >/dev/null 2>&1   # graceful: SIGTERM reaches the trainer
    docker rm ft-sweep >/dev/null 2>&1
    [ -n "${OLD_MAX:-}" ] && echo "$OLD_MAX" | sudo tee "$GT/max_freq" >/dev/null 2>&1
    docker start vllm-xpu >/dev/null 2>&1
    sudo systemctl start apt-daily.timer apt-daily-upgrade.timer 2>/dev/null
    echo "[restore] vllm-xpu start issued, apt timers resumed, max_freq restored to ${OLD_MAX:-untouched} $(date -Is)" >> "$LOG"
}
trap restore EXIT

echo "[window] open $(date -Is)" >> "$LOG"
# An unattended containerd upgrade mid-window SIGKILLed training and wedged
# the GPU (2026-07-07). Pause the apt timers for the window; trap restores.
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer 2>/dev/null

# Saturation hardening: cap GPU clocks for the window (restored by trap).
OLD_MAX=$(cat "$GT/max_freq" 2>/dev/null)
[ -n "$OLD_MAX" ] && echo "$FREQ_CAP" | sudo tee "$GT/max_freq" >/dev/null 2>&1 \
    && echo "[window] max_freq ${OLD_MAX} -> ${FREQ_CAP} MHz" >> "$LOG"

# Watchdog: at the FIRST kernel GPU warning, stop training gracefully (the
# trainer checkpoints on SIGTERM) instead of letting a hang become a wedge.
( sudo dmesg --follow-new 2>/dev/null \
    | grep -m1 --line-buffered -E "forcewake|MMIO unreliable|cdclk|GT0: reset|GuC" \
    >> "$LOG" \
  && echo "[watchdog] GPU warning -> graceful stop $(date -Is)" >> "$LOG" \
  && docker stop -t 90 ft-sweep >/dev/null 2>&1 ) &
# repair runs/ ownership (containers write as root; host-side tools need access)
docker run --rm -v "$PWD":/work intel/vllm:latest chown -R 1000:1000 /work/runs /work/arc_window.log 2>/dev/null
docker stop -t 120 vllm-xpu >> "$LOG" 2>&1
docker rm ft-sweep >/dev/null 2>&1
echo "[window] training start $(date -Is)" >> "$LOG"
docker run --name ft-sweep --device /dev/dri --group-add 992 --group-add 44 \
    -v "$PWD":/work -w /work intel/vllm:latest \
    bash scripts/b70_sweep_entry.sh "$@" >> "$LOG" 2>&1
echo "[window] training container exited (rc=$?) $(date -Is)" >> "$LOG"
