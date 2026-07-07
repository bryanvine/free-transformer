#!/usr/bin/env bash
# Container entrypoint for the B70 sweep. setvars.sh can exit nonzero even on
# success (e.g. code 3 with partial components), so never && off of it.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
python3 -c "import torch; assert torch.xpu.is_available(), 'XPU not available after setvars'" || exit 9
exec bash scripts/sweep_dev.sh xpu python3
