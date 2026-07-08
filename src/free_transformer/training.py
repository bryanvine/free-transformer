"""Single-GPU training loop shared by both arms (and both vendors).

bf16 autocast, gradient accumulation, cosine LR with warmup, grad clipping,
periodic evaluation, checkpointing, and CSV metric logging. Runs unchanged on
CUDA (RTX 5060/4080S) and Intel XPU (Arc Pro B70) — device quirks are isolated
here so the model code stays vendor-free. The same code path trains baseline
and free models (only ``FTConfig.model_type`` differs), so training is a
controlled variable. For free models the CSV additionally logs the CE/KL split
— the KL trace is the posterior-collapse instrument.
"""

from __future__ import annotations

import csv
import json
import math
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .config import FTConfig
from .data import make_get_batch
from .model import FreeTransformer


@dataclass
class TrainConfig:
    data_dir: str = "data/tinystories"
    out_dir: str = "runs/default"
    # optimization
    batch_size: int = 32
    grad_accum: int = 4
    max_iters: int = 5000
    lr_decay_iters: int = 0          # 0 -> use max_iters
    warmup_iters: int = 200
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # eval / logging
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 10
    always_save_checkpoint: bool = False
    # system
    seed: int = 1337
    device: str = "auto"             # auto -> cuda > xpu > cpu
    dtype: str = "bfloat16"
    compile: bool = True
    resume: bool = True


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def device_type_of(device: str) -> str:
    for t in ("cuda", "xpu"):
        if device.startswith(t):
            return t
    return "cpu"


def _lr_at(it: int, c: TrainConfig) -> float:
    horizon = c.lr_decay_iters if c.lr_decay_iters > 0 else c.max_iters
    if it < c.warmup_iters:
        return c.lr * (it + 1) / (c.warmup_iters + 1)
    if it > horizon:
        return c.min_lr
    ratio = (it - c.warmup_iters) / max(1, horizon - c.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return c.min_lr + coeff * (c.lr - c.min_lr)


@torch.no_grad()
def estimate_loss(model, get_batch, tc: TrainConfig, ctx) -> dict[str, float]:
    """Mean loss per split; for free models also mean CE and KL (bits/token)."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(tc.eval_iters)
        ces = torch.zeros(tc.eval_iters)
        kls = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            x, y = get_batch(split, tc.batch_size)
            with ctx:
                logits, loss, aux = model(x, y)
            losses[k] = loss.item()
            ces[k] = aux.get("ce", loss).item()
            kls[k] = aux.get("kl_bits_per_token", torch.tensor(0.0)).item()
            logits = loss = None  # release full-vocab logits before the next forward
        out[split] = losses.mean().item()
        out[f"{split}_ce"] = ces.mean().item()
        out[f"{split}_kl_bits"] = kls.mean().item()
    model.train()
    return out


def run_training(model_cfg: FTConfig, tc: TrainConfig) -> dict:
    tc.device = resolve_device(tc.device)
    device_type = device_type_of(tc.device)
    torch.manual_seed(tc.seed)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(tc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[tc.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=ptdtype)

    block_size = model_cfg.block_size
    get_batch = make_get_batch(tc.data_dir, block_size, tc.device)
    tokens_per_iter = tc.batch_size * tc.grad_accum * block_size

    model = FreeTransformer(model_cfg).to(tc.device)
    raw_model = model
    if tc.compile and device_type in ("cuda", "xpu"):
        try:
            model = torch.compile(model)
        except Exception as e:  # pragma: no cover
            print(f"[warn] torch.compile failed ({e}); continuing eager")
            model = raw_model

    optimizer = raw_model.configure_optimizers(tc.weight_decay, tc.lr, (tc.beta1, tc.beta2), device_type)
    scaler = torch.amp.GradScaler(enabled=(tc.dtype == "float16" and device_type == "cuda"))

    (out_dir / "config.json").write_text(
        json.dumps({"model": model_cfg.to_dict(), "train": asdict(tc),
                    "params": raw_model.num_params(),
                    "tokens_per_iter": tokens_per_iter}, indent=2)
    )

    best_val = float("inf")
    start_iter = 0
    ckpt_path = out_dir / "ckpt.pt"
    if tc.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=tc.device)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_iter = int(ck["iter"])
        best_val = float(ck["best_val"])
        torch.set_rng_state(ck["cpu_rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        print(f"[resume] {tc.out_dir} continuing from iter {start_iter} (best_val {best_val:.4f})")

    metrics_path = out_dir / "metrics.csv"
    fresh = start_iter == 0 or not metrics_path.exists()
    metrics_file = metrics_path.open("w" if fresh else "a", newline="")
    writer = csv.writer(metrics_file)
    if fresh:
        writer.writerow(["iter", "time_s", "lr", "train_loss", "val_loss",
                         "val_ce", "val_kl_bits", "tokens", "tok_per_s"])

    print(f"[{tc.out_dir}] {model_cfg.model_type} | params={raw_model.num_params():,} "
          f"| device={tc.device} | tok/iter={tokens_per_iter:,}")

    # SIGTERM -> finish the current iteration, checkpoint, exit 0. Abrupt
    # kills mid-GPU-op wedge the Arc B70 (see RESEARCH_LOG 2026-07-07/08);
    # this plus signal-forwarding in sweep_dev.sh makes `docker stop` safe.
    stop_flag = {"v": False}

    def _on_term(signum, frame):  # pragma: no cover - signal timing
        stop_flag["v"] = True
        print("[signal] SIGTERM: will checkpoint and exit at iteration boundary", flush=True)

    try:
        signal.signal(signal.SIGTERM, _on_term)
    except ValueError:  # not in main thread (e.g. under some test runners)
        pass

    def save_resume_ckpt(it: int) -> None:
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iter": it, "best_val": best_val,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "model_cfg": model_cfg.to_dict()}, ckpt_path)

    t0 = time.time()
    x, y = get_batch("train", tc.batch_size)
    running = None
    for it in range(start_iter, tc.max_iters + 1):
        if stop_flag["v"]:
            save_resume_ckpt(it)
            print(f"[signal] checkpointed at iter {it}; exiting cleanly", flush=True)
            break
        lr = _lr_at(it, tc)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % tc.eval_interval == 0 and not (it == start_iter and start_iter > 0):
            losses = estimate_loss(model, get_batch, tc, ctx)
            dt = time.time() - t0
            done = it - start_iter
            tps = (done * tokens_per_iter) / dt if done > 0 else 0.0
            writer.writerow([it, f"{dt:.1f}", f"{lr:.2e}", f"{losses['train']:.4f}",
                             f"{losses['val']:.4f}", f"{losses['val_ce']:.4f}",
                             f"{losses['val_kl_bits']:.3f}", it * tokens_per_iter, f"{tps:.0f}"])
            metrics_file.flush()
            print(f"iter {it:>6d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| val_ce {losses['val_ce']:.4f} | kl {losses['val_kl_bits']:.3f}b "
                  f"| lr {lr:.2e} | {tps/1e3:.1f}k tok/s")
            if losses["val"] < best_val or tc.always_save_checkpoint:
                best_val = min(best_val, losses["val"])
                torch.save({"model": raw_model.state_dict(), "model_cfg": model_cfg.to_dict(),
                            "iter": it, "val_loss": losses["val"]}, out_dir / "best.pt")
            save_resume_ckpt(it)

        if it == tc.max_iters:
            break

        for micro in range(tc.grad_accum):
            with ctx:
                _, loss, _ = model(x, y)
                loss = loss / tc.grad_accum
            x, y = get_batch("train", tc.batch_size)  # prefetch next while GPU works
            scaler.scale(loss).backward()
        if tc.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running = loss.item() * tc.grad_accum if running is None else \
            0.9 * running + 0.1 * loss.item() * tc.grad_accum
        if it % tc.log_interval == 0:
            print(f"iter {it:>6d} | loss {running:.4f} | lr {lr:.2e}", flush=True)

    metrics_file.close()
    total_time = time.time() - t0
    summary = {"model_type": model_cfg.model_type, "best_val_loss": best_val,
               "params": raw_model.num_params(), "total_time_s": total_time,
               "final_iter": tc.max_iters, "tokens_seen": tc.max_iters * tokens_per_iter,
               "device": tc.device, "seed": tc.seed}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] best val {best_val:.4f} in {total_time/60:.1f} min -> {out_dir}")
    return summary
