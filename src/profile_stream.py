"""Profile one Qwen3Guard-Stream per-token forward with torch.profiler.

Answers: at bs=1 per-token decode, how much of the ~25 ms wall time is
spent in GPU compute vs host-side overhead (Python + CUDA dispatch +
kernel launch), and which kernels dominate.

Run:
  uv run python src/profile_stream.py               # 0.6B, 32 per-token ops
  uv run python src/profile_stream.py --model-id Qwen/Qwen3Guard-Stream-4B

Output:
  - stdout: wall-time, kernel top-N, aggregate CPU vs CUDA time
  - results/profile_stream_<stamp>.json (chrome://tracing loadable)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_common import pick_device, pick_dtype, sync


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Stream-0.6B")
    ap.add_argument("--n-tokens", type=int, default=32,
                    help="Number of per-token forwards to profile.")
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = pick_device()
    dtype = pick_dtype(device)
    if device != "cuda":
        print("[profile-stream] requires CUDA", file=sys.stderr)
        return 1

    print(f"[profile-stream] model={args.model_id} device={device} dtype={dtype}")
    print(f"[profile-stream] torch={torch.__version__} "
          f"cuda={torch.version.cuda}")
    print(f"[profile-stream] sdp_math_enabled={torch.backends.cuda.math_sdp_enabled()} "
          f"flash_enabled={torch.backends.cuda.flash_sdp_enabled()} "
          f"efficient_enabled={torch.backends.cuda.mem_efficient_sdp_enabled()}")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id, dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    user_text = "What are the best practices for writing clean Python code?"
    messages = [{"role": "user", "content": user_text}]
    uid = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    uid_t = torch.tensor(uid, dtype=torch.long)

    asst_text = (
        "I understand your question. Here is a detailed response that tries to "
        "be helpful while staying within safe guidelines. Let me walk through "
        "the reasoning step by step so that the answer is clear and well "
        "structured."
    )
    asst_ids = tok.encode(asst_text, add_special_tokens=False)
    if len(asst_ids) < args.n_tokens + args.n_warmup:
        asst_ids = asst_ids * ((args.n_tokens + args.n_warmup) // len(asst_ids) + 1)
    asst_ids = asst_ids[: args.n_tokens + args.n_warmup]

    # Prefill + warmup.
    _, state = model.stream_moderate_from_ids(uid_t, role="user", stream_state=None)
    for i in range(args.n_warmup):
        tid = torch.tensor(asst_ids[i], dtype=torch.long)
        _, state = model.stream_moderate_from_ids(tid, role="assistant", stream_state=state)
    sync(device)

    # Probe output shape on one forward (outside the profiler).
    tid_probe = torch.tensor(asst_ids[args.n_warmup], dtype=torch.long)
    out, state = model.stream_moderate_from_ids(tid_probe, role="assistant",
                                                 stream_state=state)
    describe_output("per-token k=1 output shape probe", out)
    sync(device)

    # Wall-clock baseline over the upcoming window (outside the profiler to
    # avoid profiler overhead skewing the number).
    t_wall_start = time.perf_counter()
    for i in range(args.n_warmup + 1, args.n_warmup + args.n_tokens):
        tid = torch.tensor(asst_ids[i], dtype=torch.long)
        _, state = model.stream_moderate_from_ids(tid, role="assistant", stream_state=state)
    sync(device)
    wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
    n_ops = args.n_tokens - 2  # rough; used for average only
    print(f"[profile-stream] wall-clock per per-token op ≈ "
          f"{wall_ms / max(n_ops, 1):7.2f} ms (over {n_ops} ops, no profiler)")

    # Fresh prefill + warmup, then run the profiler.
    _, state = model.stream_moderate_from_ids(uid_t, role="user", stream_state=None)
    for i in range(args.n_warmup):
        tid = torch.tensor(asst_ids[i], dtype=torch.long)
        _, state = model.stream_moderate_from_ids(tid, role="assistant", stream_state=state)
    sync(device)

    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"profile_stream_{stamp}.json"

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for i in range(args.n_warmup, args.n_warmup + args.n_tokens):
            tid = torch.tensor(asst_ids[i], dtype=torch.long)
            _, state = model.stream_moderate_from_ids(tid, role="assistant",
                                                     stream_state=state)
        sync(device)

    prof.export_chrome_trace(str(trace_path))
    print(f"[profile-stream] wrote chrome trace {trace_path}")
    print()
    print(f"=== top 20 ops by self CUDA time ({args.n_tokens} per-token forwards) ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    print()
    print(f"=== top 20 ops by self CPU time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print()

    # Aggregate CPU vs CUDA totals — tells us how much of wall time is GPU
    # compute vs host overhead. CUDA time ≠ wall (launch latency bubbles can
    # leave GPU idle), so a large wall vs CUDA gap is the "host overhead" tell.
    avgs = prof.key_averages()
    total_self_cuda_us = sum(a.self_device_time_total for a in avgs)
    total_self_cpu_us = sum(a.self_cpu_time_total for a in avgs)
    print(f"[profile-stream] aggregate self-time over {args.n_tokens} per-token forwards:")
    print(f"                 self CUDA = {total_self_cuda_us / 1000.0:7.2f} ms "
          f"→ {total_self_cuda_us / 1000.0 / args.n_tokens:7.2f} ms / forward")
    print(f"                 self CPU  = {total_self_cpu_us / 1000.0:7.2f} ms "
          f"→ {total_self_cpu_us / 1000.0 / args.n_tokens:7.2f} ms / forward")
    print(f"                 wall      ≈ {wall_ms:7.2f} ms "
          f"→ {wall_ms / max(n_ops, 1):7.2f} ms / forward")
    gpu_util = (total_self_cuda_us / 1000.0) / wall_ms if wall_ms > 0 else 0.0
    print(f"                 GPU util (self CUDA / wall) ≈ {gpu_util:.2%}")
    return 0


def describe_output(label: str, out) -> None:
    """Print the structure of a stream_moderate_from_ids output for API discovery."""
    print(f"[probe] {label}")
    print(f"[probe]   type: {type(out).__name__}")
    if isinstance(out, tuple):
        print(f"[probe]   tuple len={len(out)}")
        for i, x in enumerate(out):
            _print_item(f"[{i}]", x)
        return
    if isinstance(out, dict):
        for k, v in out.items():
            _print_item(f"[{k!r}]", v)
        return
    # Dataclass-like object: dump public attrs.
    attrs = [a for a in dir(out) if not a.startswith("_")]
    for a in attrs:
        try:
            v = getattr(out, a)
        except Exception:
            continue
        if callable(v):
            continue
        _print_item(f".{a}", v)


def _print_item(key: str, v) -> None:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(v, torch.Tensor):
        print(f"[probe]   {key} Tensor shape={tuple(v.shape)} dtype={v.dtype} "
              f"device={v.device}")
    elif isinstance(v, (list, tuple)) and v and hasattr(v[0], "shape"):
        shapes = [tuple(x.shape) for x in v[:3]]
        print(f"[probe]   {key} list(len={len(v)}) first-shapes={shapes}")
    else:
        s = repr(v)
        if len(s) > 80:
            s = s[:80] + "..."
        print(f"[probe]   {key} {type(v).__name__} = {s}")


if __name__ == "__main__":
    sys.exit(main())
