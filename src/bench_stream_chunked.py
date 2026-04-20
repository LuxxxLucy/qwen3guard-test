"""Chunked-streaming latency sweep for Qwen3Guard-Stream.

Hypothesis under test: the per-token regime (one forward per emitted
token) is wasteful because each forward pays full weight-bandwidth cost
regardless of how much new context it advances. A single forward that
advances k new tokens should cost ~ the same as a single forward that
advances 1 token at short context, giving effective per-token latency
≈ forward_ms(k) / k. Since LLM streaming naturally delivers tokens in
chunks (host scheduling granularity is rarely 1 token), classifying in
chunks is the natural mitigation and does not require batching across
concurrent streams.

For each chunk size k, we prefill once, then advance the stream in
fixed-size chunks of k assistant tokens and time each chunk's forward.
We report per-chunk P50/P99 latency and effective per-token P50/P99
latency (= per-chunk / k), plus a T2 verdict (< 8 ms effective).

A second pass probes the return-value shape: if stream_moderate_from_ids
yields one verdict per new position when given a k-token chunk, chunked
streaming preserves token-granular verdicts (true "chunked stream"
semantic). If it yields a single verdict per call, chunking becomes a
subsample-at-chunk-boundaries semantic — still useful, but different.

Run:
  uv run python src/bench_stream_chunked.py \
      --model-id Qwen/Qwen3Guard-Stream-0.6B \
      --chunks 1 2 4 8 16 32 64
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from bench_common import pick_device, pick_dtype, sync


def pct(sorted_ms: list[float], p: float) -> float:
    if not sorted_ms:
        return 0.0
    k = max(0, min(len(sorted_ms) - 1, int(round((p / 100.0) * (len(sorted_ms) - 1)))))
    return sorted_ms[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Stream-0.6B")
    ap.add_argument("--chunks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--n-assistant-tokens", type=int, default=128,
                    help="Total assistant tokens streamed per prompt.")
    ap.add_argument("--n-prompts", type=int, default=20,
                    help="Number of full streaming runs (fresh prefill each).")
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--user-len", type=int, default=81,
                    help="Target user-content token length (wrapped in chat template).")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = pick_device()
    dtype = pick_dtype(device)

    print(f"[bench-chunked] model={args.model_id} device={device} dtype={dtype}")
    print(f"[bench-chunked] n_prompts={args.n_prompts} n_warmup={args.n_warmup} "
          f"n_assistant_tokens={args.n_assistant_tokens} user_len={args.user_len}")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id, dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    # Build a stable user prompt at the requested length, wrapped in the
    # response-moderation chat template (role="user" side).
    seed = ("The quick brown fox jumps over the lazy dog near the old wooden "
            "bridge while the river flows silently through the valley. ")
    user_ids_raw: list[int] = []
    while len(user_ids_raw) < args.user_len:
        user_ids_raw.extend(tok.encode(seed, add_special_tokens=False))
    user_text = tok.decode(user_ids_raw[:args.user_len], skip_special_tokens=True)
    messages = [{"role": "user", "content": user_text}]
    uid = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print(f"[bench-chunked] prefill token count = {len(uid)}")

    # A long-enough synthetic assistant continuation.
    asst_text = (
        "I understand your question. Here is a detailed response that tries to "
        "be helpful while staying within safe guidelines. Let me walk through "
        "the reasoning step by step so that the answer is clear and well "
        "structured. Each suggestion builds on the previous one, and together "
        "they should give you a concrete, actionable plan going forward now."
    )
    asst_ids_all = tok.encode(asst_text, add_special_tokens=False)
    while len(asst_ids_all) < args.n_assistant_tokens:
        asst_ids_all = asst_ids_all + asst_ids_all
    asst_ids = asst_ids_all[: args.n_assistant_tokens]

    # One-shot probe: dump the output shape for a k=1 and a k=max(chunks) call.
    k_max = max(args.chunks)
    _probe_output_shape(model, tok, uid, asst_ids, k=1, label="probe k=1")
    _probe_output_shape(model, tok, uid, asst_ids, k=k_max, label=f"probe k={k_max}")
    sync(device)

    def run_chunk_size(k: int) -> dict:
        per_chunk_s: list[float] = []

        # Warmup — include one full streaming run at this k.
        for _ in range(args.n_warmup):
            _, state = model.stream_moderate_from_ids(
                torch.tensor(uid, dtype=torch.long), role="user", stream_state=None)
            for i in range(0, min(k * 4, len(asst_ids)), k):
                chunk_ids = asst_ids[i:i + k]
                if len(chunk_ids) != k:
                    break
                chunk = torch.tensor(chunk_ids, dtype=torch.long)
                _, state = model.stream_moderate_from_ids(
                    chunk, role="assistant", stream_state=state)
            _close(model, state)
        sync(device)

        # Timed runs.
        for _ in range(args.n_prompts):
            _, state = model.stream_moderate_from_ids(
                torch.tensor(uid, dtype=torch.long), role="user", stream_state=None)
            for i in range(0, len(asst_ids), k):
                chunk_ids = asst_ids[i:i + k]
                if len(chunk_ids) != k:
                    break  # only measure full-size chunks
                chunk = torch.tensor(chunk_ids, dtype=torch.long)
                sync(device)
                t0 = time.perf_counter()
                _, state = model.stream_moderate_from_ids(
                    chunk, role="assistant", stream_state=state)
                sync(device)
                per_chunk_s.append(time.perf_counter() - t0)
            _close(model, state)

        ms_sorted = sorted(s * 1000.0 for s in per_chunk_s)
        n = len(ms_sorted)
        return {
            "k": k,
            "n_chunks": n,
            "p50_ms": pct(ms_sorted, 50),
            "p95_ms": pct(ms_sorted, 95),
            "p99_ms": pct(ms_sorted, 99),
            "mean_ms": statistics.fmean(ms_sorted) if ms_sorted else 0.0,
        }

    rows = [run_chunk_size(k) for k in args.chunks]

    # Pretty-print summary.
    print()
    print(f"{'k':>4} {'n_chunks':>9} "
          f"{'chunk P50 ms':>13} {'chunk P99 ms':>13} "
          f"{'eff/tok P50':>12} {'eff/tok P99':>12} "
          f"{'T2 (<8 ms)':>11}  speedup_vs_k1")
    p99_k1 = next((r["p99_ms"] for r in rows if r["k"] == 1), rows[0]["p99_ms"])
    eff_k1 = p99_k1  # effective per-token P99 at k=1 = chunk P99 at k=1.
    for r in rows:
        eff_p50 = r["p50_ms"] / r["k"]
        eff_p99 = r["p99_ms"] / r["k"]
        t2 = "yes" if eff_p99 < 8.0 else "no"
        speedup = eff_k1 / eff_p99 if eff_p99 > 0 else 0.0
        print(f"{r['k']:>4} {r['n_chunks']:>9} "
              f"{r['p50_ms']:>10.2f} ms {r['p99_ms']:>10.2f} ms "
              f"{eff_p50:>9.2f} ms {eff_p99:>9.2f} ms "
              f"{t2:>11}  {speedup:>6.2f}x")
    return 0


def _close(model, state) -> None:
    if hasattr(model, "close_stream"):
        try:
            model.close_stream(state)
        except Exception:
            pass


def _probe_output_shape(model, tok, uid, asst_ids, k: int, label: str) -> None:
    import torch
    _, state = model.stream_moderate_from_ids(
        torch.tensor(uid, dtype=torch.long), role="user", stream_state=None)
    chunk = torch.tensor(asst_ids[:k], dtype=torch.long)
    out, _ = model.stream_moderate_from_ids(
        chunk, role="assistant", stream_state=state)
    print(f"[bench-chunked] {label}: input chunk shape={tuple(chunk.shape)}  "
          f"output type={type(out).__name__}")
    # Dump shapes of tensor-valued attributes / entries.
    if isinstance(out, tuple):
        for i, x in enumerate(out):
            _dump(f"  out[{i}]", x)
    elif isinstance(out, dict):
        for key, v in out.items():
            _dump(f"  out[{key!r}]", v)
    else:
        for a in dir(out):
            if a.startswith("_"):
                continue
            try:
                v = getattr(out, a)
            except Exception:
                continue
            if callable(v):
                continue
            _dump(f"  out.{a}", v)


def _dump(label: str, v) -> None:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(v, torch.Tensor):
        print(f"[bench-chunked] {label}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(v, (list, tuple)) and v and hasattr(v[0], "shape"):
        print(f"[bench-chunked] {label}: list(len={len(v)}) first shape={tuple(v[0].shape)}")


if __name__ == "__main__":
    sys.exit(main())
