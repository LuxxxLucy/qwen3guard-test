"""Direct-path Qwen3Guard-Stream length × chunk-size sweep.

Measures per-chunk latency across a grid of user-prompt lengths and
streaming chunk sizes, so we can see where per-token cost crosses the
8 ms T2 budget as accumulated context grows.

Uses the same direct path as bench_stream_direct_heads.py — calls
model.forward(...) with a DynamicCache and advances the stream in
fixed k-token chunks. Classification heads are inside the timed region.
No correctness checks (those are done once by bench_stream_direct_heads;
see debug_stream_direct_all.sh).

Run:
  uv run python src/bench_stream_direct_length_sweep.py \
      --model-id Qwen/Qwen3Guard-Stream-0.6B \
      --lengths 81 256 1024 2048 4096 \
      --chunks 1 4 8 16
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from bench_common import pick_device, pick_dtype, sync


def pct(sorted_ms: list[float], p: float) -> float:
    if not sorted_ms:
        return 0.0
    k = max(0, min(len(sorted_ms) - 1, int(round((p / 100.0) * (len(sorted_ms) - 1)))))
    return sorted_ms[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Stream-0.6B")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[81, 256, 1024, 2048, 4096])
    ap.add_argument("--chunks", type=int, nargs="+",
                    default=[1, 4, 8, 16])
    ap.add_argument("--n-assistant-tokens", type=int, default=64,
                    help="Total assistant tokens per prompt per cell.")
    ap.add_argument("--n-prompts", type=int, default=5,
                    help="Number of full streaming runs per cell.")
    ap.add_argument("--n-warmup", type=int, default=2)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[length-sweep] model={args.model_id} device={device} dtype={dtype}")
    print(f"[length-sweep] lengths={args.lengths}  chunks={args.chunks}  "
          f"n_prompts={args.n_prompts}  n_asst={args.n_assistant_tokens}")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id, dtype=dtype, trust_remote_code=True,
    ).to(device).eval()
    print(f"[length-sweep] model class = {type(model).__name__}")

    # Long-enough assistant continuation.
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

    seed = ("The quick brown fox jumps over the lazy dog near the old wooden "
            "bridge while the river flows silently through the valley. ")

    def build_uid(target_user_tokens: int):
        user_raw: list[int] = []
        while len(user_raw) < target_user_tokens:
            user_raw.extend(tok.encode(seed, add_special_tokens=False))
        user_text = tok.decode(user_raw[: target_user_tokens],
                               skip_special_tokens=True)
        uid = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=True, add_generation_prompt=True,
        )
        return uid

    def run_cell(uid_tensor: torch.Tensor, k: int) -> dict:
        per_chunk_s: list[float] = []

        # Warmup.
        for _ in range(args.n_warmup):
            cache = DynamicCache()
            with torch.inference_mode():
                model.forward(input_ids=uid_tensor, past_key_values=cache,
                              use_cache=True)
                cache_next = cache
                for i in range(0, min(k * 4, len(asst_ids)), k):
                    chunk_ids = asst_ids[i:i + k]
                    if len(chunk_ids) != k:
                        break
                    chunk = torch.tensor([chunk_ids], dtype=torch.long,
                                         device=device)
                    out = model.forward(input_ids=chunk,
                                        past_key_values=cache_next,
                                        use_cache=True)
                    cache_next = out.past_key_values
        sync(device)

        # Timed runs.
        for _ in range(args.n_prompts):
            cache = DynamicCache()
            with torch.inference_mode():
                model.forward(input_ids=uid_tensor, past_key_values=cache,
                              use_cache=True)
                cache_next = cache
                for i in range(0, len(asst_ids), k):
                    chunk_ids = asst_ids[i:i + k]
                    if len(chunk_ids) != k:
                        break
                    chunk = torch.tensor([chunk_ids], dtype=torch.long,
                                         device=device)
                    sync(device)
                    t0 = time.perf_counter()
                    out = model.forward(input_ids=chunk,
                                        past_key_values=cache_next,
                                        use_cache=True)
                    sync(device)
                    per_chunk_s.append(time.perf_counter() - t0)
                    cache_next = out.past_key_values

        ms_sorted = sorted(s * 1000.0 for s in per_chunk_s)
        return {
            "k": k, "n_chunks": len(ms_sorted),
            "p50_ms": pct(ms_sorted, 50),
            "p95_ms": pct(ms_sorted, 95),
            "p99_ms": pct(ms_sorted, 99),
            "mean_ms": statistics.fmean(ms_sorted) if ms_sorted else 0.0,
        }

    # Sweep.
    for target_user_len in args.lengths:
        uid = build_uid(target_user_len)
        uid_tensor = torch.tensor([uid], dtype=torch.long, device=device)
        total_prefill_tokens = len(uid)
        actual_user_tokens = total_prefill_tokens - 8  # approximate template overhead

        print()
        print("=" * 72)
        print(f"[length-sweep] user_len target={target_user_len}  "
              f"actual_user≈{actual_user_tokens}  total_prefill={total_prefill_tokens}")
        print("=" * 72)
        print(f"{'k':>4} {'n_chunks':>9} "
              f"{'chunk P50 ms':>13} {'chunk P99 ms':>13} "
              f"{'eff/tok P50':>12} {'eff/tok P99':>12} "
              f"{'T2 (<8 ms)':>11}")
        for k in args.chunks:
            r = run_cell(uid_tensor, k)
            eff_p50 = r["p50_ms"] / r["k"]
            eff_p99 = r["p99_ms"] / r["k"]
            t2 = "yes" if eff_p99 < 8.0 else "no"
            print(f"{r['k']:>4} {r['n_chunks']:>9} "
                  f"{r['p50_ms']:>10.2f} ms {r['p99_ms']:>10.2f} ms "
                  f"{eff_p50:>9.2f} ms {eff_p99:>9.2f} ms "
                  f"{t2:>11}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
