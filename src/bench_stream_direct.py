"""Direct-path Qwen3Guard-Stream latency sweep — bypasses the custom
`stream_moderate_from_ids` API.

The custom API has two inefficiencies that this path eliminates:
  (a) Its internal `stream_generate` coroutine hardcodes 1 token per
      `.send()` call via `torch.tensor([next_token_id])`, so k>1 chunks
      cannot be advanced in a single forward.
  (b) It re-reports every accumulated per-token verdict as Python
      strings/floats on every call, forcing ~160 host syncs per forward
      (memcpy DtoH + `_local_scalar_dense`) that stall the CUDA stream
      between kernels. The profiler shows these syncs cost ~32 % of wall
      time on 0.6B.

This path calls the underlying Qwen3 transformer backbone directly with
a `DynamicCache`, feeding chunks of k assistant tokens per forward. It
measures pure backbone-forward latency — no verdict readout inside the
timed region — so the numbers answer the chunking-amortization question
independent of the custom API's overhead.

Interpretation:
  - If per-chunk latency at k=8 is close to per-chunk at k=1, chunking
    is near-free at short context and effective per-token ms drops ~k×.
  - If direct-k=1 is much faster than `stream_moderate_from_ids`-k=1,
    the custom API wrapper is itself a major cost (host syncs).

Run:
  uv run python src/bench_stream_direct.py \
      --model-id Qwen/Qwen3Guard-Stream-0.6B \
      --chunks 1 2 4 8 16 32 64
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
    ap.add_argument("--chunks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--n-assistant-tokens", type=int, default=128)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--user-len", type=int, default=81)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[bench-direct] model={args.model_id} device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id, dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    # --- Introspect the top-level model to find the transformer backbone.
    print("[bench-direct] top-level children of the loaded model:")
    backbone = None
    backbone_name = None
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        has_layers = hasattr(child, "layers")
        has_embed = hasattr(child, "embed_tokens")
        tag = ""
        if has_layers and has_embed:
            tag = " <- backbone candidate"
            if backbone is None:
                backbone, backbone_name = child, name
        print(f"  .{name}: {type(child).__name__}  "
              f"({n_params:,} params){tag}")

    if backbone is None:
        print("[bench-direct] ERROR: could not find a transformer backbone "
              "(looked for a child module with both .layers and .embed_tokens). "
              "Dumping full module tree (top 3 levels) and exiting.", file=sys.stderr)
        for name, module in model.named_modules():
            if name.count(".") <= 2:
                print(f"  {name or '<root>'}: {type(module).__name__}")
        return 1

    n_layers = len(backbone.layers) if hasattr(backbone, "layers") else -1
    print(f"[bench-direct] backbone = model.{backbone_name} "
          f"({type(backbone).__name__}, n_layers={n_layers})")

    # --- Load the Cache utility. Fall back if unavailable (very old HF).
    try:
        from transformers.cache_utils import DynamicCache
        cache_ctor = DynamicCache
        print(f"[bench-direct] using DynamicCache")
    except ImportError:
        cache_ctor = lambda: None  # type: ignore
        print(f"[bench-direct] DynamicCache unavailable; using None past_key_values")

    # --- Build user prompt of the requested length, wrapped in chat template.
    seed = ("The quick brown fox jumps over the lazy dog near the old wooden "
            "bridge while the river flows silently through the valley. ")
    user_raw: list[int] = []
    while len(user_raw) < args.user_len:
        user_raw.extend(tok.encode(seed, add_special_tokens=False))
    user_text = tok.decode(user_raw[: args.user_len], skip_special_tokens=True)
    uid = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True, add_generation_prompt=True,
    )
    print(f"[bench-direct] prefill token count = {len(uid)}")

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

    uid_tensor = torch.tensor([uid], dtype=torch.long, device=device)

    # --- Probe a single chunked forward to confirm the backbone accepts
    #     multi-token input + KV cache, and dump shapes.
    print()
    print("[bench-direct] --- direct-path probe ---")
    try:
        cache = cache_ctor()
        with torch.inference_mode():
            out = backbone(input_ids=uid_tensor, past_key_values=cache,
                           use_cache=True)
            probe_cache = out.past_key_values
            print(f"[bench-direct] prefill OK — last_hidden_state shape="
                  f"{tuple(out.last_hidden_state.shape)}")
            probe_k = max(args.chunks)
            probe_chunk = torch.tensor(
                [asst_ids[:probe_k]], dtype=torch.long, device=device)
            out2 = backbone(input_ids=probe_chunk, past_key_values=probe_cache,
                            use_cache=True)
            print(f"[bench-direct] k={probe_k} chunk forward OK — "
                  f"last_hidden_state shape="
                  f"{tuple(out2.last_hidden_state.shape)}")
    except Exception as e:
        print(f"[bench-direct] ERROR during direct backbone probe: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        print("[bench-direct] The backbone does not accept multi-token forward "
              "in the expected HF shape. Dumping backbone signature:",
              file=sys.stderr)
        import inspect
        try:
            print(inspect.signature(backbone.forward))
        except Exception:
            pass
        return 2
    sync(device)
    print()

    def run_chunk_size(k: int) -> dict:
        per_chunk_s: list[float] = []

        # Warmup with one full streaming run at this k.
        for _ in range(args.n_warmup):
            cache = cache_ctor()
            with torch.inference_mode():
                out = backbone(input_ids=uid_tensor, past_key_values=cache,
                               use_cache=True)
                cache = out.past_key_values
                for i in range(0, min(k * 4, len(asst_ids)), k):
                    chunk_ids = asst_ids[i:i + k]
                    if len(chunk_ids) != k:
                        break
                    chunk = torch.tensor([chunk_ids], dtype=torch.long,
                                         device=device)
                    out = backbone(input_ids=chunk, past_key_values=cache,
                                   use_cache=True)
                    cache = out.past_key_values
        sync(device)

        # Timed runs.
        for _ in range(args.n_prompts):
            cache = cache_ctor()
            with torch.inference_mode():
                out = backbone(input_ids=uid_tensor, past_key_values=cache,
                               use_cache=True)
                cache = out.past_key_values
                for i in range(0, len(asst_ids), k):
                    chunk_ids = asst_ids[i:i + k]
                    if len(chunk_ids) != k:
                        break
                    chunk = torch.tensor([chunk_ids], dtype=torch.long,
                                         device=device)
                    sync(device)
                    t0 = time.perf_counter()
                    out = backbone(input_ids=chunk, past_key_values=cache,
                                   use_cache=True)
                    sync(device)
                    per_chunk_s.append(time.perf_counter() - t0)
                    cache = out.past_key_values

        ms_sorted = sorted(s * 1000.0 for s in per_chunk_s)
        n = len(ms_sorted)
        return {
            "k": k, "n_chunks": n,
            "p50_ms": pct(ms_sorted, 50),
            "p95_ms": pct(ms_sorted, 95),
            "p99_ms": pct(ms_sorted, 99),
            "mean_ms": statistics.fmean(ms_sorted) if ms_sorted else 0.0,
        }

    rows = [run_chunk_size(k) for k in args.chunks]

    print("[bench-direct] --- chunk-size sweep (backbone-forward, "
          "no verdict readout inside timed region) ---")
    print(f"{'k':>4} {'n_chunks':>9} "
          f"{'chunk P50 ms':>13} {'chunk P99 ms':>13} "
          f"{'eff/tok P50':>12} {'eff/tok P99':>12} "
          f"{'T2 (<8 ms)':>11}  speedup_vs_k1")
    p99_k1 = next((r["p99_ms"] for r in rows if r["k"] == 1),
                  rows[0]["p99_ms"] if rows else 0.0)
    for r in rows:
        eff_p50 = r["p50_ms"] / r["k"]
        eff_p99 = r["p99_ms"] / r["k"]
        t2 = "yes" if eff_p99 < 8.0 else "no"
        speedup = (p99_k1 / eff_p99) if eff_p99 > 0 else 0.0
        print(f"{r['k']:>4} {r['n_chunks']:>9} "
              f"{r['p50_ms']:>10.2f} ms {r['p99_ms']:>10.2f} ms "
              f"{eff_p50:>9.2f} ms {eff_p99:>9.2f} ms "
              f"{t2:>11}  {speedup:>6.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
