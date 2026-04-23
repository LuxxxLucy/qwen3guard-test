"""Direct-path Qwen3Guard-Stream bench with classification heads inside the
timed region, plus cross-path verdict equivalence checks.

The bench in bench_stream_direct.py timed `backbone(...)` only; the cost of
the two classification heads was excluded. This script instead calls the
full `Qwen3ForGuardModel.forward(...)` inside the timed region — the heads
are ~1 MB of params on 0.6B and add a small cost; the numbers below are
the production-realistic latency.

Equivalence checks:
  (a) direct per-token (k=1)  vs  direct chunked (k=K): must match at
      every position — a causal transformer with a correct KV cache
      produces the same per-position hidden state whether a chunk arrives
      in k steps or one step, so the classification heads must produce
      the same argmax.
  (b) direct per-token (k=1)  vs  stock API stream_moderate_from_ids
      (k=1): diagnostic. If this also matches, the stock API is
      semantically equivalent — just slow. If it disagrees, the shipped
      stream_generate loop (which passes the full accumulated sequence
      with a growing cache) behaves differently from the paper's stated
      "per-token moderation without reprocessing prior tokens."

Latency sweep: same k ∈ {1, 2, 4, 8, 16, 32, 64} as bench_stream_direct,
but the timed region includes both heads.

Run:
  uv run python src/bench_stream_direct_heads.py --model-id Qwen/Qwen3Guard-Stream-0.6B
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
    ap.add_argument("--equivalence-chunk", type=int, default=16,
                    help="Chunk size K used for the direct-chunked vs "
                         "direct-per-token equivalence check.")
    ap.add_argument("--equivalence-n-tokens", type=int, default=64,
                    help="Number of assistant tokens to run through the "
                         "equivalence check (kept small because per-token "
                         "stock-API comparison is slow).")
    ap.add_argument("--n-assistant-tokens", type=int, default=128,
                    help="Total assistant tokens streamed in the latency sweep.")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--user-len", type=int, default=81)
    ap.add_argument("--skip-stock-api", action="store_true",
                    help="Skip the stock-API equivalence path (faster).")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[direct-heads] model={args.model_id} device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id, dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    # Sanity: the top-level model must expose the Guard forward + heads +
    # response_risk_level_map. These are defined on Qwen3ForGuardModel in
    # the cached modeling source.
    for required in ("risk_level_head", "category_head",
                     "response_risk_level_map", "response_category_map"):
        if not hasattr(model, required):
            print(f"[direct-heads] ERROR: model is missing `{required}` — "
                  f"unexpected class {type(model).__name__}", file=sys.stderr)
            return 1
    print(f"[direct-heads] model class = {type(model).__name__}")
    print(f"[direct-heads] backbone.layers = "
          f"{len(model.model.layers)}")
    print(f"[direct-heads] response_risk_level_map = "
          f"{dict(model.response_risk_level_map)}")

    # Build a stable user prompt of the requested length.
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
    print(f"[direct-heads] prefill token count = {len(uid)}")

    asst_text = (
        "I understand your question. Here is a detailed response that tries to "
        "be helpful while staying within safe guidelines. Let me walk through "
        "the reasoning step by step so that the answer is clear and well "
        "structured. Each suggestion builds on the previous one, and together "
        "they should give you a concrete, actionable plan going forward now."
    )
    asst_ids_all = tok.encode(asst_text, add_special_tokens=False)
    while len(asst_ids_all) < max(args.n_assistant_tokens,
                                  args.equivalence_n_tokens):
        asst_ids_all = asst_ids_all + asst_ids_all
    asst_ids = asst_ids_all[: args.n_assistant_tokens]
    eq_asst_ids = asst_ids_all[: args.equivalence_n_tokens]

    uid_tensor = torch.tensor([uid], dtype=torch.long, device=device)

    # ---- Causality check --------------------------------------------------
    # The bench's correctness argument rests on: verdict at asst position i
    # depends only on tokens at positions <= i. In a causal transformer
    # this is true by construction (causal attention mask + pointwise heads)
    # — this function PROVES it empirically on the loaded checkpoint:
    #
    #   Run A: forward [asst_0..asst_{long-1}]   -> risk logits at all long positions
    #   Run B: forward [asst_0..asst_{short-1}]  -> risk logits at all short positions
    #
    # Positions 0..short-1 in Run A were computed with Run A's later tokens
    # in attention range; positions 0..short-1 in Run B did not see those
    # later tokens. If causality holds, the logits at positions 0..short-1
    # must match between A and B (up to fp16 numerical noise).
    def causality_check(asst_tokens_long: list[int], short_len: int) -> dict:
        long_len = len(asst_tokens_long)
        assert short_len < long_len, "need short_len < long_len for a meaningful test"

        def run_one(asst_ids_slice: list[int]):
            cache = DynamicCache()
            with torch.inference_mode():
                model.forward(input_ids=uid_tensor, past_key_values=cache,
                              use_cache=True)
                chunk = torch.tensor([asst_ids_slice], dtype=torch.long,
                                     device=device)
                out = model.forward(input_ids=chunk, past_key_values=cache,
                                    use_cache=True)
            return out.risk_level_logits[0]  # [seq_len, num_risk]

        logits_A = run_one(asst_tokens_long)
        logits_B = run_one(asst_tokens_long[:short_len])

        # Verdict equality at shared positions.
        argmax_A = logits_A[:short_len].argmax(dim=-1).tolist()
        argmax_B = logits_B.argmax(dim=-1).tolist()
        verdict_matches = sum(a == b for a, b in zip(argmax_A, argmax_B))

        # Logit numerical closeness — softer bar than exact equality because
        # SDPA/attention kernels may reorder reductions across sequence length
        # in bf16.
        diff = (logits_A[:short_len].float() - logits_B.float()).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())

        return {
            "short_len": short_len, "long_len": long_len,
            "verdict_matches": verdict_matches,
            "max_abs_logit_diff": max_abs,
            "mean_abs_logit_diff": mean_abs,
            "argmax_A": argmax_A, "argmax_B": argmax_B,
        }

    # ---- Equivalence check helpers ----------------------------------------
    def direct_verdicts(asst_tokens: list[int], k: int) -> list[str]:
        """Run direct path with chunk size k, return per-asst-token risk-level strings."""
        cache = DynamicCache()
        with torch.inference_mode():
            model.forward(input_ids=uid_tensor, past_key_values=cache,
                          use_cache=True)
        verdicts: list[str] = []
        with torch.inference_mode():
            for i in range(0, len(asst_tokens), k):
                chunk_ids = asst_tokens[i:i + k]
                chunk = torch.tensor([chunk_ids], dtype=torch.long,
                                     device=device)
                out = model.forward(input_ids=chunk, past_key_values=cache,
                                    use_cache=True)
                cache = out.past_key_values
                argmax = out.risk_level_logits.argmax(dim=-1)[0].tolist()
                if isinstance(argmax, int):
                    argmax = [argmax]
                verdicts.extend(
                    model.response_risk_level_map[int(a)] for a in argmax)
        return verdicts

    def stock_verdicts(asst_tokens: list[int]) -> list[str]:
        """Run the stock stream_moderate_from_ids API per-token."""
        # Prime with user turn.
        uid_1d = torch.tensor(uid, dtype=torch.long, device=device)
        _, state = model.stream_moderate_from_ids(
            uid_1d, role="user", stream_state=None)
        verdicts: list[str] = []
        for tid in asst_tokens:
            tid_t = torch.tensor(tid, dtype=torch.long, device=device)
            result, state = model.stream_moderate_from_ids(
                tid_t, role="assistant", stream_state=state)
            # result['risk_level'] is a list of all accumulated positions;
            # the last element is the newly processed asst token.
            verdicts.append(result["risk_level"][-1])
        if hasattr(model, "close_stream"):
            try:
                model.close_stream(state)
            except Exception:
                pass
        return verdicts

    # Warmup.
    direct_verdicts(eq_asst_ids[:4], k=1)
    sync(device)

    print()
    print("=" * 68)
    print("[direct-heads] CAUSALITY CHECK")
    print("[direct-heads] verifies: verdict at asst position i does NOT")
    print("[direct-heads] depend on tokens at positions > i (i.e. the model")
    print("[direct-heads] is genuinely causal and chunking is safe).")
    print("=" * 68)
    # Use a context where long_len > short_len by a comfortable margin so
    # the test is meaningful: short=16 positions vs long=64 positions means
    # positions 0..15 in Run A had 48 future tokens in attention range, and
    # they must still match Run B which saw no future tokens.
    caus = causality_check(eq_asst_ids, short_len=16)
    verdict_status = ("PASS" if caus["verdict_matches"] == caus["short_len"]
                      else "FAIL")
    print(f"[direct-heads] short={caus['short_len']} vs long={caus['long_len']}: "
          f"{caus['verdict_matches']}/{caus['short_len']} verdicts match  "
          f"[{verdict_status}]")
    print(f"[direct-heads] logits (shared positions, bf16): "
          f"max |diff| = {caus['max_abs_logit_diff']:.4e}  "
          f"mean |diff| = {caus['mean_abs_logit_diff']:.4e}")
    if caus["verdict_matches"] != caus["short_len"]:
        print(f"[direct-heads] *** CAUSALITY VIOLATED *** "
              f"— chunking is unsafe on this checkpoint.")
        print(f"[direct-heads]   A (long={caus['long_len']}) argmaxes: "
              f"{caus['argmax_A']}")
        print(f"[direct-heads]   B (short={caus['short_len']}) argmaxes: "
              f"{caus['argmax_B']}")

    print()
    print("=" * 68)
    print("[direct-heads] EQUIVALENCE CHECK")
    print(f"[direct-heads] running over {args.equivalence_n_tokens} asst tokens")
    print("=" * 68)

    t0 = time.perf_counter()
    v_direct_1 = direct_verdicts(eq_asst_ids, k=1)
    t_direct_1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    v_direct_K = direct_verdicts(eq_asst_ids, k=args.equivalence_chunk)
    t_direct_K = time.perf_counter() - t0

    agree_direct = sum(a == b for a, b in zip(v_direct_1, v_direct_K))
    print(f"[direct-heads] direct k=1  ({t_direct_1*1000:7.1f} ms) "
          f"vs direct k={args.equivalence_chunk} ({t_direct_K*1000:7.1f} ms): "
          f"{agree_direct}/{len(v_direct_1)} positions agree")
    if agree_direct < len(v_direct_1):
        print(f"[direct-heads] first 10 positions:")
        for i in range(min(10, len(v_direct_1))):
            mark = " " if v_direct_1[i] == v_direct_K[i] else "*"
            print(f"  {mark} pos {i:3d}  k=1={v_direct_1[i]:14s}  "
                  f"k={args.equivalence_chunk}={v_direct_K[i]}")

    if not args.skip_stock_api:
        t0 = time.perf_counter()
        v_stock = stock_verdicts(eq_asst_ids)
        t_stock = time.perf_counter() - t0
        agree_stock = sum(a == b for a, b in zip(v_direct_1, v_stock))
        print(f"[direct-heads] direct k=1  vs stock-API k=1 "
              f"({t_stock*1000:7.1f} ms): "
              f"{agree_stock}/{len(v_direct_1)} positions agree")
        if agree_stock < len(v_direct_1):
            print(f"[direct-heads] first 10 disagreements:")
            shown = 0
            for i in range(len(v_direct_1)):
                if v_direct_1[i] != v_stock[i]:
                    print(f"  * pos {i:3d}  direct={v_direct_1[i]:14s}  "
                          f"stock={v_stock[i]}")
                    shown += 1
                    if shown >= 10:
                        break

    # ---- Latency sweep with heads in timed region -------------------------
    def run_chunk_size(k: int) -> dict:
        per_chunk_s: list[float] = []

        # Warmup at this k.
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

    print()
    print("=" * 68)
    print("[direct-heads] LATENCY SWEEP "
          "(direct path, heads included in timed region)")
    print("=" * 68)

    rows = [run_chunk_size(k) for k in args.chunks]

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
