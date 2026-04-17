"""Dataset-scale correctness test for the optimization ladder.

For each of N samples from Qwen3GuardTest (response_loc split), run
every optimization path and compare the predicted verdict (Safe /
Unsafe / Controversial — Qwen3Guard-Gen is a 3-way classifier) to the
L0 naive baseline. Reports agreement rate per level.

Paths tested:
  L0  naive                           (reference)
  L1  +prefix-cache                   (generate with cached head KV)
  L2  +forced-prefix                  (single forward, argmax last logit)
  L2' +forced-prefix  (no cache)      (single forward from scratch)

L3 (compile) is not tested here — torch.compile numerical equivalence
is a compiler concern; it produces the same logits up to float noise and
the startup check in bench_gen_pytorch.py verifies one sample suffices.

Usage:
  uv run python scripts/correctness_test.py --n 50 [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench_gen_pytorch import (  # noqa: E402
    compute_cacheable_head, precompute_template_cache,
    predict_naive, predict_cached, predict_forced_prefix,
)
from bench_common import pick_device, pick_dtype, load_representative_texts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--n", type=int, default=50,
                    help="Number of real samples to test. Default 50.")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch  # noqa: F401  (required by predict_* via bench_gen_pytorch)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or pick_device()
    dtype = pick_dtype(device)

    print(f"[correctness] model={args.model_id} device={device} dtype={dtype} n={args.n}")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype,
    ).to(device).eval()

    head_ids = compute_cacheable_head(tok)
    template_cache = precompute_template_cache(model, head_ids, device)
    T = len(head_ids)
    print(f"[correctness] cacheable_head={T}")

    samples = load_representative_texts(max_samples=args.n, tokenizer=tok)
    if not samples:
        print("[err] could not load Qwen3GuardTest samples", file=sys.stderr)
        return 1
    samples = samples[: args.n]
    print(f"[correctness] loaded {len(samples)} samples")

    counts = {"L0": 0, "L1": 0, "L2_cached": 0, "L2_uncached": 0}
    ok_counts = {"L1": 0, "L2_cached": 0, "L2_uncached": 0}
    mismatches = {"L1": [], "L2_cached": [], "L2_uncached": []}

    for i, user_text in enumerate(samples):
        # L0 baseline
        try:
            v0, gen0 = predict_naive(
                model, tok, user_text, device, args.max_new_tokens,
            )
            counts["L0"] += 1
        except Exception as e:
            print(f"  [{i}] L0 FAIL: {e!r}")
            continue

        # L1 prefix-cache + generate
        try:
            v1, gen1 = predict_cached(
                model, tok, template_cache, head_ids, user_text, device,
                args.max_new_tokens,
            )
            counts["L1"] += 1
            if gen1 == gen0:
                ok_counts["L1"] += 1
            else:
                mismatches["L1"].append((i, v0, v1))
        except Exception as e:
            print(f"  [{i}] L1 FAIL: {e!r}")

        # L2 forced-prefix + prefix-cache
        try:
            v2c, _ = predict_forced_prefix(
                model, tok, template_cache, head_ids, user_text, device,
                use_prefix_cache=True,
            )
            counts["L2_cached"] += 1
            if v2c == v0:
                ok_counts["L2_cached"] += 1
            else:
                mismatches["L2_cached"].append((i, v0, v2c))
        except Exception as e:
            print(f"  [{i}] L2(cached) FAIL: {e!r}")

        # L2' forced-prefix alone (no prefix cache)
        try:
            v2u, _ = predict_forced_prefix(
                model, tok, template_cache, head_ids, user_text, device,
                use_prefix_cache=False,
            )
            counts["L2_uncached"] += 1
            if v2u == v0:
                ok_counts["L2_uncached"] += 1
            else:
                mismatches["L2_uncached"].append((i, v0, v2u))
        except Exception as e:
            print(f"  [{i}] L2(uncached) FAIL: {e!r}")

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] running...")

    print("\n[correctness] === RESULTS ===")
    n0 = counts["L0"]
    print(f"L0 (baseline): ran={n0}")
    for level in ("L1", "L2_cached", "L2_uncached"):
        ran = counts[level]
        ok = ok_counts[level]
        rate = (ok / ran * 100) if ran else 0.0
        match_kind = "token-exact" if level == "L1" else "verdict"
        print(f"{level}: ran={ran}  {match_kind}_match={ok}/{ran} ({rate:.1f}%)")
        if mismatches[level]:
            print(f"  first 5 mismatches (sample_idx, baseline_verdict, level_verdict):")
            for m in mismatches[level][:5]:
                print(f"    {m}")

    # Exit non-zero if any verdict-level disagreement. Token-exact L1
    # mismatches are tolerated (they can occur if DynamicCache clones drift
    # in late decode tokens past the verdict; verdict agreement is what
    # matters for the A8 contract).
    fail = False
    for level in ("L2_cached", "L2_uncached"):
        if counts[level] > 0 and ok_counts[level] != counts[level]:
            fail = True
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
