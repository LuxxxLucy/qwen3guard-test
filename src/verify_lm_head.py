"""Gate: backend last-position verdict logits == full-sequence fp32 reference.

fp32 backends must match within FP32_ATOL (the slice is value-preserving).
fp16/quantized backends only checked against gross corruption (QUANT_SANITY).
Exit non-zero on failure.

Usage: uv run python src/verify_lm_head.py --runtime pytorch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bench_common import load_representative_texts
from contract import DEFAULT_PRECISION, RUNTIMES, TEMPLATES
from gen_backends import make_backend
from gen_common import (VERDICT_LABELS, build_forced_ids,
                        discover_verdict_token_ids)

FP32_ATOL = 0.05      # fp32 must reproduce reference logits to fp noise
QUANT_SANITY = 0.66   # 3-way classifier well above 1/3 random rate


def reference_logits(model_id: str, samples: list[list[int]],
                     verdict_token_ids: list[int]) -> list[list[float]]:
    """Cached full-sequence fp32 reference forwards. One per (model, samples,
    verdict_ids) hash; reused across backend gate runs."""
    key = hashlib.sha1(
        (model_id + json.dumps(samples) + str(verdict_token_ids)).encode()
    ).hexdigest()[:16]
    # Cache outside results/ so summarize_cpu's glob doesn't see it.
    cache = Path("results/.cache") / f"lm_head_ref_{key}.json"
    if cache.exists():
        print(f"[verify-lm-head] reference: cached ({cache.name})", flush=True)
        return json.loads(cache.read_text())

    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32).to("cpu").eval()
    out = []
    with torch.no_grad():
        for ids in samples:
            t = torch.tensor([ids], dtype=torch.long)
            logits = model(input_ids=t, attention_mask=torch.ones_like(t),
                           use_cache=False).logits[0, -1]
            out.append([float(logits[v]) for v in verdict_token_ids])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(cache)  # atomic — two gate runs may race on this file
    print(f"[verify-lm-head] reference: computed and cached ({cache.name})",
          flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=list(RUNTIMES))
    ap.add_argument("--precision", default=None)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--template", default=TEMPLATES[1], choices=list(TEMPLATES))
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    precision = args.precision or DEFAULT_PRECISION[args.runtime]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    verdict_ids = discover_verdict_token_ids(tok, args.template)
    vids = [verdict_ids[v] for v in VERDICT_LABELS]
    texts = load_representative_texts(max_samples=args.n_samples, tokenizer=tok)
    samples = [build_forced_ids(tok, t, args.template) for t in texts]
    print(f"[verify-lm-head] {args.runtime}/{precision} "
          f"template={args.template} n={len(samples)}", flush=True)

    ref = reference_logits(args.model_id, samples, vids)

    # Request last-position projection on pytorch — the gate's whole job.
    backend = make_backend(args.runtime, precision, vids, args.threads,
                           last_pos_logits=(args.runtime == RUNTIMES[0]))
    backend.load(args.model_id, args.artifact, max(len(s) for s in samples))

    max_logit_diff = 0.0
    argmax_match = 0
    for ids, ref_row in zip(samples, ref):
        got = backend.verdict_logits(ids)
        max_logit_diff = max(max_logit_diff,
                             max(abs(a - b) for a, b in zip(got, ref_row)))
        if got.index(max(got)) == ref_row.index(max(ref_row)):
            argmax_match += 1

    n = len(samples)
    strict_fp32 = precision == "fp32"
    if strict_fp32:
        ok = argmax_match == n and max_logit_diff <= FP32_ATOL
        gate = f"fp32: argmax {n}/{n} and max logit diff <= {FP32_ATOL}"
    else:
        # fp16/quantized: gate only against gross corruption; weight-precision
        # drift vs fp32 is expected and not a slice error.
        ok = argmax_match >= QUANT_SANITY * n
        gate = f"argmax >= {QUANT_SANITY:.0%} (corruption tripwire)"
    print(f"[verify-lm-head] verdict argmax match vs reference: {argmax_match}/{n}")
    print(f"[verify-lm-head] max verdict-logit diff vs reference: {max_logit_diff:.4g}")
    print(f"[verify-lm-head] gate — {gate}")
    print(f"[verify-lm-head] {args.runtime}/{precision}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
