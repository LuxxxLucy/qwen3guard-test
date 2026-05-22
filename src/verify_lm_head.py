"""Correctness gate for the verdict-logit computation.

The L2 forced-prefix path reads only the last position's logits. The backends
restrict the output projection (the hidden->vocabulary matmul) to that last
position, skipping ~199/200 of a 1024x151,936 matmul over the prompt. That
restriction must not change the result.

This check, for the representative inputs, recomputes the 3 verdict logits with
a full-sequence PyTorch fp32 forward (the reference, independent of the
backends) and compares them to the backend under test. pytorch and onnx fp32
must reproduce the reference logits within tolerance — that proves the slice is
value-preserving. OpenVINO's "fp32" export keeps fp16 weights (optimum-intel
default), and quantized backends drift for the int8/int4 accuracy tradeoff;
both sit a little off the fp32 reference for a weight-precision reason, not a
slice error, so they are gated only against gross corruption. Exit non-zero on
failure — this is a gate.

Usage:
  uv run python src/verify_lm_head.py --runtime pytorch
  uv run python src/verify_lm_head.py --runtime onnx --precision fp32 \\
      --artifact onnx_models/Qwen3Guard-Gen-0.6B/fp32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bench_common import load_representative_texts
from gen_backends import DEFAULT_PRECISION, make_backend
from gen_common import (TEMPLATES, VERDICT_LABELS, build_forced_ids,
                        discover_verdict_token_ids)

# An fp32 backend should reproduce the reference verdict logits to fp noise —
# any larger gap means the slice changed the result.
FP32_ATOL = 0.05
# Quantized backends flip some borderline verdicts vs fp32 (expected). The gate
# only catches a corrupted graph: a working 3-way classifier sits far above the
# 1/3 random rate.
QUANT_SANITY = 0.66


def reference_logits(model_id: str, samples: list[list[int]],
                     verdict_token_ids: list[int]) -> list[list[float]]:
    """Ground truth: a full-sequence fp32 forward, last-position verdict logits.
    Computed straight from the HF model so it is unaffected by any backend.
    Cached under results/ — the reference is identical for every backend, so
    repeated gate runs reuse it instead of recomputing 100 forwards."""
    key = hashlib.sha1(
        (model_id + json.dumps(samples) + str(verdict_token_ids)).encode()
    ).hexdigest()[:16]
    cache = Path("results") / f".lm_head_ref_{key}.json"
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
    ap.add_argument("--runtime", required=True,
                    choices=["pytorch", "onnx", "openvino", "llamacpp"])
    ap.add_argument("--precision", default=None)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--template", default="test-200", choices=list(TEMPLATES))
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

    # pytorch backend defaults to the full-sequence projection now; the gate's
    # job is to verify the last-position projection, so request it explicitly.
    backend = make_backend(args.runtime, precision, vids, args.threads,
                           last_pos_logits=(args.runtime == "pytorch"))
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
    # The strict logit gate only isolates the slice where the backend truly
    # reproduces fp32: pytorch and onnx fp32. OpenVINO's "fp32" export keeps
    # fp16 weights (optimum-intel default), so its verdict logits sit ~0.6 off
    # a true-fp32 reference — a weight-precision gap, not a slice error.
    strict_fp32 = precision == "fp32" and args.runtime != "openvino"
    if strict_fp32:
        # fp32 has no quantization noise — the sliced output must reproduce the
        # full-sequence reference exactly. This is the lm_head-slice proof.
        ok = argmax_match == n and max_logit_diff <= FP32_ATOL
        gate = f"fp32: argmax {n}/{n} and max logit diff <= {FP32_ATOL}"
    else:
        # Quantized weights — and OpenVINO's fp16-stored "fp32" — flip some
        # borderline verdicts vs the fp32 reference. A weight-precision effect,
        # not a slice error (the slice is weight-independent and proven strict
        # at pytorch/onnx fp32). Gate only against gross corruption.
        ok = argmax_match >= QUANT_SANITY * n
        why = ("OpenVINO stores fp32-export weights as fp16"
               if precision == "fp32" else "quantized weights drift vs fp32")
        gate = (f"argmax >= {QUANT_SANITY:.0%} corruption tripwire "
                f"({why} — sub-100% drift vs the fp32 reference is expected)")
    print(f"[verify-lm-head] verdict argmax match vs reference: {argmax_match}/{n}")
    print(f"[verify-lm-head] max verdict-logit diff vs reference: {max_logit_diff:.4g}")
    print(f"[verify-lm-head] gate — {gate}")
    print(f"[verify-lm-head] {args.runtime}/{precision}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
