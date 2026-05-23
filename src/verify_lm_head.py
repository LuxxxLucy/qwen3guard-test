"""Cross-impl verdict-logit equivalence gate.

For one backend (or one pre-dumped rust logits JSON), runs the L1 forced-prefix
forward and compares the 3 verdict logits against a cached PyTorch CPU fp32
reference on N samples per template.

Gate:
  - fp32 precision: max abs logit diff <= FP32_ATOL (1e-2) AND argmax == n.
  - fp16/quant:     argmax >= QUANT_ARGMAX_FRAC * n (corruption tripwire).
                    Per-logit drift up to O(1) is expected from weight quant
                    on borderline 3-way classifier samples; we report it but
                    don't fail on it.

Exit non-zero on drift. Run from scripts/run_gen_cpu.sh once per backend.

Usage:
  uv run python src/verify_lm_head.py --runtime pytorch
  uv run python src/verify_lm_head.py --runtime onnx --precision fp32 \\
      --artifact onnx_models/Qwen3Guard-Gen-0.6B/fp32
  uv run python src/verify_lm_head.py --runtime rust-candle \\
      --logits-json rust/verify_logits.json --rust-inputs rust/bench_inputs.json
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

FP32_ATOL = 1e-2          # fp32 reproduce reference logits within fp noise
QUANT_ARGMAX_FRAC = 0.8   # fp16/quant: ≥80% argmax match (tripwire only)
N_SAMPLES = 10            # per template; verify is a gate, not a benchmark


def reference_logits(model_id: str, samples: list[list[int]],
                     verdict_token_ids: list[int]) -> list[list[float]]:
    """Cached full-sequence fp32 reference forwards. Keyed on (model, samples,
    verdict_ids); reused across backend gate runs."""
    key = hashlib.sha1(
        (model_id + json.dumps(samples) + str(verdict_token_ids)).encode()
    ).hexdigest()[:16]
    # Cache outside results/ so summarize_cpu's glob doesn't see it.
    cache = Path("results/.cache") / f"lm_head_ref_{key}.json"
    if cache.exists():
        print(f"[verify] reference: cached ({cache.name})", flush=True)
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
    print(f"[verify] reference: computed and cached ({cache.name})", flush=True)
    return out


def gather_backend_logits(args, vids: list[int], samples_by_tmpl: dict
                          ) -> dict[str, list[list[float]]]:
    """Run the backend on each template's samples; return verdict logits."""
    max_seq_len = max(len(s) for pool in samples_by_tmpl.values() for s in pool)
    # last_pos_logits=True is value-preserving on fp32 and matches what the
    # other (L2-baked) backends export. Cosmetic on argmax, honest on diff.
    backend = make_backend(args.runtime, args.precision, vids, args.threads,
                           last_pos_logits=(args.runtime == "pytorch"))
    backend.load(args.model_id, args.artifact, max_seq_len)
    return {t: [backend.verdict_logits(ids) for ids in samples]
            for t, samples in samples_by_tmpl.items()}


def load_rust_logits(args) -> tuple[dict, dict[str, list[list[int]]]]:
    """Load rust verdict-logit dump + the forced_ids it was computed on
    (the rust input bundle's first N_SAMPLES per template)."""
    logits = json.loads(Path(args.logits_json).read_text())
    rust_inputs = json.loads(Path(args.rust_inputs).read_text())
    samples_by_tmpl = {
        t: rust_inputs["templates"][t]["forced"][:N_SAMPLES] for t in TEMPLATES
    }
    return logits, samples_by_tmpl


def compare(got: list[list[float]], ref: list[list[float]], strict: bool
            ) -> tuple[float, int, int]:
    """Return (max_logit_diff, argmax_match, n)."""
    n = len(got)
    max_diff = 0.0
    argmax_match = 0
    for g, r in zip(got, ref):
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(g, r)))
        if g.index(max(g)) == r.index(max(r)):
            argmax_match += 1
    return max_diff, argmax_match, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True,
                    choices=[*RUNTIMES, "rust-candle"])
    ap.add_argument("--precision", default=None)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--logits-json", default=None,
                    help="rust-candle: pre-dumped verdict logits JSON.")
    ap.add_argument("--rust-inputs", default="rust/bench_inputs.json",
                    help="rust-candle: input bundle (for the forced_ids).")
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    precision = args.precision or (
        "fp32" if args.runtime == "rust-candle"
        else DEFAULT_PRECISION[args.runtime])

    if args.runtime == "vllm-cpu":
        # vLLM exposes a sampled token, not verdict logits — skip the gate
        # (its argmax is already covered by --verify in bench_gen_cpu.py).
        print("[verify] vllm-cpu: no verdict_logits surface; skipping.")
        return 0

    # Build per-template forced_ids + verdict token ids.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    vids = [discover_verdict_token_ids(tok, TEMPLATES[0])[v]
            for v in VERDICT_LABELS]

    if args.runtime == "rust-candle":
        if not args.logits_json:
            raise SystemExit("--logits-json is required for rust-candle.")
        got_by_tmpl, samples_by_tmpl = load_rust_logits(args)
    else:
        texts = load_representative_texts(max_samples=N_SAMPLES, tokenizer=tok)
        samples_by_tmpl = {t: [build_forced_ids(tok, x, t) for x in texts]
                           for t in TEMPLATES}
        got_by_tmpl = gather_backend_logits(args, vids, samples_by_tmpl)

    strict = precision == "fp32"
    gate = (f"fp32 atol={FP32_ATOL:.0e} AND argmax==n" if strict
            else f"argmax/n>={QUANT_ARGMAX_FRAC:.0%} (quant tripwire)")
    print(f"[verify] {args.runtime}/{precision} n={N_SAMPLES} gate: {gate}",
          flush=True)

    all_ok = True
    for tmpl in TEMPLATES:
        if tmpl not in got_by_tmpl:
            print(f"[verify] {tmpl}: missing in backend output — FAIL")
            all_ok = False
            continue
        ref = reference_logits(args.model_id, samples_by_tmpl[tmpl], vids)
        got = got_by_tmpl[tmpl]
        max_diff, argmax_match, n = compare(got, ref, strict)
        if strict:
            ok = argmax_match == n and max_diff <= FP32_ATOL
        else:
            ok = argmax_match >= QUANT_ARGMAX_FRAC * n
        status = "PASS" if ok else "FAIL"
        print(f"[verify] {tmpl:9s} max_diff={max_diff:.3e} "
              f"argmax={argmax_match}/{n} {status}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
