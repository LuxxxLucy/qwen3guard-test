"""Cross-impl trick-correctness gate.

For one (runtime, precision, opt_level, artifact) tuple, runs the path and
compares the resulting verdicts against a PyTorch CPU fp32 L0 reference on
N samples per template. L0 is the canonical model-card path; every other
trick (L1 forced-prefix, L2 lastpos lm_head, L3 prefix-KV cache) must
preserve the same verdict on fp32.

Gates:
  - fp32 path: every sample's verdict must match the L0 reference.
                For L1/L2/L3 we additionally require max abs logit diff
                <= FP32_ATOL vs the L1 fp32 logit reference (tight).
  - fp16/quant path: argmax/n >= QUANT_ARGMAX_FRAC (tripwire only;
                weight quant drifts logits by O(1) on borderline samples).

Exit non-zero on drift. Run once per row of the CPU_GEN_REPORT table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bench_common import load_representative_texts
from contract import DEFAULT_PRECISION, OPT_LEVELS, RUNTIMES, TEMPLATES
from gen_backends import make_backend
from gen_common import (VERDICT_LABELS, build_forced_ids, build_plain_ids,
                        common_prefix, discover_verdict_token_ids)

FP32_ATOL = 1e-2          # fp32 reproduce reference logits within fp noise
QUANT_ARGMAX_FRAC = 0.8   # fp16/quant: >=80% argmax match (tripwire only)
N_SAMPLES = 10            # per template; verify is a gate, not a benchmark


def _cache_path(model_id: str, samples: list[list[int]], vids: list[int],
                kind: str) -> Path:
    key = hashlib.sha1(
        (model_id + json.dumps(samples) + str(vids) + kind).encode()
    ).hexdigest()[:16]
    return Path("results/.cache") / f"{kind}_{key}.json"


def reference_logits(model_id: str, samples: list[list[int]],
                     vids: list[int]) -> list[list[float]]:
    """PyTorch fp32 last-position verdict logits over forced_ids. Cached."""
    cache = _cache_path(model_id, samples, vids, "lm_head_ref")
    if cache.exists():
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
            out.append([float(logits[v]) for v in vids])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(cache)
    return out


def reference_l0_verdicts(model_id: str, plain_samples: list[list[int]],
                          vids: list[int]) -> list[int]:
    """PyTorch fp32 L0 decode-loop verdict per sample. Cached. The user-facing
    canonical reference -- every trick on every fp32 backend must reproduce
    these verdicts on every sample."""
    cache = _cache_path(model_id, plain_samples, vids, "l0_ref")
    if cache.exists():
        return json.loads(cache.read_text())
    from gen_backends import PyTorchCPUBackend
    ref = PyTorchCPUBackend("fp32", vids, threads=None, unoptimized=True)
    ref.load(model_id, None, max(len(p) for p in plain_samples))
    out = [ref.predict(p) for p in plain_samples]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(cache)
    return out


def run_backend(args, vids: list[int], forced_by_tmpl: dict,
                plain_by_tmpl: dict, opt_level: str
                ) -> dict[str, tuple[list[int], list[list[float]] | None]]:
    """Run the backend at the requested opt_level. Returns
    {template: (argmax_verdicts, logits_or_None)} per template."""
    max_seq_len = max(len(s) for pool in forced_by_tmpl.values() for s in pool)
    max_seq_len = max(max_seq_len,
                      max(len(s) for pool in plain_by_tmpl.values() for s in pool))
    backend = make_backend(
        args.runtime, args.precision, vids, args.threads,
        kv_cache=(opt_level == "L3"),
        unoptimized=(opt_level == "L0"),
        last_pos_logits=(opt_level == "L2"),
    )
    backend.load(args.model_id, args.artifact, max_seq_len)

    out: dict = {}
    for tmpl in TEMPLATES:
        forced = forced_by_tmpl[tmpl]
        plain = plain_by_tmpl[tmpl]
        if opt_level == "L0":
            verdicts = [backend.predict(p) for p in plain]
            out[tmpl] = (verdicts, None)
        else:
            if opt_level == "L3":
                prefix = common_prefix(forced)
                backend.prime_prefix(prefix)
            logits = [backend.verdict_logits(f) for f in forced]
            verdicts = [lg.index(max(lg)) for lg in logits]
            out[tmpl] = (verdicts, logits)
    return out


def load_rust_dump(args, opt_level: str
                   ) -> tuple[dict, dict[str, list[list[int]]],
                              dict[str, list[list[int]]]]:
    """Load rust verify dump + the forced/plain inputs it was computed on.
    Rust dump shape: {template: {"L0": [v...], "L1": [[lg...]...], "L3": [[lg...]...]}}."""
    rust_inputs = json.loads(Path(args.rust_inputs).read_text())
    forced_by_tmpl = {t: rust_inputs["templates"][t]["forced"][:N_SAMPLES]
                      for t in TEMPLATES}
    plain_by_tmpl = {t: rust_inputs["templates"][t]["plain"][:N_SAMPLES]
                     for t in TEMPLATES}
    dump = json.loads(Path(args.logits_json).read_text())
    out: dict = {}
    for tmpl in TEMPLATES:
        block = dump[tmpl]
        if opt_level == "L0":
            verdicts = block["L0"]
            out[tmpl] = (verdicts, None)
        else:
            logits = block[opt_level]
            verdicts = [lg.index(max(lg)) for lg in logits]
            out[tmpl] = (verdicts, logits)
    return out, forced_by_tmpl, plain_by_tmpl


def compare(got_verdicts: list[int], got_logits: list[list[float]] | None,
            ref_logits: list[list[float]], ref_verdicts: list[int]
            ) -> tuple[float | None, int]:
    """Return (max_logit_diff or None for L0, argmax_match_count vs L0 ref)."""
    n = len(got_verdicts)
    argmax_match = sum(1 for v, r in zip(got_verdicts, ref_verdicts) if v == r)
    if got_logits is None:
        return None, argmax_match
    max_diff = 0.0
    for g, r in zip(got_logits, ref_logits):
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(g, r)))
    return max_diff, argmax_match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True,
                    choices=[*RUNTIMES, "rust-candle"])
    ap.add_argument("--precision", default=None)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--opt-level", default="L1", choices=list(OPT_LEVELS))
    ap.add_argument("--logits-json", default=None,
                    help="rust-candle: pre-dumped verdict logits JSON.")
    ap.add_argument("--rust-inputs", default="rust/bench_inputs.json",
                    help="rust-candle: input bundle (for the forced/plain ids).")
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    precision = args.precision or ("fp32" if args.runtime == "rust-candle"
                                   else DEFAULT_PRECISION[args.runtime])
    opt_level = args.opt_level

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    vids = [discover_verdict_token_ids(tok, TEMPLATES[0])[v]
            for v in VERDICT_LABELS]

    if args.runtime == "rust-candle":
        if not args.logits_json:
            raise SystemExit("--logits-json is required for rust-candle.")
        out_by_tmpl, forced_by_tmpl, plain_by_tmpl = load_rust_dump(
            args, opt_level)
    else:
        texts = load_representative_texts(max_samples=N_SAMPLES, tokenizer=tok)
        forced_by_tmpl = {t: [build_forced_ids(tok, x, t) for x in texts]
                          for t in TEMPLATES}
        plain_by_tmpl = {t: [build_plain_ids(tok, x, t) for x in texts]
                         for t in TEMPLATES}
        out_by_tmpl = run_backend(args, vids, forced_by_tmpl, plain_by_tmpl,
                                  opt_level)

    strict = precision == "fp32"
    # vLLM exposes top-K logprobs, not raw logits — log-softmax is monotone in
    # the logits so argmax matches the oracle, but the absolute values diverge
    # by the per-position log-partition term. Gate vllm-cpu on argmax only.
    argmax_only = args.runtime == "vllm-cpu"
    if argmax_only:
        gate_desc = "argmax==n (vllm-cpu exposes logprobs, not logits)"
    elif strict:
        gate_desc = f"fp32 atol={FP32_ATOL:.0e} AND argmax==n"
    else:
        gate_desc = f"argmax/n>={QUANT_ARGMAX_FRAC:.0%}"
    print(f"[verify] {args.runtime}/{precision} {opt_level} "
          f"n={N_SAMPLES} gate: {gate_desc}", flush=True)

    all_ok = True
    for tmpl in TEMPLATES:
        ref_logits = reference_logits(args.model_id, forced_by_tmpl[tmpl], vids)
        ref_verdicts = reference_l0_verdicts(args.model_id,
                                             plain_by_tmpl[tmpl], vids)
        got_verdicts, got_logits = out_by_tmpl[tmpl]
        max_diff, argmax_match = compare(got_verdicts, got_logits,
                                         ref_logits, ref_verdicts)
        n = len(got_verdicts)
        if argmax_only:
            ok = argmax_match == n
        elif strict:
            ok = argmax_match == n and (max_diff is None or max_diff <= FP32_ATOL)
        else:
            ok = argmax_match >= QUANT_ARGMAX_FRAC * n
        diff_str = f"max_diff={max_diff:.3e}" if max_diff is not None else "max_diff=  -      "
        status = "PASS" if ok else "FAIL"
        print(f"[verify] {tmpl:9s} {diff_str} argmax={argmax_match}/{n} {status}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
