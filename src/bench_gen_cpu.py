"""Qwen3Guard-Gen CPU latency benchmark — multi-runtime.

Benchmarks the L2 forced-prefix path (one forward over `prompt + "Safety: "`,
read the 3 verdict logits) across PyTorch / ONNX / ONNX-GenAI / OpenVINO /
llama.cpp / vLLM-CPU. The L2 path is one forward with no decode loop and no
KV cache, so it exports cleanly to every runtime.

Examples:
  uv run python src/bench_gen_cpu.py --runtime pytorch
  uv run python src/bench_gen_cpu.py --runtime onnx --precision int8 \\
      --artifact onnx_models/Qwen3Guard-Gen-0.6B/int8
"""
from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import sys
from pathlib import Path

from bench_common import (
    BenchResult, LatencyStats, load_representative_texts, synthesize_prompts,
    warmup_and_measure, write_result,
)
from contract import (
    L0_MAX_NEW_TOKENS, PROVIDER_TAG, RUNTIMES, TEMPLATES, OptLevel,
    ResultExtra, Template,
)
from gen_backends import make_backend
from gen_common import (
    VERDICT_LABELS, build_forced_ids, build_plain_ids, common_prefix,
    discover_verdict_token_ids,
)


def opt_level_of(args) -> OptLevel:
    """L0 (decode loop), L1 (forced-prefix), L2 (+lastpos), L3 (+prefix-KV)."""
    if args.unoptimized:
        return "L0"
    if args.kv_cache:
        return "L3"
    if args.last_pos_logits:
        return "L2"
    return "L1"


def build_samples(tok, n_samples: int, length: int | None,
                  template: Template, unoptimized: bool) -> list[list[int]]:
    """L2: forced_ids = tokenize(render + "Safety: "). L0: plain rendered ids.
    `length` None = representative dataset samples; else synthetic prompts."""
    if length is None:
        texts = load_representative_texts(max_samples=n_samples, tokenizer=tok)
    else:
        texts = synthesize_prompts(tok, target_tokens=length, n=n_samples, seed=length)
    if unoptimized:
        return [build_plain_ids(tok, t, template) for t in texts]
    return [build_forced_ids(tok, t, template) for t in texts]


def run_cell(backend, samples: list[list[int]], args, length: int | None,
             template: Template) -> None:
    median_tokens = int(statistics.median(len(s) for s in samples))
    lat = warmup_and_measure(backend.predict, samples, args.n_warmup, device="cpu")
    mode = "representative" if length is None else f"sweep-len{length}"
    opt_level = opt_level_of(args)
    output_token_count = L0_MAX_NEW_TOKENS if args.unoptimized else 1
    res = BenchResult(
        variant="gen", runtime=backend.runtime, model_id=args.model_id,
        device="cpu", dtype=backend.precision,
        provider=PROVIDER_TAG[backend.runtime],
        n_samples=len(lat), n_warmup=args.n_warmup,
        input_token_count_median=median_tokens,
        output_token_count=output_token_count,
        latency=LatencyStats.from_samples(lat),
        extra=ResultExtra(
            mode=mode, opt_level=opt_level, runtime=backend.runtime,
            precision=backend.precision, threads=args.threads,
            runtime_detail=backend.detail, target_user_tokens=length,
            template=template, kv_cache=args.kv_cache,
        ).to_dict(),
    )
    tag = f"_{backend.precision}_{template}_{opt_level}"
    if args.kv_cache:
        tag += "_kvcache"
    write_result(res, Path(args.out_dir), tag=tag)
    lt = res.latency
    kv_tag = " +kv" if args.kv_cache else ""
    mode_part = "" if length is None else f" {mode}"
    print(f"[gen] {backend.runtime}/{backend.precision} {opt_level}{kv_tag} "
          f"{template}{mode_part} len={median_tokens}  "
          f"p50={lt.p50_ms:.1f}ms p99={lt.p99_ms:.1f}ms "
          f"thr={lt.throughput_rps:.2f} rps")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=list(RUNTIMES))
    ap.add_argument("--precision", default=None,
                    help="Runtime-specific. Default: the runtime's full precision.")
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--template", nargs="+", default=list(TEMPLATES),
                    choices=list(TEMPLATES),
                    help="Templates to bench (reuses the loaded backend).")
    ap.add_argument("--artifact", default=None,
                    help="Exported-model path. Required for non-pytorch runtimes.")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None,
                    help="Pin to N CPU threads. Set equal across runtimes for fair compare.")
    ap.add_argument("--lengths", type=int, nargs="+", default=None,
                    help="Length sweep (synthetic user-token counts). Default: representative.")
    ap.add_argument("--kv-cache", action="store_true",
                    help="Precompute the shared prefix KV once, time the suffix-only forward.")
    ap.add_argument("--unoptimized", action="store_true",
                    help="L0 mode: time the model-card generate() decode loop instead of L2.")
    ap.add_argument("--last-pos-logits", action="store_true",
                    help="pytorch only: project only the last position to vocab. "
                         "Off: full-sequence projection (the L1 before-row).")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--dry-run", action="store_true",
                    help="Smoke test: 2 samples, 1 warmup. Latency numbers not meaningful.")
    args = ap.parse_args()

    if args.dry_run:
        args.n_samples, args.n_warmup = 2, 1
        if not args.lengths:
            args.lengths = [16]
        print("[bench-gen-cpu] DRY RUN — tiny params, latency numbers not meaningful.")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    # Verdict token ids are template-independent — one probe serves every cell.
    verdict_ids = discover_verdict_token_ids(tok, args.template[0])
    verdict_token_ids = [verdict_ids[v] for v in VERDICT_LABELS]

    cells: list[int | None] = list(args.lengths) if args.lengths else [None]
    # Sample-build chatter would interleave with per-cell prints; silence it.
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        sample_pools = {
            (tmpl, L): build_samples(tok, args.n_samples, L, tmpl, args.unoptimized)
            for tmpl in args.template for L in cells
        }
    max_seq_len = max(len(s) for pool in sample_pools.values() for s in pool)

    backend = make_backend(args.runtime, args.precision, verdict_token_ids,
                           args.threads, args.kv_cache, args.unoptimized,
                           args.last_pos_logits)
    backend.load(args.model_id, args.artifact, max_seq_len)

    for tmpl in args.template:
        tmpl_pools = {L: sample_pools[(tmpl, L)] for L in cells}
        if args.kv_cache:
            prefix = common_prefix([s for pool in tmpl_pools.values() for s in pool])
            backend.prime_prefix(prefix)
        for L in cells:
            run_cell(backend, tmpl_pools[L], args, L, tmpl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
