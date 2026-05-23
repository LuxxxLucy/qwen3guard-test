"""Qwen3Guard-Gen CPU latency benchmark — multi-runtime.

Benchmarks the L2 forced-prefix path (one forward over `prompt + "Safety: "`,
read the 3 verdict logits) across CPU runtimes: PyTorch, ONNX Runtime,
OpenVINO, llama.cpp — each at one or more precisions. This is the same
optimized path REPORT_GEN measures on GPU; here it runs on CPU so the gateway
can be sized for CPU-only deployment.

The L2 path is one forward pass with no decode loop and no KV cache, so it
exports cleanly to every runtime — unlike the model-card `generate()` path,
whose autoregressive decode dominates and whose no-KV ONNX export is O(N^2).

Examples:
  uv run python src/bench_gen_cpu.py --runtime pytorch
  uv run python src/bench_gen_cpu.py --runtime onnx     --precision int8 \\
      --artifact onnx_models/Qwen3Guard-Gen-0.6B/int8
  uv run python src/bench_gen_cpu.py --runtime openvino --precision int8 \\
      --artifact ov_models/Qwen3Guard-Gen-0.6B/int8
  uv run python src/bench_gen_cpu.py --runtime llamacpp --precision q8_0 \\
      --artifact gguf_models/Qwen3Guard-Gen-0.6B.q8_0.gguf
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
from gen_backends import PyTorchCPUBackend, make_backend
from gen_common import (
    VERDICT_LABELS, build_forced_ids, build_plain_ids, common_prefix,
    discover_verdict_token_ids,
)


def opt_level_of(args) -> OptLevel:
    """Result-JSON opt_level tag for this run on the cumulative ladder:
    L0 (decode loop), L1 (forced-prefix), L2 (+lastpos), L3 (+prefix-KV).
    --kv-cache implies the full preceding stack on backends that support it."""
    if args.unoptimized:
        return "L0"
    if args.kv_cache:
        return "L3"
    if args.last_pos_logits:
        return "L2"
    return "L1"


def build_samples(tok, n_samples: int, length: int | None,
                  template: Template, unoptimized: bool) -> list[list[int]]:
    """Return n input token sequences for one (length, template) cell.
    `length` None = representative dataset samples; otherwise synthetic prompts
    of that user-token length. `template` selects the system-prompt block.
    L2 (default): forced-prefix ids = tokenize(render + "Safety: "). L0
    (unoptimized): the plain rendered prompt, no trailing "Safety: "."""
    if length is None:
        texts = load_representative_texts(max_samples=n_samples, tokenizer=tok)
    else:
        texts = synthesize_prompts(tok, target_tokens=length, n=n_samples, seed=length)
    if unoptimized:
        return [build_plain_ids(tok, t, template) for t in texts]
    return [build_forced_ids(tok, t, template) for t in texts]


def build_samples_for_mode(tok, n_samples: int, length: int | None,
                           template: Template, unoptimized: bool,
                           verbose: bool) -> list[list[int]]:
    if verbose:
        return build_samples(tok, n_samples, length, template, unoptimized)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return build_samples(tok, n_samples, length, template, unoptimized)


def verify_against_pytorch(ref, backend, samples: list[list[int]],
                           verbose: bool) -> None:
    """Cross-check: the runtime under test should produce the same verdict
    argmax as the PyTorch-CPU fp32 reference. Quantization may flip a borderline
    sample, so this reports agreement rather than asserting — a low count is a
    warning, not a crash."""
    probe = samples[:10]
    agree = sum(ref.predict(s) == backend.predict(s) for s in probe)
    if verbose:
        tag = "OK" if agree == len(probe) else "DRIFT"
        print(f"[verify] verdict-argmax agreement vs pytorch-cpu fp32: "
              f"{agree}/{len(probe)}  {tag}")


def run_cell(backend, samples: list[list[int]], args, length: int | None,
             template: Template) -> None:
    median_tokens = int(statistics.median(len(s) for s in samples))
    lat = warmup_and_measure(backend.predict, samples, args.n_warmup, device="cpu")
    mode = "representative" if length is None else f"sweep-len{length}"
    opt_level = opt_level_of(args)
    output_token_count = L0_MAX_NEW_TOKENS if args.unoptimized else 1
    res = BenchResult(
        variant="gen",
        runtime=backend.runtime,
        model_id=args.model_id,
        device="cpu",
        dtype=backend.precision,
        provider=PROVIDER_TAG[backend.runtime],
        n_samples=len(lat),
        n_warmup=args.n_warmup,
        input_token_count_median=median_tokens,
        output_token_count=output_token_count,
        latency=LatencyStats.from_samples(lat),
        extra=ResultExtra(
            mode=mode,
            opt_level=opt_level,
            runtime=backend.runtime,
            precision=backend.precision,
            threads=args.threads,
            runtime_detail=backend.detail,
            target_user_tokens=length,
            template=template,
            kv_cache=args.kv_cache,
        ).to_dict(),
    )
    tag = f"_{backend.precision}_{template}_{opt_level}"
    if args.kv_cache:
        tag += "_kvcache"
    path = write_result(res, Path(args.out_dir), tag=tag)
    lt = res.latency
    kv_tag = " +kv" if args.kv_cache else ""
    if args.verbose:
        print(f"[bench-gen-cpu] {backend.runtime}/{backend.precision} {opt_level}{kv_tag} "
              f"{template} {mode} len={median_tokens}  p50={lt.p50_ms:.1f}ms "
              f"p95={lt.p95_ms:.1f}ms p99={lt.p99_ms:.1f}ms "
              f"thr={lt.throughput_rps:.2f} rps")
        print(f"[bench-gen-cpu] wrote {path}")
    else:
        mode_part = "" if length is None else f" {mode}"
        print(f"[gen] {backend.runtime}/{backend.precision} {opt_level}{kv_tag} "
              f"{template}{mode_part} len={median_tokens}  "
              f"p50={lt.p50_ms:.1f}ms p99={lt.p99_ms:.1f}ms "
              f"thr={lt.throughput_rps:.2f} rps")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=list(RUNTIMES))
    ap.add_argument("--precision", default=None,
                    help="fp32|int8 (onnx); fp16|int8|int4 (openvino); "
                         "f16|q8_0 (llamacpp). Default: the runtime's full precision.")
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B",
                    help="Source HF id — tokenizer + pytorch weights + tagging.")
    ap.add_argument("--template", nargs="+", default=list(TEMPLATES),
                    choices=list(TEMPLATES),
                    help="Input template(s). One invocation benchmarks every "
                         "listed template, reusing the loaded backend. "
                         "original: the model's built-in Qwen3Guard prompt "
                         "(~296-token overhead). test-200: a compressed policy "
                         "(~130-token overhead) so a representative input lands "
                         "near 200 total tokens.")
    ap.add_argument("--artifact", default=None,
                    help="Exported-model path: ONNX export dir, OpenVINO export "
                         "dir, or .gguf file. Required for non-pytorch runtimes.")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None,
                    help="Pin the runtime to N CPU threads. Default: runtime "
                         "default. Set equal across runtimes for fair compare.")
    ap.add_argument("--lengths", type=int, nargs="+", default=None,
                    help="Length sweep (synthetic user-token counts). One "
                         "result file per length. Default: representative.")
    ap.add_argument("--kv-cache", action="store_true",
                    help="llama.cpp / onnx: precompute the shared system-prompt "
                         "prefix KV once, then time only the variable-suffix "
                         "forward per request — the prefix-cache speedup. For "
                         "onnx, point --artifact at the withpast export dir.")
    ap.add_argument("--unoptimized", action="store_true",
                    help="L0 mode: time the model-card greedy generate() decode "
                         "loop (max_new_tokens=32) over the plain rendered "
                         "prompt instead of the L2 single forced-prefix "
                         "forward. Implemented for every runtime; onnx uses the "
                         "with-past graph (point --artifact at the withpast "
                         "export). The cross-runtime verify is skipped.")
    ap.add_argument("--last-pos-logits", action="store_true",
                    help="pytorch only: compute the output projection (token "
                         "logits) for the last position only. The L2 path "
                         "reads just that row, so projecting the prompt "
                         "positions is wasted. Off: the full-sequence "
                         "projection — the before-row for this comparison. "
                         "onnx/openvino bake this into the export; llama.cpp "
                         "always does it.")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                    help="Cross-check verdict argmax against PyTorch-CPU fp32 on "
                         "10 samples. Skipped when --runtime pytorch.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Smoke test: 2 samples, 1 warmup, one short synthetic "
                         "length, no cross-runtime verify. Confirms the runtime "
                         "executes end to end — the latency numbers are not "
                         "meaningful.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print startup diagnostics and per-cell result paths.")
    args = ap.parse_args()

    if args.dry_run:
        args.n_samples, args.n_warmup, args.verify = 2, 1, False
        if not args.lengths:
            args.lengths = [16]
        print("[bench-gen-cpu] DRY RUN — tiny params; latency numbers are not "
              "meaningful, this only checks the runtime executes.")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    # Verdict token ids are prompt-independent (the "Safety: " boundary is
    # fixed text), so one probe serves every template; the first suffices.
    verdict_ids = discover_verdict_token_ids(tok, args.template[0])
    verdict_token_ids = [verdict_ids[v] for v in VERDICT_LABELS]
    if args.verbose:
        print(f"[bench-gen-cpu] model={args.model_id} runtime={args.runtime} "
              f"templates={args.template} opt_level={opt_level_of(args)} "
              f"verdict_token_ids={dict(verdict_ids)}")

    cells: list[int | None] = list(args.lengths) if args.lengths else [None]
    # sample pools indexed by (template, length); built once, reused per cell.
    sample_pools = {
        (tmpl, L): build_samples_for_mode(
            tok, args.n_samples, L, tmpl, args.unoptimized, args.verbose,
        )
        for tmpl in args.template for L in cells
    }
    max_seq_len = max(len(s) for pool in sample_pools.values() for s in pool)

    backend = make_backend(args.runtime, args.precision, verdict_token_ids,
                           args.threads, args.kv_cache, args.unoptimized,
                           args.last_pos_logits)
    backend.load(args.model_id, args.artifact, max_seq_len)
    if args.verbose:
        print(f"[bench-gen-cpu] backend loaded: {backend.detail}")

    # PyTorch-CPU fp32 reference for the cross-runtime verdict check — loaded
    # once and reused across templates (the model is template-independent).
    ref = None
    if args.verify and args.runtime != "pytorch" and not args.unoptimized:
        ref = PyTorchCPUBackend("fp32", verdict_token_ids, threads=None)
        ref.load(args.model_id, None, max_seq_len)

    for tmpl in args.template:
        tmpl_pools = {L: sample_pools[(tmpl, L)] for L in cells}
        if args.kv_cache:
            prefix = common_prefix([s for pool in tmpl_pools.values() for s in pool])
            backend.prime_prefix(prefix)
            if args.verbose:
                print(f"[bench-gen-cpu] kv-cache primed [{tmpl}]: shared prefix = "
                      f"{len(prefix)} tokens (suffix-only forward per request)")
        if ref is not None:
            verify_against_pytorch(ref, backend, tmpl_pools[cells[0]], args.verbose)
        for L in cells:
            run_cell(backend, tmpl_pools[L], args, L, tmpl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
