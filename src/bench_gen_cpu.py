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
  uv run python src/bench_gen_cpu.py --runtime openvino --precision int4 \\
      --artifact ov_models/Qwen3Guard-Gen-0.6B/int4
  uv run python src/bench_gen_cpu.py --runtime llamacpp --precision q4_k_m \\
      --artifact gguf_models/Qwen3Guard-Gen-0.6B.q4_k_m.gguf
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from bench_common import (
    BenchResult, LatencyStats, load_representative_texts, synthesize_prompts,
    warmup_and_measure, write_result,
)
from gen_backends import PROVIDER_TAG, make_backend
from gen_common import VERDICT_LABELS, build_forced_ids, discover_verdict_token_ids


def build_samples(tok, n_samples: int, length: int | None) -> list[list[int]]:
    """Return n forced-prefix token sequences. `length` None = representative
    dataset samples; otherwise synthetic prompts of that user-token length."""
    if length is None:
        texts = load_representative_texts(max_samples=n_samples, tokenizer=tok)
    else:
        texts = synthesize_prompts(tok, target_tokens=length, n=n_samples, seed=length)
    return [build_forced_ids(tok, t) for t in texts]


def verify_against_pytorch(model_id: str, backend, samples: list[list[int]],
                           verdict_token_ids: list[int]) -> None:
    """Cross-check: the runtime under test should produce the same verdict
    argmax as PyTorch-CPU fp32. Quantization may flip a borderline sample, so
    this reports agreement rather than asserting — a low count is a warning,
    not a crash."""
    from gen_backends import PyTorchCPUBackend
    ref = PyTorchCPUBackend("fp32", verdict_token_ids, threads=None)
    ref.load(model_id, None, max_seq_len=max(len(s) for s in samples))
    probe = samples[:10]
    agree = sum(ref.predict(s) == backend.predict(s) for s in probe)
    tag = "OK" if agree == len(probe) else "DRIFT"
    print(f"[verify] verdict-argmax agreement vs pytorch-cpu fp32: "
          f"{agree}/{len(probe)}  {tag}")
    del ref


def run_cell(backend, samples: list[list[int]], args, length: int | None) -> None:
    median_tokens = int(statistics.median(len(s) for s in samples))
    lat = warmup_and_measure(backend.predict, samples, args.n_warmup, device="cpu")
    mode = "representative" if length is None else f"sweep-len{length}"
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
        output_token_count=1,
        latency=LatencyStats.from_samples(lat),
        extra={
            "mode": mode,
            "opt_level": "L2",
            "runtime": backend.runtime,
            "precision": backend.precision,
            "threads": args.threads,
            "runtime_detail": backend.detail,
            "target_user_tokens": length,
            "chat_template_applied": True,
        },
    )
    path = write_result(res, Path(args.out_dir), tag=f"_{backend.precision}")
    lt = res.latency
    print(f"[bench-gen-cpu] {backend.runtime}/{backend.precision} {mode} "
          f"len={median_tokens}  p50={lt.p50_ms:.1f}ms p95={lt.p95_ms:.1f}ms "
          f"p99={lt.p99_ms:.1f}ms thr={lt.throughput_rps:.2f} rps")
    print(f"[bench-gen-cpu] wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True,
                    choices=["pytorch", "onnx", "openvino", "llamacpp"])
    ap.add_argument("--precision", default=None,
                    help="fp32|int8|int4 (onnx/openvino); f16|q8_0|q4_k_m "
                         "(llamacpp). Default: the runtime's full precision.")
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B",
                    help="Source HF id — tokenizer + pytorch weights + tagging.")
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
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                    help="Cross-check verdict argmax against PyTorch-CPU fp32 on "
                         "10 samples. Skipped when --runtime pytorch.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Smoke test: 2 samples, 1 warmup, one short synthetic "
                         "length, no cross-runtime verify. Confirms the runtime "
                         "executes end to end — the latency numbers are not "
                         "meaningful.")
    args = ap.parse_args()

    if args.dry_run:
        args.n_samples, args.n_warmup, args.verify = 2, 1, False
        if not args.lengths:
            args.lengths = [16]
        print("[bench-gen-cpu] DRY RUN — tiny params; latency numbers are not "
              "meaningful, this only checks the runtime executes.")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    verdict_ids = discover_verdict_token_ids(tok)
    verdict_token_ids = [verdict_ids[v] for v in VERDICT_LABELS]
    print(f"[bench-gen-cpu] model={args.model_id} runtime={args.runtime} "
          f"verdict_token_ids={dict(verdict_ids)}")

    cells: list[int | None] = list(args.lengths) if args.lengths else [None]
    sample_pools = {L: build_samples(tok, args.n_samples, L) for L in cells}
    max_seq_len = max(len(s) for pool in sample_pools.values() for s in pool)

    backend = make_backend(args.runtime, args.precision, verdict_token_ids, args.threads)
    backend.load(args.model_id, args.artifact, max_seq_len)
    print(f"[bench-gen-cpu] backend loaded: {backend.detail}")

    if args.verify and args.runtime != "pytorch":
        verify_against_pytorch(args.model_id, backend,
                               sample_pools[cells[0]], verdict_token_ids)

    for L in cells:
        run_cell(backend, sample_pools[L], args, L)
    return 0


if __name__ == "__main__":
    sys.exit(main())
