"""Qwen3Guard-Gen ONNX Runtime batch=1 latency benchmark.

Uses optimum's `ORTModelForCausalLM` so generation reuses the HF generate()
path but with ORT kernels under the hood. Provider: CUDAExecutionProvider
if available, else CPUExecutionProvider (override with --provider).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from bench_common import (
    BenchResult, LatencyStats, load_representative_texts, synthesize_input_ids,
    warmup_and_measure, write_result,
)


def pick_ort_provider(override: str | None) -> str:
    if override:
        return override
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return "CPUExecutionProvider"
    if "CUDAExecutionProvider" in avail:
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True,
                    help="Directory containing exported ONNX model "
                         "(see scripts/export_gen_onnx.py).")
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B",
                    help="Source HF id (for tokenizer + metadata tagging).")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--provider", default=None,
                    help="Override ORT provider.")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--lengths", type=int, nargs="+", default=None,
                    help="Length sweep (synthetic prompts of these token "
                         "counts). Default: representative dataset.")
    args = ap.parse_args()

    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    provider = pick_ort_provider(args.provider)
    print(f"[bench-gen-onnx] onnx={args.onnx_dir} provider={provider}")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    # We export with task="text-generation" (no past-KV) because Qwen3's
    # GQA head_dim != hidden/heads trips optimum's dummy KV generator on
    # text-generation-with-past. That means use_cache=False at load time.
    model = ORTModelForCausalLM.from_pretrained(
        args.onnx_dir, provider=provider, use_cache=False, use_io_binding=False,
    )
    model.generation_config.use_cache = False

    def step_text(text: str) -> None:
        inputs = tok([text], return_tensors="pt")
        model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    device_tag = "cuda" if "CUDA" in provider else "cpu"

    empty_rendered = tok.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=False,
    )
    template_overhead = len(tok.encode(empty_rendered, add_special_tokens=False))

    if args.lengths:
        # Sweep mode: build user content of L tokens, wrap with chat template,
        # then measure. Report user / template / total so the classifier's
        # template floor is visible alongside the length axis.
        for L in args.lengths:
            user_ids = synthesize_input_ids(tok, target_tokens=L)
            user_text = tok.decode(user_ids, skip_special_tokens=True)
            user_tokens_actual = len(tok.encode(user_text, add_special_tokens=False))
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": user_text}], tokenize=False,
            )
            total_tokens = len(tok.encode(rendered, add_special_tokens=False))
            rendered_pool = [rendered] * args.n_samples
            lat = warmup_and_measure(step_text, rendered_pool, args.n_warmup, device="cpu")
            res = BenchResult(
                variant="gen",
                runtime="onnx",
                model_id=args.model_id,
                device=device_tag,
                dtype="fp32",
                provider=provider,
                n_samples=len(lat),
                n_warmup=args.n_warmup,
                input_token_count_median=total_tokens,
                output_token_count=args.max_new_tokens,
                latency=LatencyStats.from_samples(lat),
                extra={"onnx_dir": args.onnx_dir, "mode": f"sweep-len{L}",
                       "target_user_tokens": L,
                       "user_tokens_actual": user_tokens_actual,
                       "template_overhead_tokens": template_overhead,
                       "total_input_tokens": total_tokens,
                       "chat_template_applied": True},
            )
            path = write_result(res, Path(args.out_dir))
            print(f"[bench-gen-onnx] (sweep-len{L}  user={user_tokens_actual}  "
                  f"template={template_overhead}  total={total_tokens}) wrote {path}")
            print(f"[bench-gen-onnx] p50={res.latency.p50_ms:.1f}ms "
                  f"p95={res.latency.p95_ms:.1f}ms p99={res.latency.p99_ms:.1f}ms "
                  f"thr={res.latency.throughput_rps:.2f} rps")
    else:
        samples = load_representative_texts(max_samples=args.n_samples, tokenizer=tok)
        if not samples:
            print("[err] no samples.", file=sys.stderr)
            return 1
        rendered = [
            tok.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False,
            )
            for t in samples
        ]
        input_token_lens = [len(tok.encode(r, add_special_tokens=False)) for r in rendered]
        median_tokens = int(statistics.median(input_token_lens))
        lat = warmup_and_measure(step_text, rendered, args.n_warmup, device="cpu")
        res = BenchResult(
            variant="gen",
            runtime="onnx",
            model_id=args.model_id,
            device=device_tag,
            dtype="fp32",
            provider=provider,
            n_samples=len(lat),
            n_warmup=args.n_warmup,
            input_token_count_median=median_tokens,
            output_token_count=args.max_new_tokens,
            latency=LatencyStats.from_samples(lat),
            extra={"onnx_dir": args.onnx_dir, "mode": "representative",
                   "target_input_tokens": None,
                   "chat_template_applied": True},
        )
        path = write_result(res, Path(args.out_dir))
        print(f"[bench-gen-onnx] (representative len={median_tokens}) wrote {path}")
        print(f"[bench-gen-onnx] p50={res.latency.p50_ms:.1f}ms "
              f"p95={res.latency.p95_ms:.1f}ms p99={res.latency.p99_ms:.1f}ms "
              f"thr={res.latency.throughput_rps:.2f} rps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
