"""Per-component breakdown of llama.cpp L1 forced-prefix latency.

Times reset / eval / get_logits per call so their sum matches cell wall-clock,
and reports llama_perf_context engine-side prefill ms so wrapper overhead is
visible separately.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench_common import synthesize_prompts
from gen_common import build_forced_ids


def percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--artifact", required=True, help="path to .gguf")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--template-len", type=int, default=200)
    args = ap.parse_args()

    from llama_cpp import Llama
    import llama_cpp
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    texts = synthesize_prompts(tok, target_tokens=args.template_len,
                                n=args.iters + args.warmup, seed=args.template_len)
    forced = [build_forced_ids(tok, t, "test-200") for t in texts]
    print(f"[probe] forced_ids length p50={percentile([len(f) for f in forced], 50):.0f}")

    n_ctx = max(2048, ((args.template_len + 256) // 256 + 1) * 256)
    print(f"[probe] loading {args.artifact}  n_ctx={n_ctx}  threads={args.threads}")
    t0 = time.perf_counter()
    llm = Llama(model_path=args.artifact, n_ctx=n_ctx,
                n_threads=args.threads, n_threads_batch=args.threads,
                n_gpu_layers=0, verbose=False)
    print(f"[probe] load: {(time.perf_counter() - t0) * 1000:.1f} ms")
    ctx = llm._ctx.ctx
    get_logits = llama_cpp.llama_get_logits_ith

    def one_call(ids: list[int]) -> dict[str, float]:
        t0 = time.perf_counter()
        llm.reset()
        t1 = time.perf_counter()
        llm.eval(ids)
        t2 = time.perf_counter()
        _ = get_logits(ctx, -1)[0]
        t3 = time.perf_counter()
        return {
            "reset_ms": (t1 - t0) * 1000,
            "eval_ms":  (t2 - t1) * 1000,
            "logits_ms":(t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }

    # Warmup
    for ids in forced[:args.warmup]:
        one_call(ids)
    llama_cpp.llama_perf_context_reset(ctx)

    # Measure
    samples = [one_call(ids) for ids in forced[args.warmup:]]
    keys = ["reset_ms", "eval_ms", "logits_ms", "total_ms"]
    print(f"\n[probe] {args.artifact}  iters={len(samples)}  template=test-{args.template_len}")
    print(f"  {'component':<12} {'p50':>8} {'p99':>8}  ({'mean':>7})")
    for k in keys:
        xs = [s[k] for s in samples]
        print(f"  {k:<12} {percentile(xs, 50):>8.2f} {percentile(xs, 99):>8.2f}  "
              f"({sum(xs)/len(xs):>7.2f})")

    # ggml engine perf
    pdata = llama_cpp.llama_perf_context(ctx)
    n_p = pdata.n_p_eval or 1
    n_e = pdata.n_eval or 1
    print(f"\n[probe] llama_perf_context (engine-side, totals over {len(samples)} calls):")
    print(f"  t_p_eval_ms    = {pdata.t_p_eval_ms:8.1f}  ({pdata.t_p_eval_ms/len(samples):6.2f} ms/call)")
    print(f"  t_eval_ms      = {pdata.t_eval_ms:8.1f}  ({pdata.t_eval_ms/len(samples):6.2f} ms/call)")
    print(f"  n_p_eval       = {pdata.n_p_eval}  ({n_p/len(samples):.1f} prefill tok/call)")
    print(f"  n_eval         = {pdata.n_eval}    ({n_e/len(samples):.1f} decode tok/call)")
    print(f"  prefill ms/tok = {pdata.t_p_eval_ms/n_p:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
