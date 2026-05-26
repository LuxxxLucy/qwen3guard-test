# qwen3guard-test

Batch=1 CPU latency benchmark for [Qwen3Guard-Gen-0.6B](https://arxiv.org/abs/2510.14276) across nine inference backends, with a cross-backend correctness gate.

## What it does

Measures per-call latency (p50 / p99) for one safety-classification verdict at batch size 1, across two input templates (`original` ≈ 371 tokens; `test-200` = 200 tokens).
100 timed iterations per cell, 5 warmups.

Backends: **PyTorch**, **ONNX Runtime** (fp32 + int8), **llama.cpp** (f32 + f32-kopt + f16 + q8_0), **CTranslate2** (fp32), **MNN-LLM** (fp16), **vLLM-CPU** (fp32), **Rust candle** (fp32).

Cumulative optimization ladder:

| Level | Trick |
|---|---|
| L0 | unoptimized — `tokenize → generate(max_new=32) → parse verdict` |
| L1 | + forced-prefix — teacher-force `"Safety: "`, one forward, read 3 verdict logits |
| L2 | + lastpos lm_head — slice hidden state to last position before the vocab projection |
| L3 | + prefix KV cache — precompute the shared system-prompt KV once |

After a full run, the method × template comparison lands as `var/results/REPORT_CPU_GEN_AUTO.md`, with per-cell JSONs under `var/results/`.

## Layout

```
scripts/      entry points + shell helpers
src/          bench drivers, backends/, exporters/
docs/         self-contained reports + figures
var/          runtime state (gitignored): models/, results/, logs/, vendor/
```

## Run

```bash
bash scripts/run_gen_cpu.sh --dry-run   # smoke test — exports + correctness gate + tiny bench
bash scripts/run_gen_cpu.sh             # full results
```

## Misc

- vLLM lives in a separate venv (its `torch` pin conflicts with the main one): `bash scripts/setup_vllm_venv.sh` (once).
- `pymnn` is not in `pyproject.toml` (no reliable aarch64 wheel cadence). Install separately if the MNN-LLM row is wanted: `uv pip install MNN`. The bench cell auto-skips otherwise.
- Correctness gate only (skips exports and bench cells): `bash scripts/verify_correctness.sh`.
- Regenerate the auto-table from existing JSONs without re-running: `uv run python scripts/summarize_cpu.py` (writes `var/results/REPORT_CPU_GEN_AUTO.md`).
