# qwen3guard-test

Batch=1 latency benchmarks for [Qwen3Guard](https://arxiv.org/abs/2510.14276) — both the **generative classifier** (`Gen`) and the **token-stream classifier** (`Stream`).
The **GPU** path (Linux + CUDA) covers PyTorch and ONNX Runtime.
The **CPU** path benchmarks Gen across four runtimes — PyTorch, ONNX Runtime, OpenVINO, llama.cpp — and Stream on PyTorch-CPU.

## What it measures

### GPU (Linux + CUDA)

| Variant  | Runtime   | Regime                              | Metric                  |
|----------|-----------|-------------------------------------|-------------------------|
| Gen      | PyTorch   | full `apply_chat_template → generate → decode` | per-request latency (P50/P95/P99) |
| Gen      | ONNX RT   | same, via `optimum.onnxruntime`     | per-request latency     |
| Stream   | PyTorch   | prefill (user prompt) + per-token (assistant stream) | prefill latency + per-token latency |

### CPU

| Variant  | Runtime              | Precisions            | Regime                          |
|----------|----------------------|-----------------------|---------------------------------|
| Gen      | PyTorch              | fp32                  | L2 forced-prefix (one forward)  |
| Gen      | ONNX Runtime         | fp32, int8, int4      | L2 forced-prefix (one forward)  |
| Gen      | OpenVINO             | fp16, int8, int4      | L2 forced-prefix (one forward)  |
| Gen      | llama.cpp (GGUF)     | f16, q8_0, q4_k_m     | L2 forced-prefix (one forward)  |
| Stream   | PyTorch              | fp32                  | shipped API + direct path       |

The CPU Gen benchmark runs the **L2 forced-prefix** path only: one forward pass over `prompt + "Safety: "`, read the three verdict logits.
The ladder also includes **L2-lastpos**: the same forced-prefix input with projection restricted to the last logit row.
This is the optimized path from the Gen report — no decode loop, no KV cache — so it exports cleanly to every runtime.
Stream CPU stays on PyTorch: its stateful `trust_remote_code` API with four classification heads does not export cleanly to the other runtimes.

Input modes:
- **Representative** (default): samples near the median token length from `Qwen/Qwen3GuardTest`.
- **Length sweep** (`--lengths 32 64 128 … 16384`): synthetic fixed-length prompts — latency-vs-length curve.

PyTorch device: CUDA (bf16). ONNX Runtime provider: `CUDAExecutionProvider` if available, else `CPUExecutionProvider`.

## Quickstart

```bash
uv sync                                # install deps (torch cu12 + onnxruntime CPU wheel)
uv run python scripts/download.py      # pre-fetch 0.6B models + dataset
bash scripts/run_all.sh                # run the uncommented combos
# → results/bench_*_<timestamp>.json
```

### Larger models

```bash
uv run python scripts/download.py --sizes 0.6B 4B 8B
# then edit scripts/run_all.sh to uncomment the 4B / 8B lines
```

### Length sweeps

Edit `scripts/run_all.sh`: uncomment the block under "Length-sweep mode". Default lengths are `32 64 128 256 512 1024 2048 4096 8192 16384` — change `LENGTHS` to taste. Each length emits its own JSON result file.

## CPU benchmarks

One script does everything — sync, download, export to each runtime, benchmark, print a comparison table:

```bash
bash scripts/run_gen_cpu.sh
```

It runs unattended and writes all output to stdout. Knobs (environment variables):

| Var        | Default          | Effect                                                        |
|------------|------------------|---------------------------------------------------------------|
| `DRY_RUN`  | unset            | set to `1` for a smoke test — tiny warmup/iters/length; every runtime is still exported and exercised, latency numbers are not real |
| `THREADS`  | physical cores   | CPU threads each runtime is pinned to (set equal for fairness) |
| `NSAMPLES` | `100`            | Gen timed iterations per cell                                 |
| `LENGTHS`  | unset            | if set (e.g. `"32 128 512 2048"`), also run a Gen length sweep |
| `VERIFY`   | `--verify`       | set to `--no-verify` to skip the cross-runtime verdict check   |

Validate the whole pipeline first with `DRY_RUN=1 bash scripts/run_gen_cpu.sh`, then run it for real.

Each runtime/precision step is independent: a failure (missing export, runtime not installed) is reported and skipped, the rest of the run continues.

Individual cells can also be run directly:

```bash
uv run python scripts/export_gen_onnx.py --precisions fp32 int8 int4
uv run python src/bench_gen_cpu.py --runtime onnx --precision int8 \
    --artifact onnx_models/Qwen3Guard-Gen-0.6B/int8
```

GGUF conversion shallow-clones llama.cpp into `vendor/`; `q4_k_m` additionally needs `cmake` to build `llama-quantize` (skipped with a note if absent). On Apple Silicon, OpenVINO runs on the ARM CPU plugin — its latency there is a sanity check, not a predictor of x86 numbers.

## Layout

```
qwen3guard-test/
├── pyproject.toml           # uv-managed project
├── docs/                    # self-contained reports
│   ├── REPORT.md            # index
│   ├── REPORT_GEN.md        # Gen-variant findings
│   └── REPORT_STREAM.md     # Stream-variant findings
├── scripts/
│   ├── run_all.sh           # GPU runner — one line per combo
│   ├── run_gen_cpu.sh       # CPU runner — sync, export, benchmark, summarize
│   ├── run_optim_ladder_gen.sh
│   ├── run_optim_ladder_stream.sh
│   ├── download.py          # pre-fetch weights + dataset
│   ├── correctness_test.py  # L0/L1/L2 cached+uncached verdict equivalence
│   ├── accuracy_check.py    # labeled-data accuracy probe
│   ├── export_gen_onnx.py     # Gen → ONNX (fp32 / int8 / int4)
│   ├── export_gen_openvino.py # Gen → OpenVINO IR (fp16 / int8 / int4)
│   ├── export_gen_gguf.py     # Gen → GGUF for llama.cpp
│   ├── summarize_cpu.py     # CPU result JSON → comparison table
│   └── make_fig{1,2}{,_stream}.py  # figure generators
├── src/
│   ├── bench_common.py      # device, sample pools, latency stats
│   ├── gen_common.py        # Gen chat-template + forced-prefix helpers
│   ├── gen_backends.py      # CPU runtimes: pytorch / onnx / openvino / llamacpp
│   ├── bench_gen_cpu.py     # multi-runtime CPU Gen benchmark
│   ├── bench_gen_pytorch.py
│   └── bench_stream_pytorch.py          # shipped-API path
├── figures/                 # PNGs embedded by reports
└── results/                 # JSON output (gitignored)
```

## Output schema

Each run writes one JSON file:

```json
{
  "variant": "gen",                "runtime": "pytorch",
  "model_id": "Qwen/Qwen3Guard-Gen-0.6B",
  "device": "cuda",                "dtype": "bfloat16",
  "provider": null,                "n_samples": 100,
  "n_warmup": 5,
  "input_token_count_median": 42,  "output_token_count": 32,
  "latency": { "p50_ms": ..., "p95_ms": ..., "p99_ms": ..., "mean_ms": ..., "throughput_rps": ... },
  "extra": { "mode": "representative", "target_input_tokens": null },
  "timestamp_utc": "...", "host": "...", "torch_version": "..."
}
```

Stream results additionally include `extra.per_token` with the same latency struct for the per-token regime.

## Notes

- Qwen3Guard-Gen output is short (`Safety: X\nCategories: Y\nRefusal: Z`), so `max_new_tokens=32` is the default.
- Stream-ONNX export is deferred — the stateful `stream_moderate_from_ids` API and four classification heads require a custom wrapper.
- Reports live in `docs/` (`REPORT.md` is the index; Gen and Stream each have their own self-contained write-up).
- The CPU Gen benchmark exports with task `text-generation` (plain forward, no past-KV) and runs the L2 single-forward path. L2-lastpos restricts the output projection to the verdict row when the runtime supports it.
