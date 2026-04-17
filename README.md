# qwen3guard-test

Batch=1 latency benchmarks for [Qwen3Guard](https://arxiv.org/abs/2510.14276) on **Linux + CUDA** — both the **generative classifier** (`Gen`) and the **token-stream classifier** (`Stream`), under **PyTorch** and **ONNX Runtime**.

## What it measures

| Variant  | Runtime   | Regime                              | Metric                  |
|----------|-----------|-------------------------------------|-------------------------|
| Gen      | PyTorch   | full `apply_chat_template → generate → decode` | per-request latency (P50/P95/P99) |
| Gen      | ONNX RT   | same, via `optimum.onnxruntime`     | per-request latency     |
| Stream   | PyTorch   | prefill (user prompt) + per-token (assistant stream) | prefill latency + per-token latency |
| Stream   | ONNX RT   | *not yet supported — see `scripts/export_stream_onnx.py`* | — |

Input modes:
- **Representative** (default): samples near the median token length from `Qwen/Qwen3GuardTest`.
- **Length sweep** (`--lengths 32 64 128 … 16384`): synthetic fixed-length prompts — latency-vs-length curve.

PyTorch device: CUDA (bf16). ONNX Runtime provider: `CUDAExecutionProvider` if available, else `CPUExecutionProvider`.

## Quickstart

```bash
uv sync                                # install deps (torch cu12 + onnxruntime-gpu)
uv run python scripts/download.py      # pre-fetch 0.6B models + dataset
bash run_all.sh                        # run the uncommented combos
# → results/bench_*_<timestamp>.json
```

### Larger models

```bash
uv run python scripts/download.py --sizes 0.6B 4B 8B
# then edit run_all.sh to uncomment the 4B / 8B lines
```

### Length sweeps

Edit `run_all.sh`: uncomment the block under "Length-sweep mode". Default lengths are `32 64 128 256 512 1024 2048 4096 8192 16384` — change `LENGTHS` to taste. Each length emits its own JSON result file.

## Layout

```
qwen3guard-test/
├── pyproject.toml           # uv-managed project
├── run_all.sh               # runner — one line per combo, heavily commented
├── scripts/
│   ├── download.py          # pre-fetch weights + dataset
│   ├── export_gen_onnx.py   # Gen → ONNX via optimum
│   └── export_stream_onnx.py  # stub; see comment
├── src/
│   ├── bench_common.py      # device, sample pools, latency stats
│   ├── bench_gen_pytorch.py
│   ├── bench_gen_onnx.py
│   └── bench_stream_pytorch.py
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
- Stream-ONNX export is deferred — the stateful `stream_moderate_from_ids` API and two classification heads require a custom wrapper. See `scripts/export_stream_onnx.py`.
- Gen-ONNX is currently exported without past-KV cache because optimum's dummy KV generator picks the wrong head-dim for Qwen3 (`head_dim=128` vs `hidden/heads=64`). ONNX numbers are therefore O(N²) per generated token; treat as a CPU/EP floor until the KV-cache export is fixed.
