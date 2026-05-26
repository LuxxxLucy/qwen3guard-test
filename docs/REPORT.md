# qwen3guard-test — Reports

This repository benchmarks the Qwen3Guard content-safety classifier family
against the gateway's A8 performance contract. The two variants differ
enough structurally to each warrant their own report.

- **[REPORT_GEN.md](REPORT_GEN.md)** — Qwen3Guard-Gen-0.6B (full-text
  classification, A8 T1). Covers the chat-template anatomy, baseline
  naive-decode latency, the forced-prefix single-forward optimization,
  and the verdict against the 200 ms P99 budget.
- **[REPORT_STREAM.md](REPORT_STREAM.md)** — Qwen3Guard-Stream-0.6B
  (per-token streaming classification, A8 T2). Covers the shipped-API
  baseline, the direct-path bypass, and the long-context × chunk-size
  sweep.

Source code lives in `src/`. The CPU benchmark companion to REPORT_GEN.md is driven by `scripts/run_gen_cpu.sh`; results land under `var/`.
