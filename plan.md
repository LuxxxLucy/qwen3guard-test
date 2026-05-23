# Qwen3Guard-Gen CPU bench — re-plan

## Goal

Expand the CPU benchmark with three additions: a renumbered strictly-cumulative trick ladder (L0 / L1 / L2 / L3 — L3 = prefix-KV, vocab-subset measured and dropped), more ONNX-side baselines (ONNX Runtime GenAI, plus a verification pass on whether trick L2 is actually baked into our current ONNX / OpenVINO exports — confirmed baked), and a single-row whole-runtime baseline (vLLM CPU; eLLM and InferLLM dropped for architecture incompatibility).

Sanity-run on Mac. Move to Kunpeng 925 once the matrix is stable.

## Trick ladder

Each row adds **one** trick on top of the previous. Strictly cumulative — a backend can skip a level it cannot express, but cannot reorder.

- **L0 — unoptimized.** `tokenize → generate() 32-token decode loop → parse 'Safety: <verdict>'`. Per-call KV cache during decode is on (that is what `generate()` does); none of L1–L4 are.

- **L1 — +forced-prefix.** Teacher-force `"Safety: "` and read the 3 verdict logits from one forward. The decode loop is gone. Output projection still computes logits for every prompt position.

- **L2 — +lastpos lm_head only.** Slice hidden state to last position before the vocab projection. `[B, S, H] → [B, 1, H]` then `@ [H, V]`. Skips the ~200 prompt-position vocab projections. Same trick as ChatGPT's "lm_head trim" and ORT GenAI's `prune_lm_head=true`.

- **L3 — +shared system-prompt KV cache.** Precompute the shared prefix KV once. Per-call cost shrinks to the variable-suffix forward.

(Strict cumulative means no `L0 + L3` row: we never measure prefix-KV without forced-prefix. That row exists in the old PyTorch optim ladder as "L1 = prefix-cache only" and we are not bringing it forward.)

A vocab-subset projection trick (project only the 3 verdict-token rows of `lm_head`: `[B, 1, H] @ [H, 3]` instead of `[H, 151,936]`) was measured on PyTorch CPU between L2 and L3 — see the Dropped section.

## Backend matrix

Which tricks each backend can express. "default" means baked-in and always on for that runtime; "n/a" means the runtime cannot express it without source surgery we are not doing.

| backend            | L0     | L1  | L2        | L3         |
| ------------------ | ------ | --- | --------- | ---------- |
| pytorch fp32       | yes    | yes | yes       | n/a [^1]   |
| onnx fp32          | yes    | yes | default   | yes        |
| onnx int8 [^2]     | -      | yes | default   | -          |
| onnx-genai fp16    | yes (L2 baked) | yes | default | n/a [^4]  |
| openvino fp16      | yes    | yes | default   | n/a [^1]   |
| openvino int8 [^2] | -      | yes | default   | -          |
| llamacpp q8_0      | yes    | yes | default   | yes        |
| llamacpp f16       | -      | yes | default   | yes        |
| rust candle fp32   | yes    | yes | default   | yes        |
| vllm cpu           | single row, all tricks baked in (Kunpeng pre-built wheel; Mac arm64 build-from-source) |

[^1]: PyTorch and OpenVINO do not have a prefix-KV mode in the bench today; adding one is out of scope for this plan.
[^2]: Quantized rows kept at lower priority — populate after the fp32 / fp16 ladders are stable.
[^4]: Resolved — ORT GenAI's generator API does not expose cross-call KV reuse. Single-Generator + `rewind_to(prefix_len)` is the closest pattern, but it is single-session; persisting KV across separate Generators is not in the public API.

"default" = the trick is baked into the export or is the only mode the runtime supports — the same row covers L2 as L1. Phase 1 verified ONNX and OV bake the lastpos slice (`Slice` with `start=-1, axes=1` feeding the lm_head MatMul).

For **onnx-genai** specifically: ORT GenAI's model builder is LLM-aware and exposes `prune_lm_head=true` which slices the sequence dim before the LM-head matmul. We build with that flag on, so the runtime baseline already has L2 (lastpos) baked into L0. Adding L1 (forced-prefix) on top is a bench-side change, not a build flag.

## Report table layout

Two-column row label. First column is the backend; second column is the variant. Backend cell only printed on the first row of each block. Pseudo (no numbers):

| backend           | variant                       | original    | test-200    |
| ----------------- | ----------------------------- | ----------- | ----------- |
| pytorch fp32      | L0                            | -           | -           |
|                   | +L1 forced-prefix             | -           | -           |
|                   | +L2 lastpos                   | -           | -           |
| onnx fp32         | L0 (L2 baked)                 | -           | -           |
|                   | +L1 (L2 baked)                | -           | -           |
|                   | +L3 prefix-KV                 | -           | -           |
| onnx int8         | L1 (L2 baked)                 | -           | -           |
| onnx-genai fp16   | L0 (L2 baked)                 | -           | -           |
|                   | +L1                           | -           | -           |
| openvino fp16     | L0 (L2 baked)                 | -           | -           |
|                   | +L1 (L2 baked)                | -           | -           |
| openvino int8     | L1 (L2 baked)                 | -           | -           |
| llamacpp q8_0     | L0 (L2 baked)                 | -           | -           |
|                   | +L1 (L2 baked)                | -           | -           |
|                   | +L3 prefix-KV                 | -           | -           |
| llamacpp f16      | L1 (L2 baked)                 | -           | -           |
|                   | +L3 prefix-KV                 | -           | -           |
| rust-candle fp32  | L0 (L2 baked)                 | -           | -           |
|                   | +L1 (L2 baked)                | -           | -           |
|                   | +L3 prefix-KV                 | -           | -           |
| vllm cpu          | default (all baked)           | -           | -           |

Cells are `p50 / p99` ms when populated.

## Naming clash — resolved

`src/contract.py` currently defines `OPT_LEVELS = ("LP", "L0", "L1", "L2", "L2-lastpos", "L3")`. The non-CPU levels (`LP`, `L1`, `L3`) belong to an older PyTorch-only optim ladder (`bench_gen_pytorch.py`, `scripts/run_optim_ladder_gen.sh`) where `LP` = prefill-only, `L1` = prefix-cache only, `L3` = full stack with `torch.compile`. The CPU bench only ever used `L0 / L2 / L2-lastpos`.

**Decision:** drop the old PyTorch optim ladder. Replace `OPT_LEVELS` with the new ladder `("L0", "L1", "L2", "L3")` and the corresponding string literals. `bench_gen_pytorch.py` and `run_optim_ladder_gen.sh` retire alongside; the CPU bench supersedes that workflow for our scope. Phase 2 does the refactor.

## Phases

1. **Verify the L2-baked-in claim for ONNX / OpenVINO.** Inspect current `onnx_models/<name>/fp32` and `ov_models/<name>/fp16` graphs around the LM head. Confirm Gather / Slice → MatMul vs MatMul → `[B, S, V]`. Report findings before any code lands. → verify: graph dump records `[B, ?, V]` shape entering output node.

2. **Refactor `contract.py` ladder.** Replace `OPT_LEVELS` with `("L0", "L1", "L2", "L3", "L4")`. Retire `bench_gen_pytorch.py` and `scripts/run_optim_ladder_gen.sh`. Update `bench_gen_cpu.py`, `summarize_cpu.py`, `gen_backends.py` capability flags. → verify: existing CPU bench still runs end-to-end on `--dry-run`.

3. **L3 vocab-subset, PyTorch first.** Done — measured, dropped, ladder renumbered (L4 prefix-KV → L3). PyTorch L3 was on both templates **slower** than L2 (e.g. original 874 ms vs 656 ms, test-200 488 ms vs 475 ms with 20/20 verdict-match correctness). Theoretical savings (~0.8 ms on the 150 MFLOP lm_head matmul) sat below the noise floor, and the extra Python path made the wall-clock worse. Moved to Dropped.

4. **L3 vocab-subset, ONNX + OpenVINO** — skipped (no-op; Phase 3 dropped the trick).

5. **onnx-genai backend.** Add `runtime="onnx-genai"`. Build with `prune_lm_head=true` (L2 baked in). Two rows: L0 default, and +L1 forced-prefix on top. Share IO-binding / numpy plumbing with the existing ONNX backend where reusable. → verify: correctness oracle + first-call latency sanity check.

6. **vLLM CPU baseline.** Single-row, batch=1. Skip the trick ladder. Backend code (`VLLMCPUBackend` in `gen_backends.py`, runtime tag `vllm-cpu`) landed; the bench cell in `run_gen_cpu.sh` is gated on `import vllm` succeeding. On Mac the cell is skipped because the only macOS arm64 vLLM wheel (0.8.5.post1) predates Qwen3 architecture support, and the Qwen3-supporting releases (≥ 0.10) require torch ≥ 2.7.1 which conflicts with the rest of our toolchain. Kunpeng (Linux aarch64) gets pre-built wheels from 0.11+ and will populate this row. → verify on Kunpeng: end-to-end produces a verdict matching the PyTorch oracle on ≥ 9/10 samples; latency landed.

7. **eLLM and InferLLM baselines.** Both dropped after investigation (see Dropped section). Phase is a no-op; kept as a placeholder so the phase numbering stays stable.

8. **Update the report.** New two-column layout in `summarize_cpu.py`. Legend explains L0–L3 and the "baked" annotation. → verify: report renders for any subset of populated cells without crashing.

9. **Sanity-run on Mac.** Full `bash scripts/run_gen_cpu.sh`. Inspect ledger + table. → verify: all populated rows have non-`-` cells; ledger is all PASS.

10. **Kunpeng 925 port.** Pre-flight checks validated under `Dockerfile.aarch64` (Linux 6.8 aarch64 under colima on Apple Silicon, native arm64 virt, not QEMU emulation):
    - **ONNX Runtime aarch64 wheel** — published on PyPI; `pip install onnxruntime` resolves the right tag.
    - **ONNX Runtime GenAI aarch64** — *no Linux aarch64 wheel for any version (latest 0.13.2 checked).* Wheels exist for Mac arm64, Linux x86_64, Windows amd64/arm64 only. Dep is platform-gated in `pyproject.toml`; bench cell and export step gate on `import onnxruntime_genai` and skip cleanly. Row is dashed in the aarch64 summary. Re-enable when Microsoft ships the wheel or we build from source.
    - **OpenVINO ARM** — `openvino` (the runtime) and `optimum-intel` both publish aarch64 wheels; ARM CPU plugin uses a different kernel set with no AMX/AVX dependency.
    - **llama.cpp NEON / SVE** — `GGML_NATIVE=ON` lets ggml pick aarch64 SIMD at build time. Kunpeng 925 is ARMv8 with NEON; SVE is platform-dependent (Kunpeng 920 had no SVE; 925 needs checking).
    - **Rust candle** — pure-Rust kernels with optional Accelerate / MKL; aarch64 just works via `cargo build --release`. No SIMD intrinsics required for correctness.
    - **vLLM aarch64** — pre-built `vllm==0.11.0` wheels for `linux_aarch64`. Add `vllm>=0.11` to deps and unguard the bench cell (currently gated on `import vllm` succeeding).
    - **PyTorch arm64 thread tuning** — Kunpeng 925 has many physical cores; `qg_detect_threads` caps at 16 by `min(detected, 16)`. Confirm 16 is still the right cap (the AMD 5800X result suggested it; re-measure on aarch64).
    → verify: `Dockerfile.aarch64` builds, `scripts/run_gen_cpu.sh --dry-run` ledger all PASS in the container (onnx-genai row dashed as expected). Real numbers from Kunpeng 925 land once hardware access opens.

## Open questions

- (None at this stage. **Resolved:** ORT GenAI generator API does not expose cross-call KV reuse — the matrix entry for onnx-genai L4 stays at "n/a". Single-Generator + `rewind_to(prefix_len)` is the closest pattern but it is single-session; persisting KV across separate Generators is not in the public API.)

## Dropped

- **Intel `llm-on-ray`.** No Intel chip on hand and the dev target is Kunpeng 925 (ARM). Revisit if the deployment target changes.
- **Non-cumulative `L0 + L3` row** (the old PyTorch optim ladder's `L1 = prefix-cache only`). Strict-cumulative ladder by decision; row does not add information that the cumulative rows do not already give.
- **eLLM.** Targets Intel AMX (Xeon 4th-gen+) only. Not runnable on M3 macOS arm64 or Kunpeng 925 aarch64. The README lists Qwen3 in scope but the build path is x86-AMX-specific.
- **InferLLM.** No Qwen3 architecture support; README lists LLaMA / ChatGLM / Baichuan only. Custom `.bin` format with no Qwen3 converter shipped. Adding it would require porting the graph and writing a HF→`.bin` converter — out of scope for a single-row baseline.
- **L3 vocab-subset.** Replace the full `[1, H] @ [V, H].T` lm_head matmul with a `[3, H] @ [H]` projection against only the verdict-token rows. On PyTorch CPU at n=80 with `--threads 8`: L2 original 655.7 ms vs L3 original 874.0 ms (+33% **slower**); L2 test-200 475.0 ms vs L3 test-200 488.0 ms (+2.7% slower). Correctness 20/20 verdicts matched. The theoretical savings (~155 MFLOPs out of the full forward) is ~0.8 ms — below the noise floor — and the alternate Python path (`self.model.model(...)` + numpy matmul + `[float(v) for v in row]`) costs more than the savings. Ladder renumbered: L4 prefix-KV → L3.

## Deferred (post-Mac, possibly post-Kunpeng)

- **Manual `torch.onnx.export` with `LastLogitsWrapper`.** A third ONNX export path alongside Optimum and ORT GenAI. Worth doing if Phase 1 finds Optimum is not pruning and we want a clean A/B against `prune_lm_head`.
- **llama.cpp L0 off-by-one.** Currently does 33 forwards (1 prefill + 32 decode) vs 32 elsewhere.
- **PyTorch and OpenVINO L3 (prefix-KV).** Out of scope for now; the existing backends with L3 are enough to characterize the trick.
