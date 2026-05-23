# Qwen3Guard-Gen CPU benchmark

| backend          | variant             | original        | test-200        |
| ---------------- | ------------------- | --------------- | --------------- |
| pytorch fp32     | L0                  | 1764.5 / 2310.9 | 724.3 / 928.6   |
|                  | +L1 forced-prefix   | 781.6 / 1660.4  | 506.8 / 633.3   |
|                  | +L2 lastpos         | 773.2 / 941.5   | 434.2 / 527.5   |
| onnx fp32        | L0 (L2 baked)       | 3070.3 / 3520.7 | 6595.7 / 9510.5 |
|                  | +L1 (L2 baked)      | 901.2 / 1145.0  | 470.5 / 734.9   |
|                  | +L3 prefix-KV       | 527.1 / 647.4   | 285.2 / 349.3   |
| onnx int8        | L1 (L2 baked)       | 677.1 / 823.1   | 288.1 / 351.6   |
| onnx-genai fp32  | L0 (L2 baked)       | 4254.3 / 7256.1 | 2240.0 / 3146.6 |
|                  | +L1 (L2 baked)      | 3214.2 / 4359.1 | 1715.9 / 2750.1 |
| openvino fp16    | L0 (L2 baked)       | 5156.5 / 6613.4 | 2689.5 / 5742.5 |
|                  | +L1 (L2 baked)      | 4458.3 / 7004.4 | 2342.8 / 3297.1 |
| openvino int8    | L1 (L2 baked)       | 4384.6 / 6737.4 | 2265.0 / 4263.3 |
| llamacpp q8_0    | L0 (L2 baked)       | 4367.1 / 5758.6 | 2588.1 / 3710.5 |
|                  | +L1 (L2 baked)      | 2964.6 / 4505.5 | 1537.7 / 2617.6 |
|                  | +L3 prefix-KV       | 1509.2 / 2658.8 | 771.8 / 1312.6  |
| llamacpp f16     | L1 (L2 baked)       | 711.7 / 926.1   | 401.7 / 592.6   |
|                  | +L3 prefix-KV       | 522.2 / 687.1   | 297.8 / 390.2   |
| rust-candle fp32 | L0 (L2 baked)       | 1505.1 / 1505.1 | 958.1 / 958.1   |
|                  | +L1 (L2 baked)      | 2068.6 / 2415.7 | 1307.9 / 2251.3 |
|                  | +L3 prefix-KV       | 1140.4 / 1376.8 | 847.2 / 975.7   |
| vllm cpu fp16    | default (all baked) | -               | -               |

Each cell is `p50 / p99` latency in milliseconds. Every method runs 5 warmup calls then 100 timed iterations. Latency is per-call wall-clock time at batch size 1; threads are pinned to the host's physical core count.

## Legend

The columns are the two input templates: **original** (the model card's built-in Qwen3Guard system prompt, ~296-token overhead) and **test-200** (a compressed policy with ~130-token overhead).

The rows are a strictly-cumulative optimization ladder. Each `+Lk` row layers one more trick on top of the previous row within the same backend. `(L2 baked)` next to a row label means the backend bakes the lastpos lm_head trick into export, so the L0 / L1 rows already include it — there is no separate L2 row for those backends.

- **L0** — unoptimized. `tokenize → generate() decode loop (~32 tokens) → parse 'Safety: <verdict>'`. The model-card path. Per-call KV cache during the decode is on (it's what `generate()` does); none of the cross-call tricks below are.

- **L1** — `+forced-prefix`. Teacher-force `"Safety: "` and read the 3 verdict logits from one forward pass. No decode loop.

- **L2** — `+lastpos lm_head only`. Slice hidden state to last position before the vocab projection: `[B, S, H] → [B, 1, H]` then `@ [H, V]`. Skips the ~200 prompt-position vocab projections. Same trick as ChatGPT's "lm_head trim" and ORT GenAI's `prune_lm_head=true`. PyTorch exposes this via `logits_to_keep=1`. ONNX, OpenVINO, llama.cpp, and Rust candle bake it into export or default — those backends show no separate L2 row.

- **L3** — `+shared system-prompt KV cache`. Precompute the shared prefix KV once and reuse it across calls. Per-call cost shrinks to the variable-suffix forward. ONNX uses the with-past graph + IO binding; llama.cpp rewinds its context in place; Rust candle clones the primed model. PyTorch and OpenVINO don't have this mode in the bench.

A vocab-subset projection trick (project to only the 3 verdict-token rows of `lm_head`) was measured on PyTorch CPU and dropped — savings (~0.8 ms on the 150 MFLOP lm_head matmul) sit below the noise floor.
