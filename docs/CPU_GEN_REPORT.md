# Qwen3Guard-Gen CPU benchmark

| backend                   | variant             | original          | test-200        |
| ------------------------- | ------------------- | ----------------- | --------------- |
| pytorch fp32              | L0                  | 2092.7 / 2754.2   | 894.8 / 1256.9  |
|                           | +L1 forced-prefix   | 678.1 / 701.8     | 419.1 / 428.1   |
|                           | +L2 lastpos         | 555.3 / 613.7     | 352.9 / 362.0   |
| onnx fp32                 | L0 (L2 baked)       | 1572.6 / 1603.6   | 1206.7 / 1221.7 |
|                           | +L1 (L2 baked)      | 483.6 / 502.9     | 255.3 / 267.6   |
|                           | +L3 prefix-KV       | 247.8 / 261.2     | 146.9 / 160.1   |
| onnx int8                 | L1 (L2 baked)       | 279.1 / 284.4     | 155.7 / 162.0   |
| onnx-genai fp32           | L0 (L2 baked)       | -                 | -               |
|                           | +L1 (L2 baked)      | -                 | -               |
| openvino fp16             | L0 (L2 baked)       | -                 | -               |
|                           | +L1 (L2 baked)      | -                 | -               |
| openvino int8             | L1 (L2 baked)       | -                 | -               |
| llamacpp q8_0             | L0 (L2 baked)       | 643.5 / 657.0     | 432.3 / 437.6   |
|                           | +L1 (L2 baked)      | 260.4 / 273.3     | 110.1 / 113.7   |
|                           | +L3 prefix-KV       | 133.9 / 139.2     | 71.9 / 76.3     |
| llamacpp f16              | L1 (L2 baked)       | 905.4 / 934.1     | 614.0 / 648.0   |
|                           | +L3 prefix-KV       | 645.5 / 688.3     | 539.3 / 572.8   |
| llamacpp f32              | L1 (L2 baked)       | 732.3 / 750.7     | 426.5 / 468.7   |
|                           | +L3 prefix-KV       | 441.6 / 452.9     | 323.0 / 408.8   |
| llamacpp f32 +kernel-opt  | L1 (L2 baked)       | 510.1 / 521.3     | 237.8 / 246.1   |
|                           | +L3 prefix-KV       | 241.4 / 249.4     | 147.3 / 155.9   |
| llamacpp f32 +kernel-opt2 | L1 (L2 baked)       | 10089.6 / 10245.7 | 5158.0 / 5318.4 |
|                           | +L3 prefix-KV       | 4806.7 / 5015.7   | 3218.3 / 3525.5 |
| llamacpp f32 +kernel-opt3 | L1 (L2 baked)       | 504.4 / 516.1     | 234.4 / 242.9   |
|                           | +L3 prefix-KV       | 239.7 / 246.4     | 148.9 / 157.5   |
| llamacpp q4_K_M           | L1 (L2 baked)       | -                 | -               |
|                           | +L3 prefix-KV       | -                 | -               |
| rust-candle fp32          | L0                  | 6156.1 / 6239.5   | 5217.1 / 5295.7 |
|                           | +L1 forced-prefix   | 1272.2 / 1347.1   | 536.4 / 553.6   |
|                           | +L3 prefix-KV       | 736.9 / 778.3     | 377.9 / 392.6   |
| ctranslate2 fp32          | L1 (L2 baked)       | 1719.8 / 1780.4   | 972.2 / 977.8   |
| mnn-llm fp16              | L0 (L2 baked)       | 1324.8 / 1370.5   | 1040.1 / 1095.8 |
|                           | +L1 (L2 baked)      | 563.0 / 566.8     | 287.1 / 297.2   |
| vllm-cpu fp32             | default (all baked) | 204.3 / 210.6     | 153.6 / 158.5   |

Each cell is `p50 / p99` latency in milliseconds. Every method runs 5 warmup calls then 100 timed iterations. Latency is per-call wall-clock time at batch size 1; threads are pinned to the host's physical core count.

## Legend

The columns are the two input templates: **original** (the model card's built-in Qwen3Guard system prompt, ~296-token overhead) and **test-200** (a compressed policy with ~130-token overhead).

The rows are a strictly-cumulative optimization ladder. Each `+Lk` row layers one more trick on top of the previous row within the same backend. `(L2 baked)` next to a row label means the backend bakes the lastpos lm_head trick into export, so the L0 / L1 rows already include it — there is no separate L2 row for those backends.

- **L0** — unoptimized. `tokenize → generate() decode loop (~32 tokens) → parse 'Safety: <verdict>'`. The model-card path. Per-call KV cache during the decode is on (it's what `generate()` does); none of the cross-call tricks below are.

- **L1** — `+forced-prefix`. Teacher-force `"Safety: "` and read the 3 verdict logits from one forward pass. No decode loop.

- **L2** — `+lastpos lm_head only`. Slice hidden state to last position before the vocab projection: `[B, S, H] → [B, 1, H]` then `@ [H, V]`. Skips the ~200 prompt-position vocab projections. Same trick as ChatGPT's "lm_head trim" and ORT GenAI's `prune_lm_head=true`. PyTorch exposes this via `logits_to_keep=1`. ONNX, OpenVINO, and llama.cpp bake it into export or default — those backends show no separate L2 row. Rust candle does not bake it (its `ModelForCausalLM.forward` projects the full sequence; the last-position slice happens after the matmul), so the rust-candle L0 / L1 rows are unannotated.

- **L3** — `+shared system-prompt KV cache`. Precompute the shared prefix KV once and reuse it across calls. Per-call cost shrinks to the variable-suffix forward. ONNX uses the with-past graph + IO binding; llama.cpp rewinds its context in place; Rust candle clones the primed model. PyTorch and OpenVINO don't have this mode in the bench.

A vocab-subset projection trick (project to only the 3 verdict-token rows of `lm_head`) was measured on PyTorch CPU and dropped — savings (~0.8 ms on the 150 MFLOP lm_head matmul) sit below the noise floor.
