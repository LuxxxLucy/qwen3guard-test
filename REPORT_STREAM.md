# Qwen3Guard Streaming Classifier Performance Report

[[_TOC_]]

# Introduction

Qwen3Guard-Stream is a language model fine-tuned for per-token safety classification (Safe / Controversial / Unsafe).

In this report we measure its batch-size-1 inference latency on an RTX 3090 at three sizes (0.6B / 4B / 8B), identify why the shipped streaming API is an order of magnitude slower than the paper's stated design, and give the corrected incremental-forward recipe.

Originally we used the model's shipped `stream_moderate_from_ids` API, which ran at 28–85 ms per-token P99 at representative input — 7×–21× over a 4 ms/token target.
Reading the shipped streaming coroutine showed why: it does not stream. It re-prefills the full accumulated sequence on every emitted token.
The paper (Section 4.5) states Stream performs "per-token moderation without reprocessing prior tokens."
The shipped code contradicts the paper.

Replacing the shipped loop with a direct `model.forward(chunk, past_kv=cache)` call (the inference path the paper actually describes) drops per-token P99 to **< 2 ms at user_len ≤ 512**, and **< 3 ms at user_len ≤ 4096**, across all three sizes, at chunk size k=16.

## 1. Background

Our gateway sits between end users and a hosted LLM.
Every assistant response passes through an Output Security Engine that must scan the text and allow / redact / block it within a latency budget.

Qwen3Guard ships two variants.
Generative Qwen3Guard (covered in REPORT_GEN) classifies a complete response once generation finishes — whole-response latency < 200 ms P99 on our contract at user_len ≤ 512.
Streaming Qwen3Guard classifies each assistant token as it is emitted: the user sees tokens as they stream, but generation is cut the instant a token flips to Unsafe.

The latency concern for Stream differs from Gen.
For Stream, what matters is **chunk latency**: the wall time between the host LLM emitting a burst of new tokens and the classifier returning verdicts for those tokens.
The derived **per-token latency** is chunk latency divided by *k*.
Chunk latency is the extra wait the user sees; per-token is the rate the classifier must sustain to keep up with the host LLM.

The T2 feature contract is **per-token P99 < 2 ms at user_len ≤ 512 tokens**.

## 2. Qwen3Guard-Stream method

### 2.1 Architecture

Stream uses a causal Qwen3 transformer backbone — the same weights as the base Qwen3 models, fine-tuned — with four linear classification heads attached to the final hidden state.

The heads are pointwise across positions.
Position *i*'s verdict comes from the hidden state at position *i*, which by causal attention depends only on input positions ≤ *i*.

For response moderation (assistant tokens), the heads compute:

> xr       = LayerNorm(Wr_pre  · h)
> yr_risk  = Softmax(Wr_risk   · xr)       # 3-way: Safe / Unsafe / Controversial
> yr_cat   = Softmax(Wr_cat    · xr)       # 9-way: harm category

where *h* is the final hidden state at the current token.
A query head pair uses the same mechanism on user-turn tokens.

### 2.2 Incremental streaming inference

The paper (Section 4.5) describes Stream inference as:

> "Stream Qwen3Guard performs real-time, per-token moderation without reprocessing prior tokens."

This is a standard KV-cache-reusing causal forward:

1. Prefill the user prompt once — the KV cache holds N entries, and the query heads produce per-position user-turn verdicts.
2. For each new assistant chunk of *k* tokens, call `forward(chunk_ids, past_kv=cache, use_cache=True)`. The cache advances by *k* entries. Read out the response heads at the *k* new positions.
3. Repeat until end-of-response.

Each forward processes only the new tokens, not the accumulated history, so the per-step cost is one model-weight read plus attention over the current context — not a fresh prefill.

### 2.3 Chunk size k

*k* is the number of new assistant tokens a single `forward(...)` call advances.
Host LLMs rarely emit one token per SSE tick — vLLM / SGLang / TRT-LLM buffer emitted tokens into bursts, with k in the 4–16 range a common production granularity.
One forward reads the model weights once regardless of *k*, so at short context and bs=1 (weight-bandwidth-bound) one forward of *k* tokens costs close to one forward of one token in wall time.
Effective per-token latency = chunk-forward latency ÷ *k*.

## 3. Latency and optimization

The first path we tested was the shipped `stream_moderate_from_ids(new_token, role, state)` API, called in a one-token-per-call loop.
The second was a direct `Qwen3ForGuardModel.forward(chunk, past_kv=cache)` call.
Both paths run the same model weights — the difference is the streaming-loop implementation.

### 3.1 Shipped API inference

At representative input (user_len = 81 tokens), per-token P99 is:

| Size | Stock API per-token P99 | vs 4 ms target |
|---|---:|---:|
| 0.6B | 27.97 ms |  7× over |
| 4B   | 56.80 ms | 14× over |
| 8B   | 85.09 ms | 21× over |

Per-token cost grows linearly with accumulated user context (27 ms at 81 user tokens, 450 ms at 4096 on 0.6B — see Appendix A1).
For a proper KV-cache-reusing incremental forward, per-token cost should be near-flat in user context at this size (weight-bandwidth bound).
Linear scaling is the signature of re-prefilling.

Reading the shipped `stream_generate` coroutine explains the scaling:

```python
past_key_values = None
current_input_ids = input_ids                                 # prompt: N tokens

while True:
    outputs = self.forward(
        input_ids=current_input_ids.unsqueeze(0),             # (1, accumulated_length)
        past_key_values=past_key_values                       # cache also grows
    )
    past_key_values = outputs.past_key_values
    next_token_id = yield (outputs.risk_level_logits, ...)
    current_input_ids = torch.cat([
        current_input_ids,
        torch.tensor([next_token_id], device=self.device),
    ])
```

`current_input_ids` is never sliced; it accumulates the full sequence.
Each iteration passes the whole accumulated sequence to the forward, along with the growing KV cache.
The forward re-embeds the entire prefix and appends new KV entries on top of the existing ones — a fresh prefill per emitted token, not an incremental forward.

A second cost compounds this one.
The wrapping `stream_moderate_from_ids` converts the full accumulated logits tuple into Python strings and floats on every call:

```python
result = {
    "risk_level": [self.response_risk_level_map[int(i)] for i in pred_risk_idx[0]],
    "risk_prob":  [round(float(i), 2)                   for i in pred_risk_prob[0]],
    ...
}
```

Profiling 32 per-token forwards on 0.6B recorded 164 GPU→CPU synchronizing memcpy calls per forward. Each call stalls the CUDA stream between kernels.

**Stock API per emitted token (0.6B at user_len = 81):**

```
  input: one new assistant token
   → re-embed accumulated sequence (N+1 tokens)          ~17 ms
   → forward + growing KV cache                          (above)
   → pull ~(N+1) × 2 verdict scalars to Python strings   ~6 ms  (164 host syncs)
   → return per-position verdict dicts
  TOTAL ≈ 25–28 ms per token
```

### 3.2 Direct-path inference

Bypass the `stream_generate` coroutine. Call `Qwen3ForGuardModel.forward(...)` directly with a `DynamicCache`. Pass only new tokens per step. Keep verdict logits on GPU; convert to strings only when the gateway acts.

```python
from transformers.cache_utils import DynamicCache

cache = DynamicCache()
model(input_ids=uid.unsqueeze(0), past_key_values=cache, use_cache=True)   # prefill once

for chunk_ids in chunks_of(asst_tokens, k=16):
    chunk = torch.tensor([chunk_ids], device=device)
    out = model(input_ids=chunk, past_key_values=cache, use_cache=True)
    # out.risk_level_logits: (1, 16, 3)  — stays on GPU
    # argmax + map to "Safe/Unsafe/Controversial" deferred until the gateway acts
```

This is what §4.5 of the paper describes.

**Direct path per chunk (0.6B at user_len = 81, k = 16):**

```
  input: k new assistant tokens
   → forward k new tokens through backbone + heads       ~22 ms
   → 16 risk_level_logits on GPU (shape 1×16×3)
   → 16 per-position verdicts (argmax deferred)
  TOTAL ≈ 22 ms per chunk  =  1.4 ms / token
```

Stream has no generative decode loop to remove, so the optimization here differs from Gen's: we eliminate the re-prefill, eliminate the host-sync storm, and amortize one forward over *k* positions.

### 3.3 0.6B across input lengths

![Qwen3Guard-Stream-0.6B, stock API vs direct path — per-token P99 across input lengths on RTX 3090 bf16](figures/fig1_stream_sizes.png)

**Figure 1.** Per-token P99 latency on Qwen3Guard-Stream-0.6B across user-content lengths, four paths.
Stock API sits at 41–450 ms P99 — per-token cost scales linearly with user context, the re-prefill signature.
Direct path at k=1 sits at 17–19 ms P99 — flat across lengths, as expected for a weight-bandwidth-bound per-token forward at 0.6B.
Direct path at k=8 amortizes one forward over 8 positions: 2.5–2.7 ms eff per token.
Direct path at k=16 amortizes further: 1.3 ms eff per token across every measured length.

Chunking (k=1 → k=16 on the direct path) accounts for a 13–14× speedup.
Re-prefill elimination (stock k=1 → direct k=1 at 4096 tokens) accounts for a further ~24× speedup at long context.

### 3.4 Cross-size comparison

![Qwen3Guard-Stream direct path at k=16 — per-token P99 across three sizes and five lengths](figures/fig2_stream_length_0.6b.png)

**Figure 2.** Per-token P99 latency for direct-path inference at k=16, across 0.6B / 4B / 8B and user_len ∈ {81, 256, 1024, 2048, 4096}.
All three sizes stay under the 3 ms T2 target at every measured length.
0.6B is essentially flat at 1.3 ms.
4B drifts up to 2.2 ms at 4096 tokens.
8B drifts up to 2.8 ms at 4096 tokens.
The 8B / 4096 cell is the binding worst case on 3090 bf16.

### 3.5 Verdict vs T2

Feature-contract target: per-token P99 < 2 ms at user_len ≤ 512 tokens.

Direct path at k=16 per-token P99 around 512 user tokens (interpolated from measured 256 and 1024):

| Size | measured 256 | ~512 interp | measured 1024 | vs 2 ms |
|---|---:|---:|---:|---:|
| 0.6B | 1.34 | ~1.35 | 1.35 | ✓ met with 0.65 ms margin |
| 4B   | 1.60 | ~1.65 | 1.69 | ✓ met with 0.35 ms margin |
| 8B   | 1.80 | **~1.85** | 1.91 | ✓ met with 0.15 ms margin |

Under a broader scope (user_len ≤ 4096), per-token P99 stays under 3 ms at k=16 on all three sizes.
Worst case: 8B at 4096 user tokens = 2.75 ms.

### 3.6 TODO — huge-chunk classification mode

At chunk sizes k ≥ 128, the classifier stops being a per-token streamer and starts behaving like a batch classifier.
The user waits for the host LLM to buffer a whole chunk before any verdict arrives, so the zero-leak guarantee relaxes to a *k*-token-leak.
In return, amortization gets very strong: one forward over a 512-token chunk is essentially one weight-read on 0.6B.

Extrapolated from the measured k=16 and k=64 numbers:

| Size | k=512 chunk latency | k=512 per-token eff |
|---|---:|---:|
| 0.6B | ~30 ms | ~0.06 ms |
| 4B   | ~130 ms | ~0.25 ms |
| 8B   | ~270 ms | ~0.5 ms |

For 4B and 8B this path becomes compute-bound (single-forward FLOPs approach 3090 peak), and chunk latency is comparable to Gen's whole-response latency at the same length.
The deployment question is a product one: is a 512-token leak acceptable in exchange for 5–20× lower per-token cost than k=16?

**Open:** measure k ∈ {128, 256, 512} directly and add a second row to the T2 contract for the batch-classify mode.

## 4. Correctness

Three checks, run at 0.6B / 4B / 8B on the direct path:

| Check | 0.6B | 4B | 8B |
|---|---|---|---|
| Causality: verdicts at positions 0–15 do not depend on tokens at positions > 15 | 16/16 match | 16/16 | 16/16 |
| Chunking: direct k=1 verdicts ≡ direct k=16 verdicts on 64 asst tokens         | 64/64 match | 64/64 | 64/64 |
| Stock-API parity: stock k=1 verdicts ≡ direct k=1 verdicts on 64 asst tokens   | 64/64 match | 64/64 | 64/64 |

Causality test: run the model once on `asst[:64]` and once on `asst[:16]`, compare `argmax(risk_level_logits)` at positions 0..15.
In the long run those positions had 48 future tokens in the input; in the short run they had none.
Agreement proves position *i*'s verdict does not attend to positions > *i*.
Logit numerical closeness: max |diff| = 8e-2 (0.6B), 5e-2 (4B), 3e-2 (8B) — bf16 SDPA reduction-order noise; the argmax agreement is the load-bearing check.

The chunking test confirms that amortizing a forward over 16 positions produces the same per-position hidden state as 16 single-token forwards.
This follows from (1) causal attention and (2) pointwise classification heads.

The stock-API parity check is a diagnostic result.
Despite the re-prefill-with-duplicate-cache behavior, the shipped API produces the same per-position argmax verdicts as clean causal streaming.
The re-embedded prefill entries it adds to the cache wash out in attention because they duplicate the already-cached content.
Conclusion: the shipped loop is inefficient but not semantically broken.
Upstream can adopt the direct path without behavioral regressions.

## 5. Conclusion

1. The paper (§4.5) describes Stream inference as a standard causal incremental forward: per-token moderation without reprocessing prior tokens.
2. The shipped `stream_moderate_from_ids` does not implement this. It passes the full accumulated sequence alongside the growing KV cache on every iteration, which is a fresh prefill per emitted token, and it pulls per-position verdicts to Python on every call (164 host syncs per forward on 0.6B). Per-token latency is 28–85 ms P99 at representative input, 7×–21× over the 4 ms target.
3. Replacing the shipped loop with a direct `model.forward(chunk, past_kv=cache)` call removes both the re-prefill and the per-call host-sync storm. The fix is roughly twenty lines of Python and stays on the shipped checkpoint.
4. At chunk size k=16, direct-path per-token P99 is **< 2 ms at user_len ≤ 512** across 0.6B / 4B / 8B. At user_len ≤ 4096 it is < 3 ms across all three sizes. Worst case: 8B at 4096 user tokens = 2.75 ms.
5. Correctness is preserved. Per-position verdicts match between direct k=1 and direct k=16, between the direct path and the stock API, and respect causality (position *i* does not attend to positions > *i*).
6. The T2 contract can be stated as **per-token P99 < 2 ms at user_len ≤ 512** for the production-default 0.6B size, with broader scope 0.6B–8B up to 4096 user tokens available at per-token P99 < 3 ms.

## References

[Q3G] Zhao et al. *Qwen3Guard Technical Report.* arXiv:2510.14276, 2025. Model cards at <https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B> (and 4B / 8B siblings). Apache 2.0.

[PEFT] HuggingFace PEFT documentation — Token Classification with LoRA. <https://huggingface.co/docs/peft/main/en/task_guides/token-classification-lora>.

[LoRA] Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021.

[3090] NVIDIA. *GeForce RTX 3090 — product specifications.* <https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/>. 24 GB GDDR6X, ~936 GB/s peak memory bandwidth.

## Appendix

### A1 Stock-API length sweep (baseline)

The numbers that prompted the investigation. Qwen3Guard-Stream-0.6B per-token P99 via `stream_moderate_from_ids` (ms):

| user tok |  32 |  64 | 128 |  256 |  512 | 1024 | 2048 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per-token P99 | 28.9 | 29.8 | 31.6 | 41.3 | 61.5 | 110.2 | 231.4 | 450.5 |
| prefill P99  | 27.1 | 27.5 | 29.4 | 38.9 | 58.2 | 108.1 | 288.3 | 457.3 |

Per-token cost ≈ prefill cost at every length — the re-prefill signature. Same shape on 4B and 8B; tables omitted for brevity.

### A2 Direct-path length × chunk-size tables

Direct-path per-token P99 (ms), with chunk P99 in parentheses.

**Qwen3Guard-Stream-0.6B**

| user tok | k=1 | k=4 | k=8 | k=16 |
|---:|---:|---:|---:|---:|
|   81 | 18.34 (18.34) |  4.96 (19.84) |  2.50 (19.99) | **1.38** (22.04) |
|  256 | 18.60 (18.60) |  5.32 (21.27) |  2.67 (21.32) | **1.34** (21.43) |
| 1024 | 18.25 (18.25) |  5.59 (22.37) |  2.70 (21.63) | **1.35** (21.56) |
| 2048 | 17.41 (17.41) |  5.13 (20.52) |  2.57 (20.54) | **1.28** (20.41) |
| 4096 | 19.06 (19.06) |  5.35 (21.39) |  2.53 (20.25) | **1.29** (20.64) |

**Qwen3Guard-Stream-4B**

| user tok | k=1 | k=4 | k=8 | k=16 |
|---:|---:|---:|---:|---:|
|   81 | 25.57 (25.57) |  6.79 (27.14) |  3.29 (26.33) | **1.69** (27.00) |
|  256 | 24.83 (24.83) |  6.78 (27.12) |  3.56 (28.46) | **1.60** (25.54) |
| 1024 | 24.28 (24.28) |  6.52 (26.06) |  3.22 (25.78) | **1.69** (26.98) |
| 2048 | 24.02 (24.02) |  6.67 (26.67) |  3.42 (27.38) | **1.64** (26.31) |
| 4096 | 24.18 (24.18) |  8.50 (33.98) |  4.26 (34.12) | **2.17** (34.66) |

**Qwen3Guard-Stream-8B**

| user tok | k=1 | k=4 | k=8 | k=16 |
|---:|---:|---:|---:|---:|
|   81 | 24.93 (24.93) |  6.88 (27.54) |  3.44 (27.56) | **1.74** (27.78) |
|  256 | 24.35 (24.35) |  6.93 (27.72) |  3.44 (27.52) | **1.80** (28.84) |
| 1024 | 24.57 (24.57) |  7.25 (29.00) |  3.71 (29.64) | **1.91** (30.55) |
| 2048 | 25.12 (25.12) |  8.36 (33.43) |  4.23 (33.81) | **2.21** (35.40) |
| 4096 | 26.78 (26.78) | 10.52 (42.06) |  5.32 (42.59) | **2.75** (44.04) |

### A3 Hardware and measurement setup

NVIDIA RTX 3090 (24 GB GDDR6X, ~936 GB/s HBM), CUDA 12.4, torch 2.6.0, bfloat16, batch size 1.

Length sweep: 2 warmup + 5 timed streaming runs of 64 assistant tokens per (user_len, k) cell.
Sample counts per cell: 320 at k=1, 80 at k=4, 40 at k=8, 20 at k=16.
Representative input (user_len = 81) bench: 3 warmup + 50 timed prefill iterations + 2000 timed per-token iterations per cell.
The timed region includes the classification heads (risk + category, response-side) for all direct-path numbers.

### A4 Correctness protocol

Causality: forward once on `asst_tokens[:64]`, once on `asst_tokens[:16]`; compare `argmax(risk_level_logits)` at positions 0..15. Pass = 16/16 match.
Chunking: full verdict sequence from direct k=1 compared to direct k=16 over the same 64 asst tokens. Pass = 64/64 match.
Stock-API parity: the same 64 asst tokens through `stream_moderate_from_ids` one token at a time; compare last-element-per-call to direct k=1. Pass = 64/64 match.
All three checks passed at all three sizes.

### A5 Open questions about the Stream training pipeline

The Qwen3Guard paper describes the classification-head architecture, the boundary-token labelling pipeline, and the backbone identity, but is silent on three points that would matter for a reproduction attempt:

- Whether the transformer backbone is frozen during Stream training, or jointly updated with the heads.
- The Stream-specific sample count (the 1.19 M figure published in §4.2 applies to Gen only).
- Whether the Stream fine-tune is full-parameter or adapter-based.

The inference-side numbers reported in §3 are independent of these training-pipeline gaps.
