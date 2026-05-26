"""vLLM CPU — all-tricks-baked single-row baseline. Skips the trick ladder.

With "Safety: " forced in the prompt the next-token argmax IS the verdict;
`verdict_logits` reads the 3 logits via SamplingParams(logprobs=K) so the
verify gate can compare against the PyTorch fp32 oracle.
"""
from __future__ import annotations

from .base import GenBackend


class VLLMCPUBackend(GenBackend):
    runtime = "vllm-cpu"

    # Precision mapping: vLLM's `dtype` argument. fp32 first per the bench plan;
    # vLLM may refuse it on some kernels — the requested dtype is kept explicit
    # so the result JSON labels the row instead of silently auto-coercing.
    _PREC = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    # Top-K logprobs requested per step. vLLM 0.13.0 caps it at 20 by default
    # (max_logprobs in EngineCore). With "Safety: " forced, Safe / Unsafe /
    # Controversial dominate the next-token distribution, so 20 has plenty of
    # headroom over the 3 verdict tokens.
    _TOPK_LOGPROBS = 20

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        self._TokensPrompt = TokensPrompt
        dtype = self._PREC.get(self.precision, self.precision)
        self.llm = LLM(
            model=model_id, dtype=dtype, enforce_eager=True,
            max_model_len=max(2048, max_seq_len + 16), disable_log_stats=True,
        )
        self._sp_predict = SamplingParams(temperature=0.0, max_tokens=1)
        self._sp_logits = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=self._TOPK_LOGPROBS,
        )
        import vllm
        self.detail = f"vllm {vllm.__version__} cpu dtype={dtype}"

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        """Top-K logprobs at the first generated position, projected onto the
        3 verdict token ids. Returns raw logprobs (post-softmax log space) —
        the verify gate only consumes argmax for vllm-cpu, so a monotonic
        transform of the logits is fine. Missing-from-top-K is treated as -inf."""
        out = self.llm.generate(
            [self._TokensPrompt(prompt_token_ids=forced_ids)],
            self._sp_logits, use_tqdm=False,
        )
        lp_map = out[0].outputs[0].logprobs[0]
        vals: list[float] = []
        for tid in self.verdict_token_ids:
            entry = lp_map.get(tid)
            if entry is None:
                vals.append(float("-inf"))
            else:
                vals.append(float(entry.logprob))
        return vals

    def predict(self, forced_ids: list[int]) -> int:
        out = self.llm.generate(
            [self._TokensPrompt(prompt_token_ids=forced_ids)],
            self._sp_predict, use_tqdm=False,
        )
        next_id = out[0].outputs[0].token_ids[0]
        try:
            return self.verdict_token_ids.index(next_id)
        except ValueError:
            return 0
