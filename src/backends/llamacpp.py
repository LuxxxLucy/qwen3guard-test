"""llama.cpp CPU backend via llama-cpp-python. Reads verdict logits from the
context pointer directly to avoid copying n_vocab floats per call.
"""
from __future__ import annotations

from pathlib import Path

from contract import L0_MAX_NEW_TOKENS

from .base import GenBackend


class LlamaCppBackend(GenBackend):
    runtime = "llamacpp"
    SUPPORTS_KV_CACHE = True
    SUPPORTS_L0 = True

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        from llama_cpp import Llama
        import llama_cpp
        if not artifact or not Path(artifact).exists():
            raise SystemExit(f"[llamacpp] GGUF file not found: {artifact} "
                             f"(run src/exporters/gguf.py first).")
        n_ctx = max(2048, ((max_seq_len + 256) // 256 + 1) * 256)
        # n_threads_batch governs prefill; pin to the same physical-core budget
        # as n_threads (else llama.cpp uses its logical-core default).
        self.llm = Llama(
            model_path=str(artifact), n_ctx=n_ctx,
            n_threads=self.threads or None, n_threads_batch=self.threads or None,
            n_gpu_layers=0, verbose=False,
        )
        # llama-cpp-python 0.3.x doesn't fill Llama.scores, and copying n_vocab
        # floats to read 3 of them is pure waste.
        self._ctx = self.llm._ctx.ctx
        self._get_logits = llama_cpp.llama_get_logits_ith
        self._prefix_len = 0
        self.detail = (f"llama-cpp-python {llama_cpp.__version__} n_ctx={n_ctx}"
                       + (" +kv-cache" if self.kv_cache else ""))

    def prime_prefix(self, prefix_ids: list[int]) -> None:
        self.llm.reset()
        self.llm.eval(prefix_ids)
        self._prefix_len = self.llm.n_tokens

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        if self.kv_cache:
            # Rewind to the cached prefix; eval() then evicts the prior suffix
            # KV in place (kv_cache_seq_rm) and appends only this request's.
            self.llm.n_tokens = self._prefix_len
            self.llm.eval(forced_ids[self._prefix_len:])
        else:
            self.llm.reset()
            self.llm.eval(forced_ids)
        row = self._get_logits(self._ctx, -1)
        return self._read_verdicts(row)

    def decode_l0(self, plain_ids: list[int]) -> int:
        import numpy as np
        self.llm.reset()
        self.llm.eval(plain_ids)
        n_vocab = self.llm.n_vocab()
        generated: list[int] = []
        # eval(plain_ids) is the prefill (counts as 1 forward); decode the
        # remaining L0_MAX_NEW_TOKENS-1 so the total matches the other backends.
        for _ in range(L0_MAX_NEW_TOKENS - 1):
            row = self._get_logits(self._ctx, -1)
            logits = np.ctypeslib.as_array(row, shape=(n_vocab,))
            tok = int(logits.argmax())
            generated.append(tok)
            self.llm.eval([tok])
        return self._verdict_from_generated(generated)
