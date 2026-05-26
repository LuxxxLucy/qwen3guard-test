"""MNN-LLM CPU backend (Alibaba's Arm-tuned LLM runtime).

Single whole-runtime row; `forward(input_ids)` returns last-position logits by
default (L2 lastpos baked), so L1 is one forced-prefix forward + 3 verdict-slot
reads. L0 uses the engine's `response()` because pymnn 3.5.0's `generate()`
returns an empty list in this build.
"""
from __future__ import annotations

from pathlib import Path

from contract import L0_MAX_NEW_TOKENS

from .base import GenBackend


class MNNLLMBackend(GenBackend):
    runtime = "mnn"
    SUPPORTS_L0 = True

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        import numpy as np
        import MNN.llm as mnn_llm
        import MNN as mnn_pkg
        from transformers import AutoTokenizer
        if not artifact:
            raise SystemExit("[mnn] --artifact (converted model dir) is required.")
        adir = Path(artifact)
        cfg = adir / "config.json"
        if not cfg.exists():
            raise SystemExit(
                f"[mnn] config.json not found in {adir} "
                f"(run src/exporters/mnn.py first).")
        self._np = np
        # The HF tokenizer is the canonical detokenizer for the L0 decode path
        # — MNN's `response()` returns generated text, which we then map to a
        # verdict index via the shared extract_verdict() helper.
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self.model = mnn_llm.create(str(cfg))
        # Threads + precision pinned per call. precision="high" keeps fp32
        # accumulators on the CPU backend; weights are still fp16 (the highest
        # precision MNN-LLM's converter writes for the language path —
        # mnn_converter.py:349 asserts 1/2/4/8 for the quantized branch).
        # max_new_tokens caps the L0 decode budget. pymnn 3.5.0's `response()`
        # reads it from config; `generate()` in this build is non-functional
        # so L0 goes through response()+detokenize.
        self.model.set_config({
            "thread_num": self.threads or 16,
            "precision": "high",
            "memory": "high",
            "max_new_tokens": L0_MAX_NEW_TOKENS,
        })
        self.model.load()
        version = getattr(mnn_pkg, "version", lambda: "?")() if hasattr(mnn_pkg, "version") else "?"
        self.detail = (f"mnn-llm {version} weights={self.precision} "
                       f"accum=fp32 threads={self.threads or 16}")

    def _logits_to_numpy(self, logits_var) -> "list[float]":
        # MNN Var -> numpy via .read(). For default forward (all_logits=false),
        # the shape is [1, 1, V] containing the last-position row directly.
        np = self._np
        arr = np.array(logits_var.read())
        return arr.reshape(-1)

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        # reset() clears the KV cache from any prior call; generate_init() arms
        # a fresh prefill so seq_len_index is correct.
        self.model.reset()
        self.model.generate_init()
        logits_var = self.model.forward([int(i) for i in forced_ids])
        row = self._logits_to_numpy(logits_var)
        return self._read_verdicts(row)

    def decode_l0(self, plain_ids: list[int]) -> int:
        from gen_common import VERDICT_LABELS, extract_verdict
        self.model.reset()
        self.model.generate_init()
        prompt = self._tok.decode(plain_ids, skip_special_tokens=False)
        out_text = self.model.response(prompt, False) or ""
        verdict = extract_verdict(out_text)
        try:
            return VERDICT_LABELS.index(verdict)
        except ValueError:
            return -1
