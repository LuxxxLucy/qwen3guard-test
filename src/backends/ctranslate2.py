"""CTranslate2 CPU backend. Single whole-runtime row; the converter bakes L2
lastpos via `forward_batch`'s last-position slice and L1 via one forward over
the forced prefix.
"""
from __future__ import annotations

from pathlib import Path

from contract import L0_MAX_NEW_TOKENS

from .base import GenBackend


class CTranslate2CPUBackend(GenBackend):
    runtime = "ctranslate2"
    SUPPORTS_L0 = True

    _PREC = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16",
             "int8": "int8"}

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        import numpy as np
        import ctranslate2
        from transformers import AutoTokenizer
        if not artifact:
            raise SystemExit("[ctranslate2] --artifact (converted model dir) is required.")
        adir = Path(artifact)
        if not (adir / "model.bin").exists():
            raise SystemExit(
                f"[ctranslate2] model.bin not found in {adir} "
                f"(run src/exporters/ctranslate2.py first).")
        compute = self._PREC.get(self.precision, self.precision)
        self._np = np
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self.generator = ctranslate2.Generator(
            str(adir), device="cpu", compute_type=compute,
            intra_threads=self.threads or 16, inter_threads=1,
        )
        self.detail = f"ctranslate2 {ctranslate2.__version__} compute={compute}"

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        tokens = self._tok.convert_ids_to_tokens(forced_ids)
        out = self.generator.forward_batch([tokens])
        arr = self._np.array(out)
        return self._read_verdicts(arr[0, -1])

    def decode_l0(self, plain_ids: list[int]) -> int:
        tokens = self._tok.convert_ids_to_tokens(plain_ids)
        results = self.generator.generate_batch(
            [tokens], max_length=L0_MAX_NEW_TOKENS, sampling_topk=1,
            beam_size=1, include_prompt_in_result=False,
        )
        gen_ids = self._tok.convert_tokens_to_ids(results[0].sequences[0])
        return self._verdict_from_generated(gen_ids)
