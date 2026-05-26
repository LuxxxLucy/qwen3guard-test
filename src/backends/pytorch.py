"""PyTorch CPU backend — fp32, runs the full L0 / L1 / L2 / L3 trick ladder."""
from __future__ import annotations

from contract import L0_MAX_NEW_TOKENS

from .base import GenBackend


class PyTorchCPUBackend(GenBackend):
    runtime = "pytorch"
    SUPPORTS_L0 = True
    SUPPORTS_LAST_POS_LOGITS = True
    SUPPORTS_KV_CACHE = True

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM
        if self.threads:
            torch.set_num_threads(self.threads)
        self._torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32,
        ).to("cpu").eval()
        self._prefix_kv = None
        self._prefix_len = 0
        self.detail = f"torch {torch.__version__} threads={torch.get_num_threads()}"

    def prime_prefix(self, prefix_ids: list[int]) -> None:
        # Forward over the shared system-prompt prefix once; stash the KV as a
        # legacy tuple-of-tuples snapshot so each per-call verdict_logits()
        # rebuilds a fresh DynamicCache (DynamicCache mutates in place).
        torch = self._torch
        t = torch.tensor([prefix_ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model(input_ids=t, attention_mask=torch.ones_like(t),
                             use_cache=True)
        kv = out.past_key_values
        if hasattr(kv, "to_legacy_cache"):
            kv = kv.to_legacy_cache()
        self._prefix_kv = tuple((k.clone(), v.clone()) for k, v in kv)
        self._prefix_len = len(prefix_ids)

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        torch = self._torch
        if self.kv_cache and self._prefix_kv is not None:
            from transformers.cache_utils import DynamicCache
            suffix_ids = forced_ids[self._prefix_len:] or forced_ids[-1:]
            t = torch.tensor([suffix_ids], dtype=torch.long)
            cache = DynamicCache.from_legacy_cache(self._prefix_kv)
            attn = torch.ones((1, self._prefix_len + len(suffix_ids)),
                              dtype=torch.long)
            with torch.no_grad():
                out = self.model(input_ids=t, attention_mask=attn,
                                 past_key_values=cache, use_cache=True,
                                 logits_to_keep=1 if self.last_pos_logits else 0)
            return self._read_verdicts(out.logits[0, -1])
        t = torch.tensor([forced_ids], dtype=torch.long)
        # logits_to_keep=1 restricts the lm_head matmul to the last position
        # (the L2 trick). 0 keeps the full-sequence projection (L1 baseline).
        with torch.no_grad():
            out = self.model(input_ids=t, attention_mask=torch.ones_like(t),
                             use_cache=False,
                             logits_to_keep=1 if self.last_pos_logits else 0)
        return self._read_verdicts(out.logits[0, -1])

    def decode_l0(self, plain_ids: list[int]) -> int:
        torch = self._torch
        t = torch.tensor([plain_ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=t, attention_mask=torch.ones_like(t),
                max_new_tokens=L0_MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                use_cache=True,
            )
        return self._verdict_from_generated(out[0, len(plain_ids):].tolist())
