"""Backend protocol: each runtime reads the 3 verdict logits for one L2 forward.

Capability flags drive the trick-ladder gating in `make_backend`. Backends opt
in by setting the class-level flags; the factory raises if a flag is requested
on a backend that doesn't support it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class GenBackend(ABC):

    SUPPORTS_KV_CACHE: bool = False
    SUPPORTS_L0: bool = False
    SUPPORTS_LAST_POS_LOGITS: bool = False
    runtime: str = "base"

    def __init__(self, precision: str, verdict_token_ids: list[int],
                 threads: int | None, kv_cache: bool = False,
                 unoptimized: bool = False, last_pos_logits: bool = False):
        self.precision = precision
        self.verdict_token_ids = verdict_token_ids
        self.threads = threads
        self.kv_cache = kv_cache
        self.unoptimized = unoptimized
        self.last_pos_logits = last_pos_logits
        self.detail = ""

    @abstractmethod
    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        raise NotImplementedError

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        raise NotImplementedError(f"{self.runtime} backend has no verdict_logits")

    def _read_verdicts(self, last_row) -> list[float]:
        return [float(last_row[v]) for v in self.verdict_token_ids]

    def predict(self, forced_ids: list[int]) -> int:
        """Returns 0=Safe, 1=Unsafe, 2=Controversial. In L0 mode runs the
        decode loop; the implementation must return the index of the first
        verdict-token produced (or -1 if no verdict in L0_MAX_NEW_TOKENS)."""
        if self.unoptimized:
            return self.decode_l0(forced_ids)
        lg = self.verdict_logits(forced_ids)
        return lg.index(max(lg))

    def _verdict_from_generated(self, token_ids) -> int:
        vmap = {t: i for i, t in enumerate(self.verdict_token_ids)}
        for t in token_ids:
            if int(t) in vmap:
                return vmap[int(t)]
        return -1
