"""CPU inference backends for the Qwen3Guard-Gen L2 forced-prefix path.

One module per runtime. Each backend implements `verdict_logits(forced_ids)`
for the timed L2 forward, plus optional `prime_prefix` (--kv-cache) and
`decode_l0` (--unoptimized). Exported artifacts come from src/exporters/.

To add a new backend:
1. Add a module here that subclasses `base.GenBackend` and implements `load`
   + `verdict_logits` (+ optional `prime_prefix` / `decode_l0`).
2. Register the class in `_BACKENDS` below.
3. Add the runtime name to `RUNTIMES` in `src/contract.py` so capability flags
   and CLI choices pick it up.
"""
from __future__ import annotations

from contract import DEFAULT_PRECISION, RUNTIMES, Runtime

from .base import GenBackend
from .ctranslate2 import CTranslate2CPUBackend
from .llamacpp import LlamaCppBackend
from .mnn import MNNLLMBackend
from .onnx import OnnxCPUBackend
from .pytorch import PyTorchCPUBackend
from .vllm_cpu import VLLMCPUBackend

__all__ = [
    "GenBackend",
    "make_backend",
    "PyTorchCPUBackend",
    "OnnxCPUBackend",
    "VLLMCPUBackend",
    "LlamaCppBackend",
    "CTranslate2CPUBackend",
    "MNNLLMBackend",
]

_BACKENDS: dict[Runtime, type[GenBackend]] = dict(zip(
    RUNTIMES,
    (PyTorchCPUBackend, OnnxCPUBackend, LlamaCppBackend, VLLMCPUBackend,
     CTranslate2CPUBackend, MNNLLMBackend),
    strict=True,
))


def make_backend(runtime: str, precision: str | None,
                 verdict_token_ids: list[int], threads: int | None,
                 kv_cache: bool = False, unoptimized: bool = False,
                 last_pos_logits: bool = False) -> GenBackend:
    cls = _BACKENDS.get(runtime)
    if cls is None:
        raise SystemExit(f"unknown runtime {runtime!r}; choose from {sorted(_BACKENDS)}")
    requested = {
        "--kv-cache":        (kv_cache,        "SUPPORTS_KV_CACHE"),
        "--unoptimized":     (unoptimized,     "SUPPORTS_L0"),
        "--last-pos-logits": (last_pos_logits, "SUPPORTS_LAST_POS_LOGITS"),
    }
    for flag, (on, cap) in requested.items():
        if on and not getattr(cls, cap):
            raise SystemExit(f"{flag} is not supported by --runtime {runtime}.")
    prec = precision or DEFAULT_PRECISION[runtime]
    return cls(prec, verdict_token_ids, threads, kv_cache, unoptimized, last_pos_logits)
