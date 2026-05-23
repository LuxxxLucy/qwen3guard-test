"""Single source of truth for the qwen3guard-test vocabulary.

Opt-level / runtime / template / precision string literals live here. Every
other module imports from `contract`; "L2-lastpos" is the sentinel string
covered by the grep gate — it must not appear outside this file."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

# --- opt levels ------------------------------------------------------------
# Strictly cumulative ladder. Each level adds one trick on top of the previous.
# Backends that bake a trick into export (e.g. ONNX/OV bake lastpos) skip the
# corresponding stand-alone row; the bake is annotated in the report legend.
#
#   L0  unoptimized — generate() decode loop (~32 tokens), parse "Safety: <v>".
#   L1  +forced-prefix — one forward over `prompt + "Safety: "`, read 3 logits.
#   L2  +lastpos lm_head — only project the last position to vocab.
#   L3  +shared system-prompt KV cache — precompute prefix KV once, reuse.
#
# A vocab-subset projection trick (project only the 3 verdict-token rows of
# lm_head) was measured on PyTorch CPU and dropped — theoretical savings
# (~0.8 ms over the ~150 MFLOP lm_head) sit below the noise floor and the
# extra Python path made the wall-clock worse. See plan.md "Dropped".
OptLevel = Literal["L0", "L1", "L2", "L3"]
OPT_LEVELS: tuple[OptLevel, ...] = ("L0", "L1", "L2", "L3")

# --- runtimes --------------------------------------------------------------
Runtime = Literal["pytorch", "onnx", "onnx-genai", "openvino", "llamacpp", "vllm-cpu"]
RUNTIMES: tuple[Runtime, ...] = (
    "pytorch", "onnx", "onnx-genai", "openvino", "llamacpp", "vllm-cpu",
)

DEFAULT_PRECISION: dict[Runtime, str] = {
    "pytorch":    "fp32",
    "onnx":       "fp32",
    "onnx-genai": "fp32",
    "openvino":   "fp16",
    "llamacpp":   "f16",
    "vllm-cpu":   "fp16",
}

PROVIDER_TAG: dict[Runtime, str] = {
    "pytorch":    "torch-cpu",
    "onnx":       "CPUExecutionProvider",
    "onnx-genai": "ort-genai-cpu",
    "openvino":   "OpenVINO-CPU",
    "llamacpp":   "llama.cpp-cpu",
    "vllm-cpu":   "vllm-cpu",
}

# --- templates -------------------------------------------------------------
Template = Literal["original", "test-200"]
TEMPLATES: tuple[Template, ...] = ("original", "test-200")

# --- L0 / decode loop ------------------------------------------------------
L0_MAX_NEW_TOKENS = 32

# --- result-JSON extra schema ---------------------------------------------
@dataclass(frozen=True)
class ResultExtra:
    """Typed view of BenchResult.extra."""
    mode: str
    opt_level: OptLevel
    runtime: Runtime | None = None
    precision: str | None = None
    threads: int | None = None
    runtime_detail: str | None = None
    template: Template | None = None
    kv_cache: bool | None = None
    target_user_tokens: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
