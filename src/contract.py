"""Single source of truth for the qwen3guard-test vocabulary.

Opt-level / runtime / template / precision string literals live here. Every
other module imports from `contract`; "L2-lastpos" is the sentinel string
covered by the grep gate — it must not appear outside this file."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

# --- opt levels ------------------------------------------------------------
OptLevel = Literal["LP", "L0", "L1", "L2", "L2-lastpos", "L3"]
OPT_LEVELS: tuple[OptLevel, ...] = ("LP", "L0", "L1", "L2", "L2-lastpos", "L3")

# --- runtimes --------------------------------------------------------------
Runtime = Literal["pytorch", "onnx", "openvino", "llamacpp"]
RUNTIMES: tuple[Runtime, ...] = ("pytorch", "onnx", "openvino", "llamacpp")

DEFAULT_PRECISION: dict[Runtime, str] = {
    "pytorch":  "fp32",
    "onnx":     "fp32",
    "openvino": "fp16",
    "llamacpp": "f16",
}

PROVIDER_TAG: dict[Runtime, str] = {
    "pytorch":  "torch-cpu",
    "onnx":     "CPUExecutionProvider",
    "openvino": "OpenVINO-CPU",
    "llamacpp": "llama.cpp-cpu",
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
    user_tokens_median: int | None = None
    n_distinct_samples: int | None = None
    nominal_template_overhead_tokens: int | None = None
    cacheable_head_tokens: int | None = None
    total_input_tokens_median: int | None = None
    chat_template_applied: bool | None = None
    prefix_cache: bool | None = None
    forced_prefix: bool | None = None
    last_pos_logits: bool | None = None
    compile: bool | None = None
    prefill_only: bool | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
