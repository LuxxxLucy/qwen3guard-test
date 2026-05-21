"""CPU inference backends for the Qwen3Guard-Gen L2 forced-prefix path.

Every backend runs ONE forward over `forced_ids` (= tokenize(prompt + "Safety: "))
and reads the 3 verdict logits at the last position. No decode loop, no KV cache
— this is the L2 path REPORT_GEN settles on. Backends differ only in the runtime
that executes the forward:

  pytorch   — transformers AutoModelForCausalLM, float32 eager
  onnx      — ONNX Runtime, CPUExecutionProvider (fp32 / int8 / int4 weights)
  openvino  — OpenVINO via optimum-intel (fp32 / int8 / int4 weight compression)
  llamacpp  — llama.cpp via llama-cpp-python (GGUF f16 / q8_0 / q4_k_m)

The exported artifacts are produced by scripts/export_gen_{onnx,openvino,gguf}.py.
"""
from __future__ import annotations

from pathlib import Path


# Default precision tag when --precision is not given, per runtime.
DEFAULT_PRECISION = {
    "pytorch": "fp32",
    "onnx": "fp32",
    "openvino": "fp32",
    "llamacpp": "f16",
}

# provider/runtime string written into BenchResult.provider.
PROVIDER_TAG = {
    "pytorch": "torch-cpu",
    "onnx": "CPUExecutionProvider",
    "openvino": "OpenVINO-CPU",
    "llamacpp": "llama.cpp-cpu",
}


class GenBackend:
    """Base class. A backend reads the 3 verdict logits for one L2 forward."""

    runtime = "base"

    def __init__(self, precision: str, verdict_token_ids: list[int],
                 threads: int | None):
        self.precision = precision
        self.verdict_token_ids = verdict_token_ids
        self.threads = threads
        self.detail = ""  # runtime/version string for the result JSON

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        raise NotImplementedError

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        """Last-position logits for the 3 verdict token ids, in VERDICT_LABELS
        order. The forward pass plus this read is the timed region."""
        raise NotImplementedError

    def predict(self, forced_ids: list[int]) -> int:
        """One L2 forward plus the restricted 3-way argmax — the timed unit of
        work. Returns the verdict index (0=Safe, 1=Unsafe, 2=Controversial);
        bench_common.warmup_and_measure ignores the return, the cross-runtime
        verify check uses it."""
        lg = self.verdict_logits(forced_ids)
        return lg.index(max(lg))


class PyTorchCPUBackend(GenBackend):
    runtime = "pytorch"

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM
        if self.threads:
            torch.set_num_threads(self.threads)
        self._torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32,
        ).to("cpu").eval()
        self.detail = f"torch {torch.__version__} threads={torch.get_num_threads()}"

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        torch = self._torch
        t = torch.tensor([forced_ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model(input_ids=t, attention_mask=torch.ones_like(t),
                             use_cache=False)
        row = out.logits[0, -1]
        return [float(row[v]) for v in self.verdict_token_ids]


class OnnxCPUBackend(GenBackend):
    runtime = "onnx"

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        import numpy as np
        import onnxruntime as ort
        if not artifact:
            raise SystemExit("[onnx] --artifact (export dir) is required.")
        onnx_path = Path(artifact)
        if onnx_path.is_dir():
            onnx_path = onnx_path / "model.onnx"
        if not onnx_path.exists():
            raise SystemExit(f"[onnx] model not found: {onnx_path} "
                             f"(run scripts/export_gen_onnx.py first).")
        self._np = np
        so = ort.SessionOptions()
        if self.threads:
            so.intra_op_num_threads = self.threads
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name
        self.detail = f"onnxruntime {ort.__version__}"

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        np = self._np
        arr = np.array([forced_ids], dtype=np.int64)
        feeds = {"input_ids": arr}
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = np.ones_like(arr)
        if "position_ids" in self.input_names:
            feeds["position_ids"] = np.arange(len(forced_ids), dtype=np.int64)[None, :]
        logits = self.session.run([self.output_name], feeds)[0]
        row = logits[0, -1]
        return [float(row[v]) for v in self.verdict_token_ids]


class OpenVINOCPUBackend(GenBackend):
    runtime = "openvino"

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        import torch
        from optimum.intel import OVModelForCausalLM
        if not artifact:
            raise SystemExit("[openvino] --artifact (export dir) is required.")
        ov_dir = Path(artifact)
        if not (ov_dir / "openvino_model.xml").exists():
            raise SystemExit(f"[openvino] IR not found in {ov_dir} "
                             f"(run scripts/export_gen_openvino.py first).")
        self._torch = torch
        ov_config = {"INFERENCE_NUM_THREADS": str(self.threads)} if self.threads else None
        self.model = OVModelForCausalLM.from_pretrained(str(ov_dir), ov_config=ov_config)
        import openvino as ov
        self.detail = f"openvino {ov.__version__}"

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        torch = self._torch
        t = torch.tensor([forced_ids], dtype=torch.long)
        out = self.model(input_ids=t, attention_mask=torch.ones_like(t))
        row = out.logits[0, -1]
        return [float(row[v]) for v in self.verdict_token_ids]


class LlamaCppBackend(GenBackend):
    runtime = "llamacpp"

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        from llama_cpp import Llama
        import llama_cpp
        if not artifact or not Path(artifact).exists():
            raise SystemExit(f"[llamacpp] GGUF file not found: {artifact} "
                             f"(run scripts/export_gen_gguf.py first).")
        # n_ctx must cover the longest forced_ids sequence; round up for headroom.
        n_ctx = max(2048, ((max_seq_len + 256) // 256 + 1) * 256)
        self.llm = Llama(
            model_path=str(artifact),
            n_ctx=n_ctx,
            n_threads=self.threads or None,
            logits_all=False,
            n_gpu_layers=0,
            verbose=False,
        )
        self.detail = f"llama-cpp-python {llama_cpp.__version__} n_ctx={n_ctx}"

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        self.llm.reset()
        self.llm.eval(forced_ids)
        # llm.scores is (n_ctx, n_vocab); the last evaluated row is the readout.
        row = self.llm.scores[self.llm.n_tokens - 1]
        return [float(row[v]) for v in self.verdict_token_ids]


_BACKENDS = {
    "pytorch": PyTorchCPUBackend,
    "onnx": OnnxCPUBackend,
    "openvino": OpenVINOCPUBackend,
    "llamacpp": LlamaCppBackend,
}


def make_backend(runtime: str, precision: str | None,
                 verdict_token_ids: list[int], threads: int | None) -> GenBackend:
    if runtime not in _BACKENDS:
        raise SystemExit(f"unknown runtime {runtime!r}; "
                         f"choose from {sorted(_BACKENDS)}")
    prec = precision or DEFAULT_PRECISION[runtime]
    return _BACKENDS[runtime](prec, verdict_token_ids, threads)
