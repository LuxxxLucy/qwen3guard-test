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

The llama.cpp and ONNX backends have a system-prompt KV-cache mode
(kv_cache=True): prime_prefix() evaluates the shared prefix once, then
verdict_logits() forwards only the variable suffix against the cached prefix
KV. llama.cpp rewinds its context in place; ONNX rebinds the prefix KV through
an IO binding. Neither re-feeds or copies the prefix per request.
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
                 threads: int | None, kv_cache: bool = False):
        self.precision = precision
        self.verdict_token_ids = verdict_token_ids
        self.threads = threads
        self.kv_cache = kv_cache
        self.detail = ""  # runtime/version string for the result JSON

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        raise NotImplementedError

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        """Last-position logits for the 3 verdict token ids, in VERDICT_LABELS
        order. The forward pass plus this read is the timed region."""
        raise NotImplementedError

    def prime_prefix(self, prefix_ids: list[int]) -> None:
        """KV-cache mode: evaluate the shared system-prompt prefix once so
        verdict_logits can reuse it. Only the llama.cpp backend implements it."""
        raise NotImplementedError(f"{self.runtime} backend has no KV-cache mode")

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
        if self.kv_cache:
            self._setup_kv_cache()

    def _setup_kv_cache(self) -> None:
        """KV-cache mode needs a with-past graph (past_key_values.* inputs).
        Record the KV-head shape and pre-make the IO binding."""
        past = sorted(n for n in self.input_names if n.startswith("past_key_values."))
        if not past:
            raise SystemExit(
                "[onnx] --kv-cache needs a with-past export — point --artifact "
                "at onnx_models/<model>/withpast (export_gen_onnx.py --with-past).")
        shape = {i.name: i.shape for i in self.session.get_inputs()}[past[0]]
        self._past_names = past
        self._present_names = [n.replace("past_key_values", "present") for n in past]
        self._kv_heads, self._head_dim = shape[1], shape[3]
        self._io = self.session.io_binding()
        self._prefix_len = 0
        self.detail += " +kv-cache"

    def prime_prefix(self, prefix_ids: list[int]) -> None:
        """Forward the shared prefix once and bind its present-KV as persistent
        IO-binding inputs. Each per-request suffix forward reuses these bound
        tensors — the prefix KV is never re-fed or copied per call."""
        import onnxruntime as ort
        np = self._np
        p = len(prefix_ids)
        empty = np.zeros((1, self._kv_heads, 0, self._head_dim), np.float32)
        feeds = {"input_ids": np.array([prefix_ids], np.int64),
                 "attention_mask": np.ones((1, p), np.int64),
                 "position_ids": np.arange(p, dtype=np.int64)[None, :]}
        feeds.update({n: empty for n in self._past_names})
        present = self.session.run(self._present_names, feeds)
        self._prefix_kv = [ort.OrtValue.ortvalue_from_numpy(kv) for kv in present]
        for name, ov in zip(self._past_names, self._prefix_kv):
            self._io.bind_ortvalue_input(name, ov)
        self._prefix_len = p

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        np = self._np
        if self.kv_cache:
            suffix = forced_ids[self._prefix_len:]
            total = self._prefix_len + len(suffix)
            self._io.bind_cpu_input("input_ids", np.array([suffix], np.int64))
            self._io.bind_cpu_input("attention_mask", np.ones((1, total), np.int64))
            self._io.bind_cpu_input(
                "position_ids",
                np.arange(self._prefix_len, total, dtype=np.int64)[None, :])
            self._io.bind_output(self.output_name)
            self.session.run_with_iobinding(self._io)
            row = self._io.get_outputs()[0].numpy()[0, -1]
            return [float(row[v]) for v in self.verdict_token_ids]
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
            n_gpu_layers=0,
            verbose=False,
        )
        # Read the verdict logits straight from the context's last-position
        # logits pointer. eval() does not populate the Llama.scores numpy
        # buffer in llama-cpp-python 0.3.x (it stays zero); and copying that
        # buffer — n_vocab floats — to read 3 of them would be pure waste.
        self._ctx = self.llm._ctx.ctx
        self._get_logits = llama_cpp.llama_get_logits_ith
        self._prefix_len = 0
        self.detail = (f"llama-cpp-python {llama_cpp.__version__} n_ctx={n_ctx}"
                       + (" +kv-cache" if self.kv_cache else ""))

    def prime_prefix(self, prefix_ids: list[int]) -> None:
        """Evaluate the shared system-prompt prefix once. Its KV stays resident
        in the context at positions 0..len-1; verdict_logits then rewinds to
        this point per call — the prefix is never recomputed or copied."""
        self.llm.reset()
        self.llm.eval(prefix_ids)
        self._prefix_len = self.llm.n_tokens

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        if self.kv_cache:
            # Rewind the token counter to the cached prefix. eval() then drops
            # the previous request's suffix KV in place (kv_cache_seq_rm) and
            # appends only this request's suffix — the prefix KV is reused as-is.
            self.llm.n_tokens = self._prefix_len
            self.llm.eval(forced_ids[self._prefix_len:])
        else:
            self.llm.reset()
            self.llm.eval(forced_ids)
        row = self._get_logits(self._ctx, -1)
        return [float(row[v]) for v in self.verdict_token_ids]


_BACKENDS = {
    "pytorch": PyTorchCPUBackend,
    "onnx": OnnxCPUBackend,
    "openvino": OpenVINOCPUBackend,
    "llamacpp": LlamaCppBackend,
}


def make_backend(runtime: str, precision: str | None,
                 verdict_token_ids: list[int], threads: int | None,
                 kv_cache: bool = False) -> GenBackend:
    if runtime not in _BACKENDS:
        raise SystemExit(f"unknown runtime {runtime!r}; "
                         f"choose from {sorted(_BACKENDS)}")
    if kv_cache and runtime not in ("llamacpp", "onnx"):
        raise SystemExit(f"--kv-cache is implemented for --runtime llamacpp "
                         f"and onnx only, not {runtime!r}.")
    prec = precision or DEFAULT_PRECISION[runtime]
    return _BACKENDS[runtime](prec, verdict_token_ids, threads, kv_cache)
