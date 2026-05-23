"""CPU inference backends for the Qwen3Guard-Gen L2 forced-prefix path.

One backend per runtime (pytorch / onnx / onnx-genai / openvino / llamacpp /
vllm-cpu). Each implements `verdict_logits(forced_ids)` for the timed L2
forward, plus optional `prime_prefix` (--kv-cache) and `decode_l0`
(--unoptimized). Exported artifacts come from scripts/export_gen_*.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from contract import DEFAULT_PRECISION, L0_MAX_NEW_TOKENS, RUNTIMES, Runtime


class GenBackend(ABC):
    """Base class. A backend reads the 3 verdict logits for one L2 forward."""

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
        self.detail = ""  # runtime/version string for the result JSON

    @abstractmethod
    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        raise NotImplementedError

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        """Last-position logits for the 3 verdict token ids, in VERDICT_LABELS
        order. Backends that override predict() directly (e.g. vLLM) skip this."""
        raise NotImplementedError(f"{self.runtime} backend has no verdict_logits")

    def _read_verdicts(self, last_row) -> list[float]:
        return [float(last_row[v]) for v in self.verdict_token_ids]

    def predict(self, forced_ids: list[int]) -> int:
        """Returns the verdict index (0=Safe, 1=Unsafe, 2=Controversial). In L0
        unoptimized mode runs the decode loop instead and returns 0 — L0 skips
        the verify check. bench_common.warmup_and_measure ignores the return."""
        if self.unoptimized:
            return self.decode_l0(forced_ids)
        lg = self.verdict_logits(forced_ids)
        return lg.index(max(lg))


class PyTorchCPUBackend(GenBackend):
    runtime = "pytorch"
    SUPPORTS_L0 = True
    SUPPORTS_LAST_POS_LOGITS = True

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
            self.model.generate(
                input_ids=t, attention_mask=torch.ones_like(t),
                max_new_tokens=L0_MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                use_cache=True,
            )
        return 0


class OnnxCPUBackend(GenBackend):
    runtime = "onnx"
    SUPPORTS_KV_CACHE = True
    SUPPORTS_L0 = True

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
        # Both --kv-cache and --unoptimized need the with-past graph.
        if self.kv_cache or self.unoptimized:
            self._probe_with_past()
        if self.kv_cache:
            self._io = self.session.io_binding()
            self._prefix_len = 0
            self.detail += " +kv-cache"

    def _probe_with_past(self) -> None:
        past = sorted(n for n in self.input_names if n.startswith("past_key_values."))
        if not past:
            raise SystemExit(
                "[onnx] with-past inputs not found — point --artifact at "
                "onnx_models/<model>/withpast (export_gen_onnx.py --with-past).")
        shape = {i.name: i.shape for i in self.session.get_inputs()}[past[0]]
        self._past_names = past
        self._present_names = [n.replace("past_key_values", "present") for n in past]
        self._kv_heads, self._head_dim = shape[1], shape[3]

    def prime_prefix(self, prefix_ids: list[int]) -> None:
        """Forward the shared prefix once and bind its present-KV as persistent
        IO-binding inputs. Suffix forwards reuse them with no copy."""
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
            return self._read_verdicts(row)
        arr = np.array([forced_ids], dtype=np.int64)
        feeds = {"input_ids": arr}
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = np.ones_like(arr)
        if "position_ids" in self.input_names:
            feeds["position_ids"] = np.arange(len(forced_ids), dtype=np.int64)[None, :]
        logits = self.session.run([self.output_name], feeds)[0]
        row = logits[0, -1]
        return self._read_verdicts(row)

    def decode_l0(self, plain_ids: list[int]) -> int:
        """L0: hand-rolled greedy decode over the with-past graph — initial
        forward with empty past_kv, then L0-1 per-token steps feeding the prior
        step's present_kv. Same per-call KV behaviour as generate(use_cache=True)."""
        np = self._np
        p = len(plain_ids)
        empty = np.zeros((1, self._kv_heads, 0, self._head_dim), np.float32)
        feeds: dict = {
            "input_ids": np.array([plain_ids], np.int64),
            "attention_mask": np.ones((1, p), np.int64),
            "position_ids": np.arange(p, dtype=np.int64)[None, :],
        }
        feeds.update({n: empty for n in self._past_names})
        outputs = self.session.run(
            [self.output_name, *self._present_names], feeds,
        )
        logits, past_kv = outputs[0], outputs[1:]
        for step in range(L0_MAX_NEW_TOKENS - 1):
            next_token = int(logits[0, -1].argmax())
            new_pos = p + step
            feeds = {
                "input_ids": np.array([[next_token]], np.int64),
                "attention_mask": np.ones((1, new_pos + 1), np.int64),
                "position_ids": np.array([[new_pos]], np.int64),
            }
            for name, kv in zip(self._past_names, past_kv):
                feeds[name] = kv
            outputs = self.session.run(
                [self.output_name, *self._present_names], feeds,
            )
            logits, past_kv = outputs[0], outputs[1:]
        return 0


class OnnxGenAICPUBackend(GenBackend):
    """ONNX Runtime GenAI CPU backend. Built with `prune_lm_head=true` so the
    L2 lastpos trick is baked into both L0 and L1."""
    runtime = "onnx-genai"
    SUPPORTS_L0 = True

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        import numpy as np
        import onnxruntime_genai as og
        if not artifact:
            raise SystemExit("[onnx-genai] --artifact (model build dir) is required.")
        artifact_dir = Path(artifact)
        if not (artifact_dir / "genai_config.json").exists():
            raise SystemExit(
                f"[onnx-genai] genai_config.json not found in {artifact_dir} "
                f"(run scripts/export_gen_onnx_genai.py first).")
        self._np = np
        self.model = og.Model(str(artifact_dir))
        # One Generator reused across calls; rewind_to(0) clears its KV per call.
        self._params = og.GeneratorParams(self.model)
        self._params.set_search_options(max_length=max_seq_len + 1, min_length=1)
        self._gen = og.Generator(self.model, self._params)
        self.detail = f"onnxruntime-genai {og.__version__}"

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        self._gen.rewind_to(0)
        self._gen.append_tokens(forced_ids)
        row = self._np.asarray(self._gen.get_logits()).reshape(-1)
        return self._read_verdicts(row)

    def decode_l0(self, plain_ids: list[int]) -> int:
        self._gen.rewind_to(0)
        self._gen.append_tokens(plain_ids)
        for _ in range(L0_MAX_NEW_TOKENS - 1):
            if self._gen.is_done():
                break
            self._gen.generate_next_token()
        return 0


class OpenVINOCPUBackend(GenBackend):
    runtime = "openvino"
    SUPPORTS_L0 = True

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
        return self._read_verdicts(row)

    def decode_l0(self, plain_ids: list[int]) -> int:
        torch = self._torch
        t = torch.tensor([plain_ids], dtype=torch.long)
        self.model.generate(
            input_ids=t, attention_mask=torch.ones_like(t),
            max_new_tokens=L0_MAX_NEW_TOKENS, do_sample=False, num_beams=1,
        )
        return 0


class VLLMCPUBackend(GenBackend):
    """vLLM CPU — all-tricks-baked single-row baseline. Skips the trick ladder.
    With "Safety: " forced in the prompt, the next-token argmax IS the verdict."""
    runtime = "vllm-cpu"

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        self._TokensPrompt = TokensPrompt
        self.llm = LLM(
            model=model_id, dtype="float16", enforce_eager=True,
            max_model_len=max(2048, max_seq_len + 16), disable_log_stats=True,
        )
        self._sp = SamplingParams(temperature=0.0, max_tokens=1)
        import vllm
        self.detail = f"vllm {vllm.__version__} cpu"

    def predict(self, forced_ids: list[int]) -> int:
        out = self.llm.generate(
            [self._TokensPrompt(prompt_token_ids=forced_ids)],
            self._sp, use_tqdm=False,
        )
        next_id = out[0].outputs[0].token_ids[0]
        try:
            return self.verdict_token_ids.index(next_id)
        except ValueError:
            return 0


class LlamaCppBackend(GenBackend):
    runtime = "llamacpp"
    SUPPORTS_KV_CACHE = True
    SUPPORTS_L0 = True

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        from llama_cpp import Llama
        import llama_cpp
        if not artifact or not Path(artifact).exists():
            raise SystemExit(f"[llamacpp] GGUF file not found: {artifact} "
                             f"(run scripts/export_gen_gguf.py first).")
        n_ctx = max(2048, ((max_seq_len + 256) // 256 + 1) * 256)
        # n_threads_batch governs prefill; pin to the same physical-core budget
        # as n_threads (else llama.cpp uses its logical-core default).
        self.llm = Llama(
            model_path=str(artifact), n_ctx=n_ctx,
            n_threads=self.threads or None, n_threads_batch=self.threads or None,
            n_gpu_layers=0, verbose=False,
        )
        # Read verdict logits from the context pointer directly; llama-cpp-python
        # 0.3.x doesn't fill Llama.scores, and copying n_vocab floats to read 3
        # of them is pure waste.
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
        for _ in range(L0_MAX_NEW_TOKENS):
            row = self._get_logits(self._ctx, -1)
            logits = np.ctypeslib.as_array(row, shape=(n_vocab,))
            self.llm.eval([int(logits.argmax())])
        return 0


_BACKENDS: dict[Runtime, type[GenBackend]] = dict(zip(
    RUNTIMES,
    (PyTorchCPUBackend, OnnxCPUBackend, OnnxGenAICPUBackend,
     OpenVINOCPUBackend, LlamaCppBackend, VLLMCPUBackend),
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
