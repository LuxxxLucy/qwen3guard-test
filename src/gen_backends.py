"""CPU inference backends for the Qwen3Guard-Gen L2 forced-prefix path.

Every backend runs ONE forward over `forced_ids` (= tokenize(prompt + "Safety: "))
and reads the 3 verdict logits at the last position. No decode loop, no KV cache
— this is the optimized L2 path. Backends differ only in the runtime that
executes the forward:

  pytorch   — transformers AutoModelForCausalLM, float32 eager
  onnx      — ONNX Runtime, CPUExecutionProvider (fp32 / int8 / int4 weights)
  openvino  — OpenVINO via optimum-intel (fp16 / int8 / int4 weight compression)
  llamacpp  — llama.cpp via llama-cpp-python (GGUF f16 / q8_0 / q4_k_m)

The exported artifacts are produced by scripts/export_gen_{onnx,openvino,gguf}.py.

The llama.cpp and ONNX backends have a system-prompt KV-cache mode
(kv_cache=True): prime_prefix() evaluates the shared prefix once, then
verdict_logits() forwards only the variable suffix against the cached prefix
KV. llama.cpp rewinds its context in place; ONNX rebinds the prefix KV through
an IO binding. Neither re-feeds or copies the prefix per request.
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

    @abstractmethod
    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        """Last-position logits for the 3 verdict token ids, in VERDICT_LABELS
        order. The forward pass plus this read is the timed region."""
        raise NotImplementedError

    def prime_prefix(self, prefix_ids: list[int]) -> None:
        """KV-cache mode: evaluate the shared system-prompt prefix once so
        verdict_logits can reuse it. Only the llama.cpp backend implements it."""
        raise NotImplementedError(f"{self.runtime} backend has no KV-cache mode")

    def decode_l0(self, plain_ids: list[int]) -> int:
        """L0 unoptimized mode: greedy generate() decode loop of
        L0_MAX_NEW_TOKENS tokens — the model-card path. The full decode is the
        timed unit; the returned verdict index satisfies predict()'s interface,
        but L0 skips the verify check."""
        raise NotImplementedError(f"{self.runtime} backend has no L0 mode")

    def predict(self, forced_ids: list[int]) -> int:
        """The timed unit of work. In L2 mode (default): one forward over
        `forced_ids` plus the restricted 3-way argmax — returns the verdict
        index (0=Safe, 1=Unsafe, 2=Controversial). In L0 unoptimized mode:
        a greedy decode loop over `forced_ids` (the plain prompt); returns 0,
        bench_gen_cpu skips the verify check.
        bench_common.warmup_and_measure ignores the return either way."""
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
        with torch.no_grad():
            # last_pos_logits: logits_to_keep=1 projects only the last position
            # to the 151,936 vocab. The L2 path reads just that row, so
            # projecting the prompt positions is wasted. Off => logits_to_keep=0
            # (the full-sequence projection), kept as the benchmark's before-row.
            out = self.model(input_ids=t, attention_mask=torch.ones_like(t),
                             use_cache=False,
                             logits_to_keep=1 if self.last_pos_logits else 0)
        row = out.logits[0, -1]
        return [float(row[v]) for v in self.verdict_token_ids]

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
        # Both --kv-cache (prefix-KV trick) and --unoptimized (L0 decode loop)
        # need the with-past graph. Probe once; the kv-cache mode also wires
        # up an IO binding for the persistent prefix.
        if self.kv_cache or self.unoptimized:
            self._probe_with_past()
        if self.kv_cache:
            self._setup_kv_cache()

    def _probe_with_past(self) -> None:
        """Record past_key_values.* input names + KV-head shape from the
        with-past export. Both --kv-cache and --unoptimized depend on this."""
        past = sorted(n for n in self.input_names if n.startswith("past_key_values."))
        if not past:
            raise SystemExit(
                "[onnx] with-past inputs not found — point --artifact at "
                "onnx_models/<model>/withpast (export_gen_onnx.py --with-past).")
        shape = {i.name: i.shape for i in self.session.get_inputs()}[past[0]]
        self._past_names = past
        self._present_names = [n.replace("past_key_values", "present") for n in past]
        self._kv_heads, self._head_dim = shape[1], shape[3]

    def _setup_kv_cache(self) -> None:
        """KV-cache mode pre-makes the IO binding for the persistent prefix."""
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

    def decode_l0(self, plain_ids: list[int]) -> int:
        """L0 unoptimized: hand-rolled greedy decode over the with-past graph.
        Matches what an ONNX user writes for batch=1 generation — initial
        forward over the full prompt with empty past_kv, then L0-1 per-token
        steps feeding the previous step's present_kv as past_kv. Same per-call
        KV behaviour as model.generate(use_cache=True) on the other backends."""
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
    """ONNX Runtime GenAI (microsoft/onnxruntime-genai) CPU backend.

    Build the model with `python -m onnxruntime_genai.models.builder` and
    `prune_lm_head=true` so the LM head only computes last-token logits during
    prefill — that bakes the L2 (lastpos) trick into the L0 baseline. Forward
    one pass over `forced_ids` via the Generator's `append_tokens` +
    `generate_next_token`; the per-call logits at the last position are then
    read via `get_logits()` and the 3 verdict ids extracted.
    """
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
        self._og = og
        self.model = og.Model(str(artifact_dir))
        # One generator is reused for every call. `rewind_to(0)` clears its
        # per-call KV before each prefill — cheaper than allocating a new one.
        self._params = og.GeneratorParams(self.model)
        self._params.set_search_options(max_length=max_seq_len + 1, min_length=1)
        self._gen = og.Generator(self.model, self._params)
        self.detail = f"onnxruntime-genai {og.__version__}"

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        # rewind_to(0) discards any KV from the previous call and starts fresh.
        self._gen.rewind_to(0)
        self._gen.append_tokens(forced_ids)
        # append_tokens runs the prefill forward; get_logits then yields the
        # last-position logits over the full vocab.
        row = self._np.asarray(self._gen.get_logits()).reshape(-1)
        return [float(row[v]) for v in self.verdict_token_ids]

    def decode_l0(self, plain_ids: list[int]) -> int:
        """L0: greedy decode loop driven by the GenAI Generator. append_tokens
        runs prefill, then generate_next_token steps the model L0-1 times for
        a total of L0_MAX_NEW_TOKENS decode steps (matches the other backends)."""
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
        return [float(row[v]) for v in self.verdict_token_ids]

    def decode_l0(self, plain_ids: list[int]) -> int:
        torch = self._torch
        t = torch.tensor([plain_ids], dtype=torch.long)
        self.model.generate(
            input_ids=t, attention_mask=torch.ones_like(t),
            max_new_tokens=L0_MAX_NEW_TOKENS, do_sample=False, num_beams=1,
        )
        return 0


class VLLMCPUBackend(GenBackend):
    """vLLM CPU backend — single-row, all-tricks-baked baseline.

    Skips the trick ladder. vLLM bakes its own paged-attention KV management
    and last-position sampling; the forced-prefix forward + sampling is the
    timed unit. The next-token argmax IS the verdict (the "Safety: " forcing
    constrains the model output to Safe / Unsafe / Controversial).

    Expects nothing under `--artifact`; vLLM resolves `--model-id` against the
    HF hub cache directly.
    """
    runtime = "vllm-cpu"

    def load(self, model_id: str, artifact: str | None, max_seq_len: int) -> None:
        # Import lazily — vLLM's import is slow (3-5s) and pulls torch/cuda
        # plumbing even on CPU.
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        self._TokensPrompt = TokensPrompt
        self.llm = LLM(
            model=model_id,
            dtype="float16",
            enforce_eager=True,
            max_model_len=max(2048, max_seq_len + 16),
            disable_log_stats=True,
        )
        self._sp = SamplingParams(temperature=0.0, max_tokens=1)
        import vllm
        self.detail = f"vllm {vllm.__version__} cpu"

    def predict(self, forced_ids: list[int]) -> int:
        """vLLM produces argmax tokens directly. With "Safety: " forced in the
        prompt, the next token is Safe / Unsafe / Controversial; we return the
        verdict index. Falls back to 0 if vLLM emits something unexpected."""
        out = self.llm.generate(
            [self._TokensPrompt(prompt_token_ids=forced_ids)],
            self._sp, use_tqdm=False,
        )
        next_id = out[0].outputs[0].token_ids[0]
        try:
            return self.verdict_token_ids.index(next_id)
        except ValueError:
            return 0

    def verdict_logits(self, forced_ids: list[int]) -> list[float]:
        # Not called — predict() is overridden directly. Stub satisfies the ABC.
        raise NotImplementedError("vLLM backend exposes predict() directly")


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
        # n_ctx must cover the longest forced_ids sequence; round up for headroom.
        n_ctx = max(2048, ((max_seq_len + 256) // 256 + 1) * 256)
        # n_threads_batch governs prefill (the whole L2 path); pin it to the
        # same physical-core count as n_threads so llama.cpp is measured on the
        # same thread budget as the other backends, not the default logical count.
        self.llm = Llama(
            model_path=str(artifact),
            n_ctx=n_ctx,
            n_threads=self.threads or None,
            n_threads_batch=self.threads or None,
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

    def decode_l0(self, plain_ids: list[int]) -> int:
        """L0: native greedy decode loop. Eval the prompt once, then for each
        of L0_MAX_NEW_TOKENS steps take the argmax of the last-position logits
        and eval it back in — the model-card autoregressive path."""
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
