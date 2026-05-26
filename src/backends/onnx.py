"""ONNX Runtime CPU backend.

All exported graphs are with-past (text-generation-with-past). The L1 path
feeds zero-shape past_kv tensors; L0 (decode loop) accumulates them per step;
L3 primes the prefix's present-KV via IO-binding and reuses it across calls.
The artifact's lastpos slicing (fp32-lastpos vs fp32) is what differentiates
L2 from L1 — the backend itself is precision-agnostic.
"""
from __future__ import annotations

from pathlib import Path

from contract import L0_MAX_NEW_TOKENS

from .base import GenBackend


class OnnxCPUBackend(GenBackend):
    runtime = "onnx"
    SUPPORTS_KV_CACHE = True
    SUPPORTS_L0 = True
    SUPPORTS_LAST_POS_LOGITS = True  # acts as opt_level=L2 manifest; the slice
                                     # is in the graph, not a runtime path.

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
                             f"(run src/exporters/onnx.py first).")
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
                "var/models/onnx/<model>/{fp32,fp32-lastpos,int8,int8-lastpos} "
                "(re-run src/exporters/onnx.py).")
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
        empty = np.zeros((1, self._kv_heads, 0, self._head_dim), np.float32)
        feeds.update({n: empty for n in self._past_names})
        logits = self.session.run([self.output_name], feeds)[0]
        row = logits[0, -1]
        return self._read_verdicts(row)

    def decode_l0(self, plain_ids: list[int]) -> int:
        """Hand-rolled greedy decode over the with-past graph: initial forward
        with empty past_kv, then L0-1 per-token steps feeding the prior step's
        present_kv. Same per-call KV behaviour as generate(use_cache=True)."""
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
        generated = []
        for step in range(L0_MAX_NEW_TOKENS - 1):
            next_token = int(logits[0, -1].argmax())
            generated.append(next_token)
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
        return self._verdict_from_generated(generated)
