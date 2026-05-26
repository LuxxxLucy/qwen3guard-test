"""Convert a HuggingFace Qwen3Guard-Gen checkpoint to an MNN-LLM model dir.

Wraps MNN's `transformers/llm/export/llmexport.py`. The bench's fp16 row uses
`--quant_bit 16`. MNN's converter rejects 32-bit weights on the language path
(mnn_converter.py:349 asserts `quant_bit in (1,2,4,8)`), so fp16 is the
highest-precision MNN cell we can produce — accumulators stay fp32 at runtime
via precision="high". MNN bakes L2 lastpos via `forward()`'s default
last-position logit slice — no separate L2 row.

Pre-reqs:
  - $MNN_HOME points at a clone of alibaba/MNN with llmexport.py available at
    $MNN_HOME/transformers/llm/export/llmexport.py.
  - `pip install MNN onnx onnxslim onnxruntime yaspin` (the llmexport
    requirements set; pymnn provides the converter the script calls into).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PREC_TO_QUANT_BIT = {"fp16": 16}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True,
                    help="Local checkpoint directory (HF format).")
    ap.add_argument("--out-dir", default="var/models/mnn",
                    help="parent dir; converted model goes to <out-dir>/<basename>-<precision>")
    ap.add_argument("--precisions", nargs="+", default=["fp16"],
                    choices=list(PREC_TO_QUANT_BIT))
    ap.add_argument("--mnn-home", default=os.environ.get("MNN_HOME", ""),
                    help="path to a clone of alibaba/MNN (or set $MNN_HOME).")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing converted dirs")
    args = ap.parse_args()

    if not args.mnn_home:
        raise SystemExit("[mnn-export] --mnn-home or $MNN_HOME must point at a MNN clone.")
    llmexport = Path(args.mnn_home) / "transformers" / "llm" / "export" / "llmexport.py"
    if not llmexport.exists():
        raise SystemExit(f"[mnn-export] llmexport.py not found at {llmexport}")

    src = Path(args.model_id)
    if not src.exists():
        raise SystemExit(f"[mnn-export] --model-id must be a local directory; got {src}")
    basename = src.name

    out_parent = Path(args.out_dir)
    out_parent.mkdir(parents=True, exist_ok=True)

    for prec in args.precisions:
        dest = out_parent / f"{basename}-{prec}"
        if dest.exists():
            if not args.force:
                print(f"[mnn-export] {dest} exists; skipping (use --force to overwrite)")
                continue
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        quant_bit = PREC_TO_QUANT_BIT[prec]
        cmd = [
            sys.executable, str(llmexport),
            "--path", str(src),
            "--export", "mnn",
            "--dst_path", str(dest),
            "--quant_bit", str(quant_bit),
            "--lm_quant_bit", str(quant_bit),
        ]
        # Embedding default is bf16; keep it fp16-sized (bf16 token embedding +
        # the rest in fp32 weights) — MNN doesn't have an fp32 embedding mode
        # exposed via this flag, so bf16 stays as the byte-cheapest path.
        print(f"[mnn-export] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Patch the generated config.json so the runtime keeps fp32 accumulators
        # (precision="high" -> Precision_High on the CPU backend), grants enough
        # memory for the verdict-prefill forward, and the backend honest per-call
        # thread count. The exporter writes precision="low" by default (mobile).
        cfg_path = dest / "config.json"
        if cfg_path.exists():
            import json
            cfg = json.loads(cfg_path.read_text())
            cfg["precision"] = "high"
            cfg["memory"] = "high"
            cfg["thread_num"] = 16
            cfg_path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False))

        # MNN's published `pymnn` 3.5.0 runtime does not honor the
        # `llm_config.json` -> `tie_embeddings` structured form (it falls back to
        # reading a standalone bf16 embedding file). Qwen3-0.6B has tied
        # word embeddings, so the exporter writes no `embeddings_bf16.bin` and
        # the engine errors at load time with "Failed to open embedding file!".
        # Workaround: slice the tied embedding off `llm.mnn.weight` using the
        # offsets the exporter records in `llm_config.json`, convert the fp16
        # slab to bf16, and write the file the engine expects. Safe to remove
        # when MNN ships a runtime that reads the structured tie_embeddings.
        _materialize_tied_embedding(dest)
        print(f"[mnn-export] wrote {dest}")
    return 0


def _materialize_tied_embedding(dest: Path) -> None:
    import json
    import numpy as np

    llm_cfg_path = dest / "llm_config.json"
    if not llm_cfg_path.exists():
        return
    llm_cfg = json.loads(llm_cfg_path.read_text())
    tie = llm_cfg.get("tie_embeddings")
    if not isinstance(tie, dict):
        return
    weight_off = int(tie["weight_offset"])
    alpha_off = int(tie["alpha_offset"])
    quant_bit = int(tie.get("quant_bit", 16))
    if quant_bit != 16:
        return  # Only the fp16 -> bf16 path is implemented here.
    hidden = int(llm_cfg.get("hidden_size", 0))
    if hidden <= 0:
        return
    bytes_per_row = hidden * 2  # fp16
    span = alpha_off - weight_off
    if span <= 0 or span % bytes_per_row != 0:
        return
    weight_path = dest / "llm.mnn.weight"
    out_path = dest / "embeddings_bf16.bin"
    if out_path.exists():
        return
    with open(weight_path, "rb") as f:
        f.seek(weight_off)
        raw = f.read(span)
    fp16 = np.frombuffer(raw, dtype=np.float16)
    fp32_bits = fp16.astype(np.float32).view(np.uint32)
    # round-to-nearest-even bf16: keep top 16 bits, round with LSB tie-break.
    bf16 = ((fp32_bits + 0x7FFF + ((fp32_bits >> 16) & 1)) >> 16).astype(np.uint16)
    bf16.tofile(out_path)
    print(f"[mnn-export] materialized tied embedding -> {out_path.name} "
          f"({bf16.nbytes/2**20:.0f} MiB)")


if __name__ == "__main__":
    sys.exit(main())
