"""Build a Qwen3Guard-Gen ONNX Runtime GenAI model.

Wraps `python -m onnxruntime_genai.models.builder`, which is the LLM-aware
export path supported by ONNX Runtime GenAI. We build with
`prune_lm_head=true` so the LM head only computes the last token's logits
during prefill — that bakes the L2 (lastpos) trick into the L0 baseline, and
the resulting graph stops at `[B, 1, V]` instead of `[B, S, V]`.

The builder downloads the HuggingFace weights (if not cached) and writes:
  ortgenai_models/<model_basename>/<precision>/{model.onnx, genai_config.json,
                                                 tokenizer.json, ...}

Usage:
  uv run python scripts/export_gen_onnx_genai.py --precisions fp32
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def build(model_id: str, out: Path, precision: str) -> None:
    if (out / "genai_config.json").exists():
        print(f"[export-genai] {precision} already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-m", model_id,
        "-o", str(out),
        "-p", precision,
        "-e", "cpu",
        "--extra_options", "prune_lm_head=true",
    ]
    print(f"[export-genai] {precision}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[export-genai] {precision} done: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--out-dir", default=None,
                    help="Base dir. Default: ortgenai_models/<model_basename>")
    ap.add_argument("--precisions", nargs="+", default=["fp32"],
                    choices=["fp32", "fp16", "int4"],
                    help="Precisions the builder accepts. fp32 is the safest "
                         "on CPU; fp16 may upcast at runtime [unverified]; "
                         "int4 quantizes the LM head.")
    args = ap.parse_args()

    base = Path(args.out_dir) if args.out_dir else \
        Path("ortgenai_models") / Path(args.model_id).name

    for prec in args.precisions:
        build(args.model_id, base / prec, prec)

    print(f"[export-genai] artifacts under {base}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
