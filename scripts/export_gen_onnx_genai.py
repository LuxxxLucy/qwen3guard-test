"""Build a Qwen3Guard-Gen ONNX Runtime GenAI model with prune_lm_head=true.

Wraps `python -m onnxruntime_genai.models.builder`. The builder downloads
HF weights (if not cached) and writes:
  ortgenai_models/<model_basename>/fp32/{model.onnx, genai_config.json, ...}
"""
from __future__ import annotations

import argparse
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
        "-m", model_id, "-o", str(out), "-p", precision, "-e", "cpu",
        "--extra_options", "prune_lm_head=true",
    ]
    print(f"[export-genai] {precision}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--out-dir", default=None,
                    help="Base dir. Default: ortgenai_models/<model_basename>")
    ap.add_argument("--precisions", nargs="+", default=["fp32"],
                    choices=["fp32"])
    args = ap.parse_args()

    base = Path(args.out_dir) if args.out_dir else \
        Path("ortgenai_models") / Path(args.model_id).name
    for prec in args.precisions:
        build(args.model_id, base / prec, prec)

    print(f"[export-genai] artifacts under {base}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
