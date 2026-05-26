"""Convert a HuggingFace Qwen3Guard-Gen checkpoint to a CTranslate2 model dir.

Wraps `ct2-transformers-converter` for the bench's single fp32 row. CTranslate2
bakes L2 lastpos via `forward_batch`'s last-position slice — no separate L2 row.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True,
                    help="HF repo id or local checkpoint directory.")
    ap.add_argument("--out-dir", default="var/models/ctranslate2",
                    help="parent dir; converted model goes to <out-dir>/<basename>-<precision>")
    ap.add_argument("--precisions", nargs="+", default=["fp32"],
                    choices=["fp32", "fp16", "int8"])
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing converted dirs")
    args = ap.parse_args()

    basename = Path(args.model_id).name
    out_parent = Path(args.out_dir)
    out_parent.mkdir(parents=True, exist_ok=True)

    prec_map = {"fp32": "float32", "fp16": "float16", "int8": "int8"}
    for prec in args.precisions:
        dest = out_parent / f"{basename}-{prec}"
        if dest.exists():
            if not args.force:
                print(f"[ct2-export] {dest} exists; skipping (use --force to overwrite)")
                continue
            shutil.rmtree(dest)
        # No --copy_files: the bench loads the HF tokenizer by model_id, so CT2
        # only needs model.bin + config.json + vocabulary.json (auto-emitted).
        cmd = [
            "ct2-transformers-converter",
            "--model", args.model_id,
            "--output_dir", str(dest),
            "--quantization", prec_map[prec],
        ]
        print(f"[ct2-export] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[ct2-export] wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
