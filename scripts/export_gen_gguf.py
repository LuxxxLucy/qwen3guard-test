"""Export Qwen3Guard-Gen to GGUF for the llama.cpp CPU benchmark.

Shallow-clones llama.cpp into vendor/, runs its convert_hf_to_gguf.py
(f16 and q8_0 need no C++ build). Output:
  gguf_models/<model_basename>.<quant>.gguf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
VENDOR = Path("vendor/llama.cpp")
OUT_DIR = Path("gguf_models")


def run(cmd: list[str], **kw) -> None:
    print(f"[export-gguf] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)


def ensure_llama_cpp() -> Path:
    if not (VENDOR / "convert_hf_to_gguf.py").exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(VENDOR)])
    return VENDOR


def convert(model_dir: str, repo_dir: Path, out: Path, outtype: str) -> None:
    if out.exists():
        print(f"[export-gguf] {out} exists; skipping.")
        return
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_dir / "gguf-py")
    run([sys.executable, str(repo_dir / "convert_hf_to_gguf.py"),
         model_dir, "--outfile", str(out), "--outtype", outtype], env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--quants", nargs="+", default=["q8_0"],
                    choices=["f32", "f16", "q8_0"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basename = Path(args.model_id).name

    if Path(args.model_id).is_dir():
        model_dir = args.model_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model_id)

    repo_dir = ensure_llama_cpp()
    for quant in args.quants:
        convert(model_dir, repo_dir, OUT_DIR / f"{basename}.{quant}.gguf", quant)

    print(f"[export-gguf] artifacts under {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
