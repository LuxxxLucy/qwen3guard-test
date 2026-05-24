"""Export Qwen3Guard-Gen to GGUF for the llama.cpp CPU benchmark.

Shallow-clones llama.cpp into vendor/, runs its convert_hf_to_gguf.py
(f32/f16/q8_0 need no C++ build), then for K-quant formats (q4_K_M, ...)
two-step quantizes from the f16 GGUF via the bundled `llama-quantize`
binary. Output:
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

# Quants that convert_hf_to_gguf.py emits directly vs. those that require a
# two-step pipeline (convert to f16, then llama-quantize to the K-quant).
DIRECT_QUANTS = {"f32", "f16", "q8_0"}
KQUANTS = {"q4_K_M", "q4_K_S", "q5_K_M", "q5_K_S", "q6_K"}


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


def find_llama_quantize() -> Path:
    """K-quant export needs the C++ `llama-quantize` binary. It ships inside
    the installed llama-cpp-python wheel under llama_cpp/lib/."""
    import llama_cpp
    candidate = Path(llama_cpp.__file__).parent / "lib" / "llama-quantize"
    if candidate.exists():
        return candidate
    raise SystemExit(
        f"[export-gguf] llama-quantize not found at {candidate}. "
        "K-quant export needs the bundled binary; rebuild llama-cpp-python "
        "without GGML_DISABLE_LLAMA_QUANTIZE."
    )


def quantize(src: Path, out: Path, quant: str) -> None:
    if out.exists():
        print(f"[export-gguf] {out} exists; skipping.")
        return
    binary = find_llama_quantize()
    run([str(binary), str(src), str(out), quant])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--quants", nargs="+", default=["q8_0"],
                    choices=sorted(DIRECT_QUANTS | KQUANTS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basename = Path(args.model_id).name

    if Path(args.model_id).is_dir():
        model_dir = args.model_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model_id)

    repo_dir = ensure_llama_cpp()

    direct = [q for q in args.quants if q in DIRECT_QUANTS]
    kquants = [q for q in args.quants if q in KQUANTS]

    for quant in direct:
        convert(model_dir, repo_dir, OUT_DIR / f"{basename}.{quant}.gguf", quant)

    if kquants:
        # K-quant quantize reads an existing f16 GGUF; emit it if missing.
        f16 = OUT_DIR / f"{basename}.f16.gguf"
        convert(model_dir, repo_dir, f16, "f16")
        for quant in kquants:
            quantize(f16, OUT_DIR / f"{basename}.{quant}.gguf", quant)

    print(f"[export-gguf] artifacts under {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
