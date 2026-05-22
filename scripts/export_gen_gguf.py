"""Export Qwen3Guard-Gen to GGUF for the llama.cpp CPU benchmark.

Two ways to get a GGUF, preferred order:

  1. Download a pre-quantized community GGUF (--gguf-repo / --gguf-file), via
     the same huggingface_hub path scripts/download.py uses. No build needed.

  2. Convert from the HF checkpoint. Shallow-clones llama.cpp into vendor/,
     runs its convert_hf_to_gguf.py (f16 and q8_0 need no C++ build). Q4_K_M
     additionally needs the llama-quantize binary, which this script builds
     with cmake when cmake is available, and skips with a note otherwise.

Output: gguf_models/<model_basename>.<quant>.gguf — the path `bench_gen_cpu.py
--runtime llamacpp --artifact` expects.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
VENDOR = Path("vendor/llama.cpp")
OUT_DIR = Path("gguf_models")


def run(cmd: list[str], **kw) -> None:
    print(f"[export-gguf] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)


def download_community(repo: str, fname: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    print(f"[export-gguf] downloading {repo}/{fname}")
    src = hf_hub_download(repo_id=repo, filename=fname)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"[export-gguf] -> {dest}")


def ensure_llama_cpp() -> Path:
    if not (VENDOR / "convert_hf_to_gguf.py").exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(VENDOR)])
    return VENDOR


def convert(model_dir: str, repo_dir: Path, out: Path, outtype: str) -> None:
    """Run llama.cpp's convert_hf_to_gguf.py at the requested outtype (f16 or
    q8_0 — neither needs a C++ build). PYTHONPATH points at the cloned gguf-py
    so the converter and its format library are version-matched."""
    if out.exists():
        print(f"[export-gguf] {out} exists; skipping.")
        return
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_dir / "gguf-py")
    run([sys.executable, str(repo_dir / "convert_hf_to_gguf.py"),
         model_dir, "--outfile", str(out), "--outtype", outtype], env=env)


def build_llama_quantize(repo_dir: Path) -> Path | None:
    """Build the llama-quantize binary. Returns its path, or None if cmake is
    unavailable (q4_k_m is then skipped)."""
    binary = repo_dir / "build" / "bin" / "llama-quantize"
    if binary.exists():
        return binary
    if shutil.which("cmake") is None:
        print("[export-gguf] cmake not found — skipping q4_k_m. Install cmake, "
              "or download a community Q4_K_M GGUF with --gguf-repo/--gguf-file.")
        return None
    run(["cmake", "-B", str(repo_dir / "build"), "-S", str(repo_dir),
         "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(repo_dir / "build"),
         "--target", "llama-quantize", "--config", "Release", "-j"])
    return binary if binary.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--quants", nargs="+", default=["q8_0"],
                    choices=["f16", "q8_0", "q4_k_m"])
    ap.add_argument("--gguf-repo", default=None,
                    help="Community GGUF repo id — download instead of convert.")
    ap.add_argument("--gguf-file", default=None,
                    help="Filename within --gguf-repo (one per --quants entry, "
                         "matched by order).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basename = Path(args.model_id).name

    if args.gguf_repo:
        for quant, fname in zip(args.quants, [args.gguf_file]):
            download_community(args.gguf_repo, fname,
                               OUT_DIR / f"{basename}.{quant}.gguf")
        return 0

    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(args.model_id)

    repo_dir = ensure_llama_cpp()
    f16_path = OUT_DIR / f"{basename}.f16.gguf"
    # q4_k_m is quantized from a f16 GGUF, so produce that base first if needed.
    if "f16" in args.quants or "q4_k_m" in args.quants:
        convert(model_dir, repo_dir, f16_path, "f16")

    for quant in args.quants:
        if quant == "f16":
            continue  # produced above
        out = OUT_DIR / f"{basename}.{quant}.gguf"
        if quant == "q8_0":
            convert(model_dir, repo_dir, out, "q8_0")
        elif quant == "q4_k_m":  # quantized from the f16 base by llama-quantize
            if out.exists():
                print(f"[export-gguf] {out} exists; skipping.")
                continue
            binary = build_llama_quantize(repo_dir)
            if binary is not None:
                run([str(binary), str(f16_path), str(out), "Q4_K_M"])

    print(f"[export-gguf] artifacts under {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
