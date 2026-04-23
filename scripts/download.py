"""Pre-fetch Qwen3Guard weights + Qwen3GuardTest dataset into HF cache.

Idempotent: if already cached, this is a no-op. Run once before benchmarks to
isolate download time from measurement time.
"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download


GEN_IDS = {
    "0.6B": "Qwen/Qwen3Guard-Gen-0.6B",
    "4B":   "Qwen/Qwen3Guard-Gen-4B",
    "8B":   "Qwen/Qwen3Guard-Gen-8B",
}
STREAM_IDS = {
    "0.6B": "Qwen/Qwen3Guard-Stream-0.6B",
    "4B":   "Qwen/Qwen3Guard-Stream-4B",
    "8B":   "Qwen/Qwen3Guard-Stream-8B",
}


def fetch_model(repo_id: str) -> None:
    print(f"[download] {repo_id}")
    snapshot_download(repo_id=repo_id)


def fetch_dataset() -> None:
    # Qwen3GuardTest splits: thinking, response_loc, thinking_loc. Prefetch all
    # so the cache is warm regardless of which split a given bench mode uses
    # (bench_common.load_representative_texts reads response_loc).
    try:
        from datasets import load_dataset
        for split in ("response_loc", "thinking", "thinking_loc"):
            print(f"[download] Qwen/Qwen3GuardTest [{split}]")
            load_dataset("Qwen/Qwen3GuardTest", split=split)
    except Exception as e:
        print(f"[warn] dataset fetch failed: {e!r} (bench will fall back to synthetic).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", nargs="+", default=["0.6B"],
        help="Which size(s) to pre-fetch. Subset of 0.6B 4B 8B.",
    )
    ap.add_argument("--variants", nargs="+", default=["gen", "stream"],
                    choices=["gen", "stream"])
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()

    for size in args.sizes:
        if "gen" in args.variants:
            fetch_model(GEN_IDS[size])
        if "stream" in args.variants:
            fetch_model(STREAM_IDS[size])

    if not args.skip_dataset:
        fetch_dataset()

    print("[download] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
