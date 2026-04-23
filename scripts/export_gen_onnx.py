"""Export Qwen3Guard-Gen to ONNX using optimum.

Uses `optimum.exporters.onnx.main_export` (Python API) — equivalent to
`optimum-cli export onnx --model <id> --task text-generation <outdir>` but
more portable (doesn't require optimum-cli on PATH).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--out-dir", default=None,
                    help="Default: onnx_models/<model_basename>")
    ap.add_argument("--task", default="text-generation-with-past")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else Path("onnx_models") / Path(args.model_id).name
    if out.exists() and any(out.glob("*.onnx")):
        print(f"[export] {out} already has ONNX files; skipping.")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    from optimum.exporters.onnx import main_export

    print(f"[export] {args.model_id} -> {out} (task={args.task}, opset={args.opset})")
    # no_post_process=True skips the tied-weight-deduplication step which
    # tries to reserialize the whole ONNX proto and can hit the 2GB protobuf
    # limit even for small models when weights are externally stored.
    main_export(
        model_name_or_path=args.model_id,
        output=str(out),
        task=args.task,
        opset=args.opset,
        trust_remote_code=True,
        no_post_process=True,
    )
    print(f"[export] done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
