"""Export Qwen3Guard-Gen to OpenVINO IR for the CPU L2 forced-prefix benchmark.

OpenVINO is the strongest CPU runtime on Intel x86 (AVX-512 / VNNI). The export
uses optimum-intel; int8 / int4 apply NNCF weight compression (weights quantized,
activations stay fp32 — the W8A16 / W4A16 pattern).

Layout (one IR per precision subdir, which `bench_gen_cpu.py --artifact` points at):

  ov_models/Qwen3Guard-Gen-0.6B/
    fp32/openvino_model.xml   full precision
    int8/openvino_model.xml   INT8 weight compression
    int4/openvino_model.xml   INT4 weight compression (group_size 128)

Idempotent: a precision whose openvino_model.xml already exists is skipped.

On Apple Silicon OpenVINO runs on the ARM CPU plugin; the latency there is not
representative of its x86 numbers. Treat Mac OpenVINO results as a sanity check
and read the Linux x86 run for the real comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_one(model_id: str, out: Path, precision: str) -> None:
    if (out / "openvino_model.xml").exists():
        print(f"[export-ov] {precision} already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

    if precision == "fp32":
        qcfg = None
    elif precision == "int8":
        qcfg = OVWeightQuantizationConfig(bits=8)
    elif precision == "int4":
        qcfg = OVWeightQuantizationConfig(bits=4, group_size=128, ratio=1.0)
    else:
        raise SystemExit(f"unknown precision {precision!r}")

    print(f"[export-ov] {precision}: {model_id} -> {out}")
    model = OVModelForCausalLM.from_pretrained(
        model_id, export=True, trust_remote_code=True, quantization_config=qcfg,
    )
    model.save_pretrained(str(out))
    print(f"[export-ov] {precision} done: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--out-dir", default=None,
                    help="Base dir. Default: ov_models/<model_basename>")
    ap.add_argument("--precisions", nargs="+", default=["fp32"],
                    choices=["fp32", "int8", "int4"])
    args = ap.parse_args()

    base = Path(args.out_dir) if args.out_dir else Path("ov_models") / Path(args.model_id).name
    for precision in args.precisions:
        export_one(args.model_id, base / precision, precision)

    print(f"[export-ov] artifacts under {base}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
