"""Export Qwen3Guard-Gen to ONNX for the CPU L2 forced-prefix benchmark.

The L2 path is ONE forward pass — no decode loop, no KV cache — so the model is
exported with task `text-generation` (plain forward, no past key/values). This
is deliberately NOT `text-generation-with-past`: the with-past export trips
optimum's dummy KV generator on Qwen3 GQA (head_dim 128 != hidden/heads 64),
and the L2 path never needs a KV cache anyway.

Layout (one `model.onnx` per precision subdir, which `bench_gen_cpu.py --artifact`
points at):

  onnx_models/Qwen3Guard-Gen-0.6B/
    fp32/model.onnx     full precision
    int8/model.onnx     dynamic INT8 weight quantization
    int4/model.onnx     4-bit blockwise weight quantization (W4, activations fp32)

Idempotent: a precision whose model.onnx already exists is skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_fp32(model_id: str, out: Path, opset: int) -> None:
    if (out / "model.onnx").exists():
        print(f"[export-onnx] fp32 already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    from optimum.exporters.onnx import main_export
    print(f"[export-onnx] fp32: {model_id} -> {out} (task=text-generation, opset={opset})")
    # no_post_process=True skips tied-weight dedup, which reserializes the whole
    # proto and can hit the 2GB protobuf limit even for small models.
    main_export(
        model_name_or_path=model_id,
        output=str(out),
        task="text-generation",
        opset=opset,
        trust_remote_code=True,
        no_post_process=True,
    )
    print(f"[export-onnx] fp32 done: {out}")


def quantize_int8(src: Path, out: Path) -> None:
    if (out / "model.onnx").exists():
        print(f"[export-onnx] int8 already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print(f"[export-onnx] int8: dynamic weight quantization {src} -> {out}")
    quantize_dynamic(
        model_input=str(src / "model.onnx"),
        model_output=str(out / "model.onnx"),
        weight_type=QuantType.QInt8,
    )
    print(f"[export-onnx] int8 done: {out}")


def quantize_int4(src: Path, out: Path) -> None:
    if (out / "model.onnx").exists():
        print(f"[export-onnx] int4 already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    print(f"[export-onnx] int4: 4-bit blockwise weight quantization {src} -> {out}")
    # Default QOperator format emits the com.microsoft MatMulNBits op, which the
    # CPU execution provider runs directly — no opset bump needed. block_size 32
    # trades a little file size for better accuracy than the 128 default.
    quant = MatMulNBitsQuantizer(
        str(src / "model.onnx"), bits=4, block_size=32, is_symmetric=True,
    )
    quant.process()
    quant.model.save_model_to_file(str(out / "model.onnx"), use_external_data_format=True)
    print(f"[export-onnx] int4 done: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--out-dir", default=None,
                    help="Base dir. Default: onnx_models/<model_basename>")
    ap.add_argument("--precisions", nargs="+", default=["fp32"],
                    choices=["fp32", "int8", "int4"],
                    help="Which precisions to produce. int8/int4 quantize the "
                         "fp32 export, which is always built first.")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    base = Path(args.out_dir) if args.out_dir else Path("onnx_models") / Path(args.model_id).name
    fp32_dir = base / "fp32"

    # fp32 is the source for any quantized precision, so build it first.
    export_fp32(args.model_id, fp32_dir, args.opset)
    if "int8" in args.precisions:
        quantize_int8(fp32_dir, base / "int8")
    if "int4" in args.precisions:
        quantize_int4(fp32_dir, base / "int4")

    print(f"[export-onnx] artifacts under {base}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
