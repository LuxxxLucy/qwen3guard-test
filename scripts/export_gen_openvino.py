"""Export Qwen3Guard-Gen to OpenVINO IR for the CPU L2 forced-prefix benchmark.

The fp16 export is unquantized (optimum-intel's compress_to_fp16 default).
int8 / int4 apply NNCF weight compression (W8A16 / W4A16).

  ov_models/Qwen3Guard-Gen-0.6B/
    fp16/openvino_model.xml
    int8/openvino_model.xml
    int4/openvino_model.xml

Idempotent: a precision whose openvino_model.xml already exists is skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def slice_lm_head_last_position(xml_path: Path) -> None:
    """Insert a Slice on the lm_head MatMul's hidden-state input keeping just
    the last position — same trick as the ONNX path, in OV's IR.

    Saves the edited IR to a sibling temp file then atomically swaps: OV's
    save_model truncates the .bin while weight constants are still lazily
    mmap'd from it — overwriting in place reads from a truncated mapping and
    crashes (SIGBUS). Writing a fresh file sidesteps that.
    """
    import openvino as ov
    from openvino import opset13 as opset

    core = ov.Core()
    model = core.read_model(str(xml_path))

    matmul = None
    for op in model.get_ops():
        name = op.get_friendly_name()
        if "lm_head" in name and "MatMul" in name:
            matmul = op
            break
    if matmul is None:
        raise SystemExit(f"[export-ov] lm_head MatMul not found in {xml_path}")

    hidden = matmul.input(0).get_source_output()
    start = opset.constant([-1], dtype=ov.Type.i32)
    stop = opset.constant([2**31 - 1], dtype=ov.Type.i32)
    step = opset.constant([1], dtype=ov.Type.i32)
    axis = opset.constant([1], dtype=ov.Type.i32)
    sliced = opset.slice(hidden, start, stop, step, axis)
    matmul.input(0).replace_source_output(sliced.output(0))

    model.validate_nodes_and_infer_types()

    xml_path = xml_path.resolve()
    tmp_xml = xml_path.with_suffix(".sliced.xml")
    tmp_bin = tmp_xml.with_suffix(".bin")
    ov.save_model(model, str(tmp_xml))
    del model
    os.replace(tmp_xml, xml_path)
    os.replace(tmp_bin, xml_path.with_suffix(".bin"))
    print(f"[export-ov] lm_head sliced to last position: {xml_path}")


def export_one(model_id: str, out: Path, precision: str) -> None:
    if (out / "openvino_model.xml").exists():
        print(f"[export-ov] {precision} already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

    if precision == "fp16":
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
    slice_lm_head_last_position(out / "openvino_model.xml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--out-dir", default=None,
                    help="Base dir. Default: ov_models/<model_basename>")
    ap.add_argument("--precisions", nargs="+", default=["fp16"],
                    choices=["fp16", "int8", "int4"])
    args = ap.parse_args()

    base = Path(args.out_dir) if args.out_dir else Path("ov_models") / Path(args.model_id).name
    for precision in args.precisions:
        export_one(args.model_id, base / precision, precision)

    print(f"[export-ov] artifacts under {base}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
