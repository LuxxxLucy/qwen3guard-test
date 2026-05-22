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
import os
import sys
from pathlib import Path


def slice_lm_head_last_position(xml_path: Path) -> None:
    """Restrict the lm_head matmul to the last sequence position only.

    The IR projects every prompt position to the 151,936-vocab logits, but the
    L2 path reads only the last position — ~199/200 of that matmul is wasted.
    This inserts a Slice on the lm_head MatMul's hidden-state input keeping just
    the last position (axis=1), so the IR emits logits of shape [batch, 1, vocab].
    Generation reads only the last position per step, so this stays correct for
    the L0 generate() path too. The lm_head MatMul exists in the fp32 / int8 /
    int4 IRs regardless of weight quantization, so this applies to all three.

    The edited IR is saved to a sibling temp path, then atomically swapped in:
    save_model truncates and rewrites the .bin, and the model's weight constants
    are lazily mmap'd from that same .bin — overwriting it in place reads from a
    truncated mapping and crashes (SIGBUS). Writing a fresh file sidesteps that.
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

    hidden = matmul.input(0).get_source_output()  # [?, ?, hidden_dim]
    # Slice axis=1 (sequence), start=-1, stop=INT_MAX, step=1 -> keep last position.
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
    del model  # drop the mmap on the original .bin before overwriting it
    os.replace(tmp_xml, xml_path)
    os.replace(tmp_bin, xml_path.with_suffix(".bin"))
    print(f"[export-ov] lm_head sliced to last position: {xml_path}")


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
    slice_lm_head_last_position(out / "openvino_model.xml")
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
