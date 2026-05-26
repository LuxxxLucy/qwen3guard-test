"""Export Qwen3Guard-Gen to ONNX for the CPU bench.

Layout (one `model.onnx` per subdir, which `bench_gen_cpu.py --artifact`
points at):

  var/models/onnx/Qwen3Guard-Gen-0.6B/
    fp32/model.onnx          unmodified pytorch-shaped graph: with past_key_values,
                              lm_head MatMul over the full sequence. Used by L0
                              (decode loop) and +L1 (forced prefix, empty past).
    fp32-lastpos/model.onnx  same with-past graph, lm_head sliced to last-pos
                              (the L2 lastpos trick baked into export). Used by
                              +L2 (forced prefix) and +L3 (forced prefix + prefix-KV).
    int8/model.onnx          dynamic-INT8 quantization of fp32/.
    int8-lastpos/model.onnx  dynamic-INT8 quantization of fp32-lastpos/.

Idempotent: a subdir whose model.onnx already exists is skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Slice `end` for "to the end of the axis".
_INT64_MAX = 2**63 - 1


def slice_lm_head_last_pos(model_path: Path) -> None:
    """Splice a Slice (axis=1, start=-1, end=INT64_MAX, step=1) onto the
    lm_head MatMul's hidden-states input so the MatMul runs over `[B, 1, H]`
    and `logits` becomes `[B, 1, V]`. Edits model.onnx in place. Idempotent.

    Loads with load_external_data=False so the multi-GB weight blob never
    touches RAM (matters on resource-constrained aarch64 builders); saves the
    graph alone, leaving the external data file in place untouched."""
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np

    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph

    init_names = {i.name for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}

    # Walk back from `logits` through any Cast to the producing MatMul.
    node = producer["logits"]
    while node.op_type == "Cast":
        node = producer[node.input[0]]
    if node.op_type != "MatMul":
        raise RuntimeError(f"expected lm_head MatMul, found {node.op_type}")
    matmul = node

    hidden = next(i for i in matmul.input if i not in init_names)
    sliced_name = "lm_head_hidden_last"
    if any(sliced_name in n.output for n in graph.node):
        print(f"[export-onnx] slice-lm-head: {model_path} already sliced; skipping.")
        return

    consts = {
        "lm_head_slice_starts": np.array([-1], dtype=np.int64),
        "lm_head_slice_ends": np.array([_INT64_MAX], dtype=np.int64),
        "lm_head_slice_axes": np.array([1], dtype=np.int64),
        "lm_head_slice_steps": np.array([1], dtype=np.int64),
    }
    for cname, arr in consts.items():
        graph.initializer.append(numpy_helper.from_array(arr, name=cname))

    slice_node = helper.make_node(
        "Slice",
        inputs=[hidden, "lm_head_slice_starts", "lm_head_slice_ends",
                "lm_head_slice_axes", "lm_head_slice_steps"],
        outputs=[sliced_name],
        name="lm_head_last_pos_slice",
    )

    for k, inp in enumerate(matmul.input):
        if inp == hidden:
            matmul.input[k] = sliced_name

    hidden_producer = producer.get(hidden)
    if hidden_producer is None:
        graph.node.insert(0, slice_node)
    else:
        idx = list(graph.node).index(hidden_producer)
        graph.node.insert(idx + 1, slice_node)

    onnx.save(model, str(model_path))
    print(f"[export-onnx] slice-lm-head: {model_path}")


def export_fp32(model_id: str, out: Path, opset: int,
                slice_lm_head: bool = False) -> None:
    """Export the with-past fp32 graph. slice_lm_head=True splices a Slice node
    onto the lm_head MatMul input (the L2 lastpos bake)."""
    if (out / "model.onnx").exists():
        print(f"[export-onnx] fp32 already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    from optimum.exporters.onnx import main_export
    print(f"[export-onnx] fp32: {model_id} -> {out}")
    main_export(
        model_name_or_path=model_id, output=str(out),
        task="text-generation-with-past", opset=opset, trust_remote_code=True,
    )
    if slice_lm_head:
        slice_lm_head_last_pos(out / "model.onnx")


def quantize_int8(src: Path, out: Path) -> None:
    if (out / "model.onnx").exists():
        print(f"[export-onnx] int8 already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print(f"[export-onnx] int8: {src} -> {out}")
    quantize_dynamic(
        model_input=str(src / "model.onnx"),
        model_output=str(out / "model.onnx"),
        weight_type=QuantType.QInt8,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--out-dir", default=None,
                    help="Base dir. Default: var/models/onnx/<model_basename>")
    ap.add_argument("--precisions", nargs="+", default=["fp32"],
                    choices=["fp32", "int8"])
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    base = Path(args.out_dir) if args.out_dir else Path("var/models/onnx") / Path(args.model_id).name

    export_fp32(args.model_id, base / "fp32",         args.opset, slice_lm_head=False)
    export_fp32(args.model_id, base / "fp32-lastpos", args.opset, slice_lm_head=True)
    if "int8" in args.precisions:
        quantize_int8(base / "fp32",         base / "int8")
        quantize_int8(base / "fp32-lastpos", base / "int8-lastpos")

    print(f"[export-onnx] artifacts under {base}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
