"""Export Qwen3Guard-Gen to ONNX for the CPU L2 forced-prefix benchmark.

The default export uses task `text-generation` — one plain forward, no past
key/values — as fp32 plus optional int8 / int4 weight quantization.

`--with-past` additionally exports a `text-generation-with-past` graph
(past_key_values.* inputs, present.* outputs). The benchmark's --kv-cache mode
uses it to precompute the fixed system-prompt prefix KV once and forward only
the variable suffix per request.

Layout (one `model.onnx` per subdir, which `bench_gen_cpu.py --artifact`
points at):

  onnx_models/Qwen3Guard-Gen-0.6B/
    fp32/model.onnx       full precision, no past
    int8/model.onnx       dynamic INT8 weight quantization
    int4/model.onnx       4-bit blockwise weight quantization
    withpast/model.onnx   fp32 with past_key_values (for --kv-cache)

Idempotent: a subdir whose model.onnx already exists is skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# INT64_MAX — Slice `end` for "to the end of the axis".
_INT64_MAX = 2**63 - 1


def slice_lm_head_last_pos(model_path: Path) -> None:
    """Restrict the output projection (lm_head) to the last sequence position.

    The lm_head is the ONNX MatMul that produces the graph's `logits` output
    (directly, or through a Cast). Its non-initializer input is the hidden
    states `[batch, seq, hidden]`. We splice a `Slice` onto that input keeping
    only the last position (axis=1, start=-1, end=INT64_MAX, step=1), so the
    MatMul runs over `[batch, 1, hidden]` and `logits` becomes `[batch, 1, V]`.
    The L2 path reads only `logits[:, -1]`, so the prompt positions were wasted.

    Edits model.onnx in place. Idempotent: a second call detects the existing
    Slice and does nothing.
    """
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np

    model = onnx.load(str(model_path))
    graph = model.graph

    init_names = {i.name for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}

    # Locate the lm_head MatMul: walk back from the `logits` output through any
    # Cast to the MatMul that feeds it.
    node = producer.get("logits")
    if node is None:
        raise RuntimeError("graph has no `logits` output producer")
    while node.op_type == "Cast":
        node = producer[node.input[0]]
    if node.op_type != "MatMul":
        raise RuntimeError(f"expected lm_head MatMul, found {node.op_type}")
    matmul = node

    # The hidden-states input is the MatMul input that is not an initializer.
    hid_inputs = [i for i in matmul.input if i not in init_names]
    if len(hid_inputs) != 1:
        raise RuntimeError(
            f"lm_head MatMul {matmul.name} has {len(hid_inputs)} non-initializer "
            f"inputs, expected 1")
    hidden = hid_inputs[0]

    sliced_name = "lm_head_hidden_last"
    if any(sliced_name in n.output for n in graph.node):
        print(f"[export-onnx] slice-lm-head: {model_path} already sliced; skipping.")
        return

    # Slice index constants as initializers.
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

    # Rewire the MatMul to consume the sliced hidden states.
    for k, inp in enumerate(matmul.input):
        if inp == hidden:
            matmul.input[k] = sliced_name

    # Insert the Slice in topological order: right after the node producing
    # `hidden` (or at the front if `hidden` is a graph input).
    hidden_producer = producer.get(hidden)
    if hidden_producer is None:
        graph.node.insert(0, slice_node)
    else:
        idx = list(graph.node).index(hidden_producer)
        graph.node.insert(idx + 1, slice_node)

    # `logits` sequence dim is now 1.
    for o in graph.output:
        if o.name == "logits":
            dims = o.type.tensor_type.shape.dim
            if len(dims) >= 2:
                dims[1].Clear()
                dims[1].dim_value = 1

    use_ext = (model_path.with_suffix(".onnx_data")).exists()
    onnx.save(model, str(model_path), save_as_external_data=use_ext,
              all_tensors_to_one_file=True,
              location=model_path.name + "_data" if use_ext else None)
    print(f"[export-onnx] slice-lm-head: {model_path} -> lm_head MatMul now "
          f"projects only the last position (logits seq dim = 1).")


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
    # Restrict lm_head to the last position before int8/int4 quantize this
    # model, so the quantized exports inherit the slice.
    slice_lm_head_last_pos(out / "model.onnx")
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


def export_withpast(model_id: str, base: Path, opset: int) -> None:
    """Export a with-past graph — past_key_values.* inputs, present.* outputs —
    for the benchmark's --kv-cache mode. optimum builds the Qwen3 GQA past
    shapes (8 KV heads, head_dim 128) correctly."""
    out = base / "withpast"
    if (out / "model.onnx").exists():
        print(f"[export-onnx] with-past already at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)
    from optimum.exporters.onnx import main_export
    print(f"[export-onnx] with-past: {model_id} -> {out} "
          f"(task=text-generation-with-past, opset={opset})")
    main_export(
        model_name_or_path=model_id,
        output=str(out),
        task="text-generation-with-past",
        opset=opset,
        trust_remote_code=True,
        no_post_process=True,
    )
    slice_lm_head_last_pos(out / "model.onnx")
    print(f"[export-onnx] with-past done: {out}")


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
    ap.add_argument("--with-past", action="store_true",
                    help="Also export a with-past graph (past_key_values.* in, "
                         "present.* out) for the benchmark's --kv-cache mode.")
    args = ap.parse_args()

    base = Path(args.out_dir) if args.out_dir else Path("onnx_models") / Path(args.model_id).name
    fp32_dir = base / "fp32"

    # fp32 is the source for any quantized precision, so build it first.
    export_fp32(args.model_id, fp32_dir, args.opset)
    if "int8" in args.precisions:
        quantize_int8(fp32_dir, base / "int8")
    if "int4" in args.precisions:
        quantize_int4(fp32_dir, base / "int4")
    if args.with_past:
        export_withpast(args.model_id, base, args.opset)

    print(f"[export-onnx] artifacts under {base}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
