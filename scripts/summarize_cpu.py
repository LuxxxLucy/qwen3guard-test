"""Print the Qwen3Guard-Gen CPU benchmark table from result JSONs.

Scans results/ for the files written by the Gen benchmark cells and prints a
method x template latency pivot to stdout. Also writes CPU_GEN_REPORT.md so the
whole benchmark is readable from a single stdout stream and a single file.
Run at the end of scripts/run_gen_cpu.sh.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from contract import DEFAULT_PRECISION, OPT_LEVELS, RUNTIMES, TEMPLATES, ResultExtra

RT_PYTORCH, RT_ONNX, RT_OPENVINO, RT_LLAMACPP = RUNTIMES
OPT_LP, OPT_L0, OPT_L1, OPT_L2, OPT_L2_LASTPOS, OPT_L3 = OPT_LEVELS

# (runtime, precision, opt_level, kv_cache) -> row label. Order = table order:
# per backend, L0 (unoptimized) then L2 variants progressively layering tricks.
# See LEGEND below for what each label means.
ROWS: list[tuple[tuple[str, str, str, bool], str]] = [
    ((RT_PYTORCH,    "fp32", OPT_L0,         False), "pytorch L0"),
    ((RT_PYTORCH,    "fp32", OPT_L2,         False), "pytorch L2"),
    ((RT_PYTORCH,    "fp32", OPT_L2_LASTPOS, False), "pytorch L2 +lastpos"),
    ((RT_ONNX,       "fp32", OPT_L0,         False), "onnx fp32 L0"),
    ((RT_ONNX,       "fp32", OPT_L2,         False), "onnx fp32 L2"),
    ((RT_ONNX,       "fp32", OPT_L2,         True),  "onnx fp32 L2 +kv"),
    ((RT_ONNX,       "int8", OPT_L2,         False), "onnx int8 L2"),
    ((RT_OPENVINO,   "fp16", OPT_L0,         False), "openvino fp16 L0"),
    ((RT_OPENVINO,   "fp16", OPT_L2,         False), "openvino fp16 L2"),
    ((RT_OPENVINO,   "int8", OPT_L2,         False), "openvino int8 L2"),
    ((RT_LLAMACPP,   "q8_0", OPT_L0,         False), "llamacpp q8_0 L0"),
    ((RT_LLAMACPP,   "q8_0", OPT_L2,         False), "llamacpp q8_0 L2"),
    ((RT_LLAMACPP,   "q8_0", OPT_L2,         True),  "llamacpp q8_0 L2 +kv"),
    ((RT_LLAMACPP,   "f16",  OPT_L2,         False), "llamacpp f16 L2"),
    ((RT_LLAMACPP,   "f16",  OPT_L2,         True),  "llamacpp f16 L2 +kv"),
    (("rust-candle", "fp32", OPT_L0,         False), "rust-candle L0"),
    (("rust-candle", "fp32", OPT_L2,         False), "rust-candle L2"),
    (("rust-candle", "fp32", OPT_L2,         True),  "rust-candle L2 +kv"),
]

_EXTRA_FIELDS = set(ResultExtra.__dataclass_fields__)
assert set(DEFAULT_PRECISION) == set(RUNTIMES)
assert {"mode", "precision", "opt_level", "kv_cache", "template"} <= _EXTRA_FIELDS
assert all(r in set(RUNTIMES) | {"rust-candle"} and o in OPT_LEVELS
           for (r, _p, o, _kv), _label in ROWS)


def load_results(results_dir: Path) -> list[dict]:
    """Read one BenchResult dict per JSON file. Tolerate stale files (list-
    typed JSONs from deleted bench variants, or anything else dropped here)
    by skipping with a warning — the table builder only knows the dict shape."""
    out = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except Exception as e:
            print(f"[warn] could not read {p.name}: {e!r}")
            continue
        if not isinstance(obj, dict):
            print(f"[warn] skipping {p.name}: top-level is {type(obj).__name__}, "
                  f"not a BenchResult dict")
            continue
        out.append(obj)
    return out


def latest_per_key(rows: list[dict], key) -> dict:
    """Keep the most recent result per key (later timestamp wins)."""
    best: dict = {}
    for r in rows:
        k = key(r)
        if k not in best or r.get("timestamp_utc", "") >= best[k].get("timestamp_utc", ""):
            best[k] = r
    return best


def result_key(r: dict) -> tuple[str, str, str, bool, str]:
    ex = r.get("extra", {})
    runtime = r.get("runtime", "")
    precision = ex.get("precision") or r.get("dtype", "")
    opt_level = ex.get("opt_level", "")
    kv_cache = bool(ex.get("kv_cache", False))
    template = ex.get("template", "")
    return (runtime, precision, opt_level, kv_cache, template)


def cell(r: dict | None) -> str:
    if r is None:
        return "-"
    lat = r["latency"]
    return f"{lat['p50_ms']:.1f} / {lat['p99_ms']:.1f}"


def build_table(rows: list[dict]) -> list[list[str]]:
    gen = [r for r in rows
           if r.get("variant") == "gen" and r.get("device") == "cpu"
           and r.get("extra", {}).get("mode") == "representative"]
    best = latest_per_key(gen, result_key)
    out = [["method", *TEMPLATES]]
    for key, label in ROWS:
        cells = [cell(best.get((*key, t))) for t in TEMPLATES]
        out.append([label, *cells])
    return out


def render_markdown(table: list[list[str]]) -> str:
    header, *body = table
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |"
    return "\n".join([fmt(header), sep, *(fmt(r) for r in body)])


def print_table(table: list[list[str]]) -> None:
    header, *body = table
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in [header, *body]:
        print("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


PARAGRAPH = (
    "Each cell is `p50 / p99` latency in milliseconds. "
    "Every method runs 5 warmup calls then 100 timed iterations. "
    "Latency is per-call wall-clock time at batch size 1; "
    "threads are pinned to the host's physical core count."
)


LEGEND = """## Legend

The columns are the two input templates: **original** (the model card's built-in Qwen3Guard system prompt, ~296-token overhead) and **test-200** (a compressed policy with ~130-token overhead).

The rows are an optimization ladder per backend. Within each backend the rows progress unoptimized → optimized; the labels name which tricks are on:

- **L0** — unoptimized. `tokenize → generate() decode loop (~32 tokens) → parse 'Safety: <verdict>'`. The model-card path. Per-call KV cache during the decode is on (it's what `generate()` does); none of the cross-call tricks below are.

- **L2** — `+forced-prefix`. Teacher-force `"Safety: "` and read the 3 verdict logits from one forward pass. No decode loop.

- **L2 +lastpos** — `+last-position lm_head`. L2 plus restrict the output projection to the last position only — skip the ~200 prompt-position logits that nobody reads. PyTorch exposes this via `logits_to_keep=1`. ONNX and OpenVINO bake it into the export; llama.cpp and Rust candle do it by default. So only PyTorch shows a separate `+lastpos` row.

- **L2 +kv** — `+system-prompt KV cache`. L2 plus precompute the shared system-prompt prefix KV once and reuse it across calls. Per-call cost shrinks to the variable-suffix forward. ONNX uses the with-past graph + IO binding; llama.cpp rewinds its context in place; Rust candle clones the primed model. PyTorch and OpenVINO don't have this mode in the bench."""


def write_report(table: list[list[str]], path: Path) -> None:
    md = render_markdown(table)
    path.write_text(
        "# Qwen3Guard-Gen CPU benchmark\n\n"
        + md
        + "\n\n"
        + PARAGRAPH
        + "\n\n"
        + LEGEND
        + "\n"
    )


def main() -> int:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if not results_dir.exists():
        print(f"[summarize] no results dir at {results_dir}")
        return 0
    rows = load_results(results_dir)
    table = build_table(rows)

    print()
    print("Qwen3Guard-Gen CPU — method x template, p50 / p99 ms")
    print_table(table)
    print()

    report = Path("CPU_GEN_REPORT.md")
    write_report(table, report)
    print(f"[summarize] wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
