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

# (runtime, precision, opt_level, kv_cache) -> row label. Order = table order.
ROWS: list[tuple[tuple[str, str, str, bool], str]] = [
    (("pytorch",     "fp32",   "L0", False), "pytorch (L0)"),
    (("pytorch",     "fp32",   "L2", False), "pytorch (L2)"),
    (("pytorch",     "fp32",   "L2-lastpos", False), "pytorch (L2 last-pos)"),
    (("onnx",        "fp32",   "L2", False), "onnx fp32"),
    (("onnx",        "int8",   "L2", False), "onnx int8"),
    (("onnx",        "fp32",   "L2", True),  "onnx fp32 +kv"),
    (("openvino",    "fp16",   "L0", False), "openvino (L0)"),
    (("openvino",    "fp16",   "L2", False), "openvino fp16"),
    (("openvino",    "int8",   "L2", False), "openvino int8"),
    (("llamacpp",    "q8_0",   "L0", False), "llamacpp (L0)"),
    (("llamacpp",    "q8_0",   "L2", False), "llamacpp q8_0"),
    (("llamacpp",    "q8_0",   "L2", True),  "llamacpp q8_0 +kv"),
    (("llamacpp",    "f16",    "L2", False), "llamacpp f16"),
    (("llamacpp",    "f16",    "L2", True),  "llamacpp f16 +kv"),
    (("rust-candle", "fp32",   "L0", False), "rust-candle (L0)"),
    (("rust-candle", "fp32",   "L2", True),  "rust-candle (L2)"),
]
TEMPLATES = ["original", "test-200"]


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[warn] could not read {p.name}: {e!r}")
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


def write_report(table: list[list[str]], path: Path) -> None:
    md = render_markdown(table)
    path.write_text(
        "# Qwen3Guard-Gen CPU benchmark\n\n"
        + md
        + "\n\n"
        + PARAGRAPH
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
