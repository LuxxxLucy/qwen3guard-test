"""Print one comparison table from the CPU benchmark result JSONs.

Scans results/ for the files written by bench_gen_cpu.py and
bench_stream_pytorch.py (--device cpu) and prints a runtime x precision
latency table to stdout. Run at the end of scripts/run_cpu.sh so the whole
benchmark is readable from a single stdout stream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[warn] could not read {p.name}: {e!r}")
    return out


def latest_per_key(rows: list[dict], key) -> list[dict]:
    """Keep the most recent result per key (later timestamp wins)."""
    best: dict = {}
    for r in rows:
        k = key(r)
        if k not in best or r.get("timestamp_utc", "") >= best[k].get("timestamp_utc", ""):
            best[k] = r
    return [best[k] for k in sorted(best)]


def print_gen_table(rows: list[dict]) -> None:
    gen = [r for r in rows
           if r.get("variant") == "gen" and r.get("device") == "cpu"
           and r.get("extra", {}).get("mode") == "representative"]
    if not gen:
        print("  (no Gen CPU representative results)")
        return
    gen = latest_per_key(gen, lambda r: (r.get("runtime", ""), r.get("dtype", "")))
    print(f"  {'runtime':<10} {'precision':<9} {'threads':>7} {'in_tok':>7} "
          f"{'p50_ms':>9} {'p95_ms':>9} {'p99_ms':>9} {'rps':>8}")
    print("  " + "-" * 74)
    for r in gen:
        lat = r["latency"]
        thr = r.get("extra", {}).get("threads")
        print(f"  {r.get('runtime',''):<10} {r.get('dtype',''):<9} "
              f"{str(thr if thr is not None else 'default'):>7} "
              f"{r.get('input_token_count_median',0):>7} "
              f"{lat['p50_ms']:>9.1f} {lat['p95_ms']:>9.1f} {lat['p99_ms']:>9.1f} "
              f"{lat['throughput_rps']:>8.2f}")


def print_stream_table(rows: list[dict]) -> None:
    stream = [r for r in rows
              if r.get("variant") == "stream" and r.get("device") == "cpu"
              and r.get("extra", {}).get("mode") == "representative"]
    if not stream:
        print("  (no Stream CPU results — bench_stream_direct.py prints its own table above)")
        return
    stream = latest_per_key(stream, lambda r: r.get("runtime", ""))
    print(f"  {'runtime':<10} {'in_tok':>7} {'prefill_p50':>12} {'prefill_p99':>12} "
          f"{'pertok_p50':>11} {'pertok_p99':>11}")
    print("  " + "-" * 68)
    for r in stream:
        pf = r["latency"]
        pt = r.get("extra", {}).get("per_token", {})
        print(f"  {r.get('runtime',''):<10} {r.get('input_token_count_median',0):>7} "
              f"{pf['p50_ms']:>12.2f} {pf['p99_ms']:>12.2f} "
              f"{pt.get('p50_ms',0):>11.3f} {pt.get('p99_ms',0):>11.3f}")


def main() -> int:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if not results_dir.exists():
        print(f"[summarize] no results dir at {results_dir}")
        return 0
    rows = load_results(results_dir)

    print()
    print("Qwen3Guard-Gen CPU — L2 forced-prefix, representative input")
    print_gen_table(rows)
    print()
    print("Qwen3Guard-Stream CPU — prefill + per-token, representative input")
    print_stream_table(rows)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
