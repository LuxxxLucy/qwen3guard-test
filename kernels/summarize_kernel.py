"""Collate the BLAS-backend sweep into a llama.cpp-cell x backend table.

run_kernel_cpu.sh rebuilds llama-cpp-python once per BLAS backend and writes
each backend's results under kernels/profile/<backend>/. This reads them and
prints one pivot per input template: each llama.cpp cell, p50 / p99 ms, one
column per backend. A wide p99/p50 gap means the host was contended for that
build — re-run that backend on a quiet box.

Reuses scripts/summarize_cpu.py for the result-file parsing and row keys, so
the kernel table and the cross-runtime table stay defined in one place.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from summarize_cpu import (  # noqa: E402
    ROWS, TEMPLATES, cell, latest_per_key, load_results, result_key,
)

PROFILE_DIR = Path("kernels/profile")
# L0 (the decode loop) is excluded: its batch-1 matmuls never route to BLAS.
LLAMACPP_ROWS = [(k, label.replace("llamacpp ", ""))
                 for k, label in ROWS if k[0] == "llamacpp" and k[2] != "L0"]


def main() -> int:
    backend_dirs = sorted(d for d in PROFILE_DIR.glob("*") if d.is_dir())
    if not backend_dirs:
        print(f"[summarize-kernel] no backend result dirs under {PROFILE_DIR}/")
        return 0
    backends = [d.name for d in backend_dirs]
    best = {}
    for d in backend_dirs:
        gen = [r for r in load_results(d)
               if r.get("variant") == "gen"
               and r.get("extra", {}).get("mode") == "representative"]
        best[d.name] = latest_per_key(gen, result_key)

    for tmpl in TEMPLATES:
        print(f"\nllama.cpp cell x BLAS backend — {tmpl}, p50 / p99 ms")
        table = [["cell", *backends]]
        for key, label in LLAMACPP_ROWS:
            table.append([label,
                          *(cell(best[b].get((*key, tmpl))) for b in backends)])
        w = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
        for r in table:
            print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
