"""Verify the active llama-cpp-python build matches the BLAS config it claims.

run_kernel_cpu.sh calls this after each rebuild. uv keys its build cache by
source-dist hash, not by CMAKE_ARGS, so without a cache eviction every config
silently reuses one wheel. This probe is the gate that catches that: it reads
the BLAS library actually linked into the extension and exits non-zero if it
does not match the expected backend.

Usage:  uv run python kernels/build_probe.py <tinyblas|openblas|blis|mkl|kleidiai>
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

# expected backend -> substring of a linked library name, or None for a build
# that links no external BLAS (tinyblas keeps its own GEMM; kleidiai compiles
# its microkernels in).
EXPECT = {"tinyblas": None, "openblas": "openblas", "blis": "blis",
          "mkl": "mkl", "kleidiai": None}
VENDORS = ("openblas", "blis", "mkl")


def linked_vendor_libs() -> list[str]:
    import llama_cpp
    libdir = os.path.dirname(llama_cpp.__file__)
    shared = (glob.glob(os.path.join(libdir, "**", "*.so"), recursive=True)
              + glob.glob(os.path.join(libdir, "**", "*.dylib"), recursive=True))
    linker = ["otool", "-L"] if sys.platform == "darwin" else ["ldd"]
    hits = []
    for so in shared:
        out = subprocess.run(linker + [so], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if any(v in line.lower() for v in VENDORS):
                hits.append(line.strip())
    return hits


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in EXPECT:
        print(f"usage: build_probe.py <{'|'.join(EXPECT)}>")
        return 2
    want = sys.argv[1]

    import llama_cpp
    try:
        llama_cpp.llama_backend_init()
        info = llama_cpp.llama_print_system_info()
        print("[probe] " + (info.decode() if isinstance(info, bytes) else info).strip())
    except Exception as e:  # system info is informational; the link check gates
        print(f"[probe] llama_print_system_info unavailable: {e!r}")

    hits = linked_vendor_libs()
    for h in hits:
        print(f"[probe] linked: {h}")
    if not hits:
        print("[probe] linked: no external BLAS library")

    expect_sub = EXPECT[want]
    ok = (not hits) if expect_sub is None else any(expect_sub in h.lower()
                                                   for h in hits)
    print(f"[probe] {want}: {'OK' if ok else 'MISMATCH — measured the wrong build'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
