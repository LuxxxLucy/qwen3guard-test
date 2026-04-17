"""Generate Figure 1: Qwen3Guard-Gen-0.6B PyTorch default vs optimized P99 latency.

Run:
  uv run --with matplotlib python scripts/make_fig1.py

Output: figures/fig1_latency_0.6b.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

# Data from the RTX 3090 runs (see REPORT_GEN.md Appendix for raw tables).
USER_TOKENS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_P99 = [215.9, 218.3, 215.8, 221.5, 228.0, 277.4, 299.1, 457.6]
OPTIMIZED_P99 = [33.9, 32.7, 36.6, 45.8, 61.5, 103.4, 190.1, 355.8]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(USER_TOKENS, DEFAULT_P99, "o-", label="PyTorch default inference",
            linewidth=2, markersize=7, color="#c0392b")
    ax.plot(USER_TOKENS, OPTIMIZED_P99, "s-", label="optimized",
            linewidth=2, markersize=7, color="#27ae60")

    ax.set_xscale("log", base=2)
    ax.set_xticks(USER_TOKENS)
    ax.set_xticklabels([str(n) for n in USER_TOKENS])
    ax.set_xlabel("User-content tokens (log₂ scale)")
    ax.set_ylabel("P99 latency (ms)")
    ax.set_title("Qwen3Guard-Gen-0.6B — RTX 3090, bf16, batch=1")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(DEFAULT_P99) * 1.1)

    out = Path(__file__).resolve().parent.parent / "figures" / "fig1_latency_0.6b.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
