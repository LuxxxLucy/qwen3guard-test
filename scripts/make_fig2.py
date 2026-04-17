"""Figure 2: optimized P99 latency across 0.6B / 4B / 8B.

Run:
  uv run --with matplotlib python scripts/make_fig2.py

Output: figures/fig2_sizes_optimized.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

USER_TOKENS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
P99_06B = [33.9, 32.7, 36.6, 45.8, 61.5, 103.4, 190.1, 355.8]
P99_4B  = [89.6, 88.3, 97.2, 145.9, 206.8, 320.4, 541.6, 1206.9]
P99_8B  = [137.1, 126.4, 173.8, 231.0, 267.9, 450.3, 839.4, 1727.3]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(USER_TOKENS, P99_06B, "o-", label="0.6B", linewidth=2, markersize=7, color="#27ae60")
    ax.plot(USER_TOKENS, P99_4B,  "s-", label="4B",   linewidth=2, markersize=7, color="#2980b9")
    ax.plot(USER_TOKENS, P99_8B,  "^-", label="8B",   linewidth=2, markersize=7, color="#8e44ad")

    ax.axhline(200.0, linestyle="--", linewidth=1, color="#7f8c8d", label="200 ms budget")

    ax.set_xscale("log", base=2)
    ax.set_xticks(USER_TOKENS)
    ax.set_xticklabels([str(n) for n in USER_TOKENS])
    ax.set_xlabel("User-content tokens (log₂ scale)")
    ax.set_ylabel("P99 latency (ms)")
    ax.set_title("Qwen3Guard-Gen 0.6B / 4B / 8B — optimized inference (RTX 3090, bf16, batch=1)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(P99_8B) * 1.05)

    out = Path(__file__).resolve().parent.parent / "figures" / "fig2_sizes_optimized.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
