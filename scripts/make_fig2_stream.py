"""Figure 2 (stream): direct-path per-token P99 at k=16 across 3 sizes × 5 lengths.

The headline "T2 is met at k=16 across all sizes and lengths" figure.
All six lengths × three sizes stay under the 3 ms T2 target.

Run:
  uv run --with matplotlib python scripts/make_fig2_stream.py

Output: figures/fig2_stream_length_0.6b.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

USER_TOKENS = [81, 256, 1024, 2048, 4096]

# Direct path per-token P99 (ms) at k=16, heads inside timed region.
K16_06B = [1.38, 1.34, 1.35, 1.28, 1.29]
K16_4B  = [1.69, 1.60, 1.69, 1.64, 2.17]
K16_8B  = [1.74, 1.80, 1.91, 2.21, 2.75]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.plot(USER_TOKENS, K16_06B, "o-", label="0.6B",
            linewidth=2, markersize=7, color="#27ae60")
    ax.plot(USER_TOKENS, K16_4B,  "s-", label="4B",
            linewidth=2, markersize=7, color="#2980b9")
    ax.plot(USER_TOKENS, K16_8B,  "^-", label="8B",
            linewidth=2, markersize=7, color="#8e44ad")

    ax.axhline(3.0, linestyle="--", linewidth=1, color="#c0392b",
               label="3 ms T2 target")

    ax.set_xscale("log", base=2)
    ax.set_xticks(USER_TOKENS)
    ax.set_xticklabels([str(n) for n in USER_TOKENS])
    ax.set_xlabel("User-content tokens (log₂ scale)")
    ax.set_ylabel("Per-token P99 latency (ms)")
    ax.set_title("Qwen3Guard-Stream direct path at k=16 — per-token P99 across sizes "
                 "(RTX 3090, bf16, batch=1)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4.0)

    out = Path(__file__).resolve().parent.parent / "figures" / "fig2_stream_length_0.6b.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
