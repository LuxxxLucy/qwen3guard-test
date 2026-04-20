"""Figure 1 (stream): per-token P99 across 0.6B / 4B / 8B and the length sweep.

Run:
  uv run --with matplotlib python scripts/make_fig1_stream.py

Output: figures/fig1_stream_sizes.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

USER_TOKENS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
PER_TOKEN_P99_06B = [28.9, 29.8, 31.6, 41.3, 61.5, 110.2, 231.4, 450.5]
PER_TOKEN_P99_4B  = [54.1, 56.1, 82.1, 127.1, 216.4, 347.8, 554.5, 1230.8]
PER_TOKEN_P99_8B  = [85.4, 85.5, 125.5, 231.9, 293.4, 504.6, 795.8, 1743.4]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(USER_TOKENS, PER_TOKEN_P99_06B, "o-", label="0.6B", linewidth=2, markersize=7, color="#27ae60")
    ax.plot(USER_TOKENS, PER_TOKEN_P99_4B,  "s-", label="4B",   linewidth=2, markersize=7, color="#2980b9")
    ax.plot(USER_TOKENS, PER_TOKEN_P99_8B,  "^-", label="8B",   linewidth=2, markersize=7, color="#8e44ad")

    ax.axhline(8.0, linestyle="--", linewidth=1, color="#c0392b", label="8 ms T2 budget")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(USER_TOKENS)
    ax.set_xticklabels([str(n) for n in USER_TOKENS])
    ax.set_xlabel("User-content tokens (log₂ scale)")
    ax.set_ylabel("Per-token P99 latency (ms, log scale)")
    ax.set_title("Qwen3Guard-Stream 0.6B / 4B / 8B — per-token P99 (RTX 3090, bf16, batch=1)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    out = Path(__file__).resolve().parent.parent / "figures" / "fig1_stream_sizes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
