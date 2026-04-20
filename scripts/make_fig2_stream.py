"""Figure 2 (stream): Qwen3Guard-Stream-0.6B prefill P99 vs per-token P99 across the length sweep.

Run:
  uv run --with matplotlib python scripts/make_fig2_stream.py

Output: figures/fig2_stream_length_0.6b.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

USER_TOKENS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
PREFILL_P99    = [27.1, 27.5, 29.4, 38.9, 58.2, 108.1, 288.3, 457.3]
PER_TOKEN_P99  = [28.9, 29.8, 31.6, 41.3, 61.5, 110.2, 231.4, 450.5]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(USER_TOKENS, PREFILL_P99,   "o-", label="prefill P99",    linewidth=2, markersize=7, color="#2980b9")
    ax.plot(USER_TOKENS, PER_TOKEN_P99, "s-", label="per-token P99", linewidth=2, markersize=7, color="#27ae60")

    ax.axhline(8.0, linestyle="--", linewidth=1, color="#c0392b", label="8 ms T2 budget")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(USER_TOKENS)
    ax.set_xticklabels([str(n) for n in USER_TOKENS])
    ax.set_xlabel("User-content tokens (log₂ scale)")
    ax.set_ylabel("P99 latency (ms, log scale)")
    ax.set_title("Qwen3Guard-Stream-0.6B — prefill vs per-token P99 (RTX 3090, bf16, batch=1)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    out = Path(__file__).resolve().parent.parent / "figures" / "fig2_stream_length_0.6b.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
