"""Figure 1 (stream): Qwen3Guard-Stream-0.6B, stock API vs direct path.

Shows the per-token P99 on 0.6B across input lengths for four paths:
  - Stock stream_moderate_from_ids (k=1) — the shipped API.
  - Direct model.forward(...) with DynamicCache (k=1, k=8, k=16).

Run:
  uv run --with matplotlib python scripts/make_fig1_stream.py

Output: figures/fig1_stream_sizes.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

USER_TOKENS = [256, 1024, 2048, 4096]

# Stock API per-token P99 (ms), 0.6B. Source: initial length sweep via
# stream_moderate_from_ids, one token per .send() call.
STOCK_API_K1 = [41.3, 110.2, 231.4, 450.5]

# Direct path per-token P99 (ms), 0.6B, heads inside timed region.
DIRECT_K1  = [18.60, 18.25, 17.41, 19.06]
DIRECT_K8  = [ 2.67,  2.70,  2.57,  2.53]
DIRECT_K16 = [ 1.34,  1.35,  1.28,  1.29]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.plot(USER_TOKENS, STOCK_API_K1, "o-", label="stock API (k=1)",
            linewidth=2, markersize=7, color="#c0392b")
    ax.plot(USER_TOKENS, DIRECT_K1,  "s-", label="direct path (k=1)",
            linewidth=2, markersize=7, color="#e67e22")
    ax.plot(USER_TOKENS, DIRECT_K8,  "^-", label="direct path (k=8)",
            linewidth=2, markersize=7, color="#2980b9")
    ax.plot(USER_TOKENS, DIRECT_K16, "D-", label="direct path (k=16)",
            linewidth=2, markersize=7, color="#27ae60")

    ax.axhline(3.0, linestyle="--", linewidth=1, color="#7f8c8d",
               label="3 ms T2 target")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(USER_TOKENS)
    ax.set_xticklabels([str(n) for n in USER_TOKENS])
    ax.set_xlabel("User-content tokens (log₂ scale)")
    ax.set_ylabel("Per-token P99 latency (ms, log scale)")
    ax.set_title("Qwen3Guard-Stream-0.6B — stock API vs direct path "
                 "(RTX 3090, bf16, batch=1)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    out = Path(__file__).resolve().parent.parent / "figures" / "fig1_stream_sizes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
