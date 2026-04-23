"""Shared benchmark utilities: device/dtype selection, sample pools, latency stats."""
from __future__ import annotations

import json
import os
import platform
import random
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str):
    import torch
    return torch.bfloat16 if device == "cuda" else torch.float32


def sync(device: str) -> None:
    import torch
    if device == "cuda":
        torch.cuda.synchronize()


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    stdev_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_rps: float

    @staticmethod
    def from_samples(samples_s: list[float]) -> "LatencyStats":
        ms = [s * 1000.0 for s in samples_s]
        ms_sorted = sorted(ms)
        n = len(ms_sorted)

        def pct(p: float) -> float:
            if n == 0:
                return 0.0
            k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return ms_sorted[k]

        mean_ms = statistics.fmean(ms) if ms else 0.0
        stdev_ms = statistics.stdev(ms) if n > 1 else 0.0
        throughput = (1000.0 / mean_ms) if mean_ms > 0 else 0.0
        return LatencyStats(
            n=n, mean_ms=mean_ms, stdev_ms=stdev_ms,
            p50_ms=pct(50), p95_ms=pct(95), p99_ms=pct(99),
            min_ms=ms_sorted[0] if ms else 0.0,
            max_ms=ms_sorted[-1] if ms else 0.0,
            throughput_rps=throughput,
        )


@dataclass
class BenchResult:
    variant: str           # "gen" | "stream"
    runtime: str           # "pytorch" | "onnx"
    model_id: str
    device: str
    dtype: str
    provider: str | None   # ONNX EP, else None
    n_samples: int
    n_warmup: int
    input_token_count_median: int
    output_token_count: int | None  # Gen: max_new_tokens; Stream: None (per-token)
    latency: LatencyStats
    extra: dict = field(default_factory=dict)
    timestamp_utc: str = ""
    host: str = ""
    torch_version: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def write_result(res: BenchResult, out_dir: Path) -> Path:
    import torch
    from datetime import datetime, timezone
    res.timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res.host = platform.node()
    res.torch_version = torch.__version__
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = res.timestamp_utc.replace(":", "").replace("-", "")
    name = f"bench_{res.variant}_{res.runtime}_{res.device}_{stamp}.json"
    path = out_dir / name
    path.write_text(json.dumps(res.to_dict(), indent=2))
    return path


def load_representative_texts(
    max_samples: int = 100,
    tokenizer=None,
    target_median_band_pct: float = 0.10,
    seed: int = 0,
) -> list[str]:
    """Load Qwen3GuardTest response field, pick samples near the median length.

    Falls back to a fixed synthetic list if dataset unavailable.
    """
    fallback = _fallback_texts()
    try:
        from datasets import load_dataset
        # Qwen3GuardTest splits: thinking, response_loc, thinking_loc.
        # response_loc holds user/assistant dialogues; we take the assistant
        # text (the classified content) as the representative sample.
        ds = load_dataset("Qwen/Qwen3GuardTest", split="response_loc")
    except Exception as e:
        print(f"[warn] Qwen3GuardTest unavailable ({e!r}); using fallback.")
        return fallback[:max_samples]

    texts: list[str] = []
    if "message" in ds.column_names:
        for msg in ds["message"]:
            # Prefer the assistant turn; fall back to the user turn.
            picked = None
            for m in msg:
                if m.get("role") == "assistant" and m.get("content"):
                    picked = m["content"]
                    break
            if picked is None and msg:
                picked = msg[0].get("content")
            if picked and isinstance(picked, str):
                texts.append(picked)
    else:
        for cand in ("response", "prompt", "text", "content"):
            if cand in ds.column_names:
                texts = [t for t in ds[cand] if isinstance(t, str) and t.strip()]
                break
    if not texts:
        return fallback[:max_samples]

    if tokenizer is None:
        # length by char count as a coarse proxy
        lens = [len(t) for t in texts]
    else:
        lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]

    median = statistics.median(lens)
    band = target_median_band_pct
    pool = [t for t, n in zip(texts, lens) if (1 - band) * median <= n <= (1 + band) * median]
    if len(pool) < max_samples:
        # widen band until we have enough
        for b in (0.2, 0.3, 0.5, 1.0):
            pool = [t for t, n in zip(texts, lens) if (1 - b) * median <= n <= (1 + b) * median]
            if len(pool) >= max_samples:
                break

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_samples] if pool else fallback[:max_samples]


def synthesize_input_ids(tokenizer, target_tokens: int, seed: int = 0) -> list[int]:
    """Produce a list of token ids with length exactly `target_tokens` by
    concatenating/truncating a natural-sounding seed paragraph. Used for
    length-sweep latency benchmarks where we want a clean length axis.
    """
    seed_text = (
        "The quick brown fox jumps over the lazy dog near the old wooden "
        "bridge while the river flows silently through the valley. Across "
        "the hills, small villages dot the landscape and children play in "
        "the fields under a bright blue sky. "
    )
    ids: list[int] = []
    while len(ids) < target_tokens:
        ids.extend(tokenizer.encode(seed_text, add_special_tokens=False))
    return ids[:target_tokens]


def synthesize_prompts(tokenizer, target_tokens: int, n: int, seed: int = 0) -> list[str]:
    """Return n text prompts each encoding to approximately `target_tokens`
    tokens. A small per-sample salt keeps them non-identical (cache doesn't
    trivially collapse them), while length stays controlled.
    """
    base_ids = synthesize_input_ids(tokenizer, target_tokens, seed=seed)
    out: list[str] = []
    rng = random.Random(seed)
    for i in range(n):
        # Swap in a few random tokens (<5%) so requests are distinct but
        # length-comparable.
        ids = list(base_ids)
        k = max(1, len(ids) // 40)
        for _ in range(k):
            j = rng.randrange(len(ids))
            ids[j] = rng.randrange(tokenizer.vocab_size)
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def _fallback_texts() -> list[str]:
    return [
        "How do I bake sourdough bread at home?",
        "Explain the concept of photosynthesis to a 10-year-old.",
        "What are the best practices for writing clean Python code?",
        "Describe the process of starting a small business from scratch.",
        "Can you help me draft a professional email requesting a raise?",
        "What are the major differences between TCP and UDP protocols?",
        "How does the human immune system respond to viral infections?",
        "Summarize the plot of Shakespeare's play Hamlet in 3 sentences.",
        "What are effective strategies for managing workplace stress?",
        "Explain how neural networks learn via backpropagation.",
        "Give me a recipe for chicken tikka masala that serves four.",
        "What are the main causes of climate change according to scientists?",
        "Help me prepare interview questions for a software engineering role.",
        "Describe three key events that led to World War I.",
        "How can I improve my posture while working at a desk all day?",
        "Explain the difference between supervised and unsupervised learning.",
        "What is the historical significance of the Silk Road trade routes?",
        "Suggest a weekly workout plan for someone new to the gym.",
        "How does encryption protect data transmitted over the internet?",
        "What are the ethical considerations of using AI in healthcare today?",
    ]


def warmup_and_measure(
    step: Callable[[str], None],
    samples: list[str],
    n_warmup: int,
    device: str,
) -> list[float]:
    for i in range(min(n_warmup, len(samples))):
        step(samples[i])
    sync(device)

    latencies: list[float] = []
    for s in samples:
        sync(device)
        t0 = time.perf_counter()
        step(s)
        sync(device)
        latencies.append(time.perf_counter() - t0)
    return latencies
