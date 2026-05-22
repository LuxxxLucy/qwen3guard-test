"""Dump pre-tokenized Qwen3Guard-Gen benchmark inputs for the Rust candle backend.

The Rust binary times only the model forward; tokenization and template rendering
stay in Python (gen_common), so the Rust and Python backends measure the same unit
of work. This writes one JSON the Rust binary reads.

Per template it dumps the L2 forced-prefix token sequences, the L0 plain
rendered-prompt token sequences, the shared system-prompt prefix, the verdict token
ids, and a 10-sample PyTorch-L2 verdict oracle for the Rust backend's self-check.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from bench_common import load_representative_texts
from contract import TEMPLATES
from gen_backends import PyTorchCPUBackend
from gen_common import (
    VERDICT_LABELS, build_forced_ids, build_plain_ids, common_prefix,
    discover_verdict_token_ids,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="rust/bench_inputs.json")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(args.model_id, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    texts = load_representative_texts(max_samples=args.n_samples, tokenizer=tok)
    print(f"[dump] model_path={model_path}")
    print(f"[dump] {len(texts)} representative samples")

    # Verdict token ids are template-independent (the "Safety: " boundary is
    # fixed text), and the PyTorch-L2 oracle is the same model for every
    # template — discover and load both once.
    verdict_ids = discover_verdict_token_ids(tok, TEMPLATES[0])
    verdict_token_ids = [verdict_ids[v] for v in VERDICT_LABELS]
    ref = PyTorchCPUBackend("fp32", verdict_token_ids, threads=None)
    ref.load(args.model_id, None, max_seq_len=0)

    dump = {
        "model_path": model_path,
        "model_id": args.model_id,
        "n_warmup": args.n_warmup,
        "max_new_tokens": args.max_new_tokens,
        "templates": {},
    }

    for template in TEMPLATES:
        forced = [build_forced_ids(tok, t, template) for t in texts]
        plain = [build_plain_ids(tok, t, template) for t in texts]
        prefix = common_prefix(forced)
        # PyTorch-L2 verdict oracle on the first 10 forced samples — the Rust
        # backend's optimized path checks its argmax against this.
        expected = [ref.predict(s) for s in forced[:10]]

        dump["templates"][template] = {
            "forced": forced,
            "plain": plain,
            "prefix": prefix,
            "verdict_token_ids": verdict_token_ids,
            "expected_verdicts": expected,
        }
        print(f"[dump] {template}: forced_med={int(statistics.median(len(s) for s in forced))} "
              f"plain_med={int(statistics.median(len(s) for s in plain))} "
              f"prefix={len(prefix)} verdicts={verdict_token_ids} expected={expected}")

    del ref

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dump))
    print(f"[dump] wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
