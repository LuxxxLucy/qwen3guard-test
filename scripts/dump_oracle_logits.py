"""Dump pytorch fp32 L2 verdict logits per representative text.

Run on Mac and on Kunpeng with identical args; the JSONs are byte-comparable
through the diff helper at the bottom.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench_common import load_representative_texts
from gen_common import (VERDICT_LABELS, build_forced_ids,
                        discover_verdict_token_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--template", default="test-200", choices=["original", "test-200"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32,
    ).to("cpu").eval()

    verdict_ids = discover_verdict_token_ids(tok, template=args.template)
    vlist = [verdict_ids[v] for v in VERDICT_LABELS]

    texts = load_representative_texts(max_samples=args.n, tokenizer=tok)
    rows = []
    for i, txt in enumerate(texts):
        fid = build_forced_ids(tok, txt, args.template)
        t = torch.tensor([fid], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=t, attention_mask=torch.ones_like(t),
                        use_cache=False, logits_to_keep=1)
        row = out.logits[0, -1]
        vlogits = [float(row[v]) for v in vlist]
        rows.append({
            "idx": i,
            "text_head": txt[:60],
            "n_forced": len(fid),
            "verdict_logits": vlogits,
            "argmax_label": VERDICT_LABELS[vlogits.index(max(vlogits))],
        })

    payload = {
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "model_id": args.model_id,
        "template": args.template,
        "n": args.n,
        "verdict_token_ids": vlist,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}  host={payload['host']}  machine={payload['machine']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
