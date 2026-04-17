"""Qwen3Guard-Stream PyTorch batch=1 latency benchmark.

Measures two distinct regimes because the Stream model is designed for
incremental inference:
  (A) prefill: classify a full user prompt in one call (whole-prompt latency).
  (B) per-token: classify each assistant token as it arrives (steady-state).

Uses the custom `stream_moderate_from_ids(token_ids, role, stream_state)` API
exposed by the trust_remote_code model.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from bench_common import (
    BenchResult, LatencyStats, load_representative_texts, synthesize_input_ids,
    pick_device, pick_dtype, sync, write_result,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Stream-0.6B")
    ap.add_argument("--n-samples", type=int, default=50,
                    help="Number of prompts (prefill regime). Per-token samples "
                         "are ~n-samples * stream-tokens.")
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--stream-tokens", type=int, default=64,
                    help="Number of assistant tokens to stream per prompt for "
                         "the per-token regime.")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=None,
                    help="If set, sweep prefill latency across these input "
                         "token counts (synthetic prompts). One result file "
                         "per length.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = args.device or pick_device()
    dtype = pick_dtype(device)

    print(f"[bench-stream-pt] model={args.model_id} device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id, dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    # Build input token sequences using the chat template (user turn only).
    def user_ids(text: str) -> list[int]:
        messages = [{"role": "user", "content": text}]
        return tok.apply_chat_template(messages, tokenize=True,
                                       add_generation_prompt=True)

    # Template overhead = ids produced by an empty-content user turn.
    template_overhead = len(user_ids(""))

    sweep_meta: dict[int, dict] = {}
    if args.lengths:
        # Build user content of L tokens then wrap with the chat template so
        # the sweep reflects real classifier input shape. Report user /
        # template / total so the reader sees the overhead floor.
        id_pools: list[tuple[int, list[list[int]]]] = []
        for L in args.lengths:
            ui = synthesize_input_ids(tok, target_tokens=L)
            utext = tok.decode(ui, skip_special_tokens=True)
            user_tokens_actual = len(tok.encode(utext, add_special_tokens=False))
            ids_full = user_ids(utext)
            total_tokens = len(ids_full)
            pool = [ids_full] * args.n_samples
            id_pools.append((L, pool))
            sweep_meta[L] = {
                "user_tokens_actual": user_tokens_actual,
                "template_overhead_tokens": template_overhead,
                "total_input_tokens": total_tokens,
            }
    else:
        samples = load_representative_texts(max_samples=args.n_samples, tokenizer=tok)
        if not samples:
            print("[err] no samples.", file=sys.stderr)
            return 1
        id_pools = [(None, [user_ids(t) for t in samples])]

    # A synthetic assistant continuation (fixed string) for per-token streaming.
    # We tokenize it once and reuse across prompts; this isolates the classifier
    # cost from any generator cost.
    assistant_text = (
        "I understand your question. Here is a detailed response that tries to "
        "be helpful while staying within safe guidelines. Let me walk through "
        "the reasoning step by step so that the answer is clear and well "
        "structured."
    )
    asst_ids_full = tok.encode(assistant_text, add_special_tokens=False)
    asst_ids = asst_ids_full[: args.stream_tokens]

    # The custom stream API expects torch tensors (the model calls .to(device)
    # on them internally).
    def run_one(uid: list[int], asst: list[int]) -> tuple[float, list[float]]:
        uid_t = torch.tensor(uid, dtype=torch.long)
        asst_t = torch.tensor(asst, dtype=torch.long)

        sync(device)
        t0 = time.perf_counter()
        _, state = model.stream_moderate_from_ids(uid_t, role="user", stream_state=None)
        sync(device)
        prefill = time.perf_counter() - t0

        per_tok: list[float] = []
        for i in range(asst_t.numel()):
            tid = asst_t[i]
            sync(device)
            tt = time.perf_counter()
            _, state = model.stream_moderate_from_ids(tid, role="assistant",
                                                     stream_state=state)
            sync(device)
            per_tok.append(time.perf_counter() - tt)

        if hasattr(model, "close_stream"):
            try:
                model.close_stream(state)
            except Exception:
                pass
        return prefill, per_tok

    for target_len, id_lists in id_pools:
        for i in range(min(args.n_warmup, len(id_lists))):
            run_one(id_lists[i], asst_ids)

        prefill_lat: list[float] = []
        pertoken_lat: list[float] = []
        for uid in id_lists:
            p, pt = run_one(uid, asst_ids)
            prefill_lat.append(p)
            pertoken_lat.extend(pt)

        median_tokens = int(statistics.median(len(x) for x in id_lists))
        tag = f"sweep-len{target_len}" if target_len is not None else "representative"
        extra: dict = {
            "mode": tag,
            "target_user_tokens": target_len,
            "regime": "prefill",
            "per_token": LatencyStats.from_samples(pertoken_lat).__dict__,
            "chat_template_applied": True,
        }
        if target_len is not None and target_len in sweep_meta:
            extra.update(sweep_meta[target_len])
        res = BenchResult(
            variant="stream",
            runtime="pytorch",
            model_id=args.model_id,
            device=device,
            dtype=str(dtype).replace("torch.", ""),
            provider=None,
            n_samples=len(prefill_lat),
            n_warmup=args.n_warmup,
            input_token_count_median=median_tokens,
            output_token_count=args.stream_tokens,
            latency=LatencyStats.from_samples(prefill_lat),
            extra=extra,
        )
        path = write_result(res, Path(args.out_dir))
        if target_len is not None:
            m = sweep_meta[target_len]
            print(f"[bench-stream-pt] ({tag}  user={m['user_tokens_actual']}  "
                  f"template={m['template_overhead_tokens']}  "
                  f"total={m['total_input_tokens']}) wrote {path}")
        else:
            print(f"[bench-stream-pt] ({tag} len={median_tokens}) wrote {path}")
        print(f"[bench-stream-pt] prefill p50={res.latency.p50_ms:.1f}ms "
              f"p95={res.latency.p95_ms:.1f}ms  "
              f"per-token p50={res.extra['per_token']['p50_ms']:.2f}ms "
              f"p95={res.extra['per_token']['p95_ms']:.2f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
