"""Accuracy check: L0 (naive generate) vs L2 (forced-prefix single forward) vs labels.

The existing correctness_test.py checks L2 *agrees with* L0 sample-by-sample on
the response_loc split (all-Unsafe) in prompt-moderation mode. That proves the
implementation of the optimization is sound, but it does not tell us whether
the model itself is accurate against ground-truth safety labels, and it does
not exercise the mode (response moderation) that a response-side A8 classifier
would actually use in production.

This script:

1. Loads the Qwen3GuardTest "thinking" split (which has both Safe and Unsafe
   labels), balanced Safe/Unsafe.
2. For each sample, renders the full [user, assistant] dialogue under Qwen3Guard's
   RESPONSE-MODERATION chat template (system prompt: "assess the last ASSISTANT's
   response"), matching how the Qwen3Guard model card intends the model to be used.
3. Runs L0 (naive generate + regex) and L2 (teacher-forced "Safety: " + restricted
   argmax over 3 verdict token ids) under that rendering.
4. Reports:
     - L0 accuracy vs dataset labels   (naive-inference accuracy)
     - L2 accuracy vs dataset labels   (optimized-inference accuracy)
     - L0 vs L2 per-sample agreement   (implementation correctness)

Interpretation: if (L0 acc) == (L2 acc) and L0/L2 agreement == 100%, the
forced-prefix optimization is correct: it reproduces the naive verdict
sample-for-sample, and therefore inherits accuracy verbatim.

Notes:
- Uses 0.6B by default (runs on CPU/MPS; CPU is most reliable on Mac).
- N defaults to 20 (balanced: 10 Safe + 10 Unsafe). Increase with --n for a
  tighter accuracy estimate; N=100 takes ~5-10 min on a laptop CPU.
- Qwen3Guard is a 3-way classifier (Safe/Unsafe/Controversial); the dataset
  is 2-way. A prediction of "Controversial" counts as incorrect regardless of
  label (it's neither "Safe" nor "Unsafe").
- Prior latency benchmarks used PROMPT-moderation (user-only) rendering on
  this same dataset; the latency numbers remain valid under that render
  (latency does not depend on semantic fit). This accuracy script diverges
  by using RESPONSE moderation, which is the correct semantic fit for
  labeling assistant responses.

Usage:
  uv run python scripts/accuracy_check.py --n 20 [--device cpu|mps|cuda]
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Reuse the verdict-label set, the forced-prefix probe, and the regex
# extractor. We deliberately do NOT reuse predict_naive / predict_forced_prefix,
# because those wrap prompt-moderation rendering; this script needs
# response-moderation rendering instead.
from bench_gen_pytorch import (  # noqa: E402
    VERDICT_LABELS, discover_forced_prefix, extract_verdict,
)


def autodetect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype_for(device: str):
    import torch
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # MPS bf16 coverage is patchy across ops for this model; fp16 is
        # the safe default. Users who see MPS matmul errors should fall
        # back to --device cpu.
        return torch.float16
    return torch.float32


def render_response_moderation(tok, user_text: str, assistant_text: str) -> str:
    """Render the full [user, assistant] dialogue under Qwen3Guard-Gen's
    response-moderation chat template. The template ends with
    `<|im_start|>assistant\n<think>\n\n</think>\n\n`, which is the start of
    the classifier's verdict turn -- appending "Safety: " lands cleanly at
    the first information-bearing position.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text},
         {"role": "assistant", "content": assistant_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_message_turns(msg_field) -> tuple[str, str] | None:
    """Return (user_text, assistant_text) or None if malformed."""
    if isinstance(msg_field, str):
        try:
            msg_field = ast.literal_eval(msg_field)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(msg_field, list):
        return None
    user_t, asst_t = None, None
    for m in msg_field:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user" and user_t is None:
            user_t = content
        elif role == "assistant" and asst_t is None:
            asst_t = content
    if user_t and asst_t:
        return (user_t, asst_t)
    return None


def load_labeled_samples(n_per_class: int, seed: int) -> list[tuple[str, str, str]]:
    """Return a list of (user_text, assistant_text, label) balanced across
    Safe / Unsafe."""
    import random
    from datasets import load_dataset

    ds = load_dataset("Qwen/Qwen3GuardTest", split="thinking")
    pool: dict[str, list[tuple[str, str]]] = {"Safe": [], "Unsafe": []}
    for row in ds:
        label = row.get("label")
        if label not in pool:
            continue
        turns = parse_message_turns(row.get("message"))
        if turns:
            pool[label].append(turns)

    rng = random.Random(seed)
    rng.shuffle(pool["Safe"])
    rng.shuffle(pool["Unsafe"])
    take = min(n_per_class, len(pool["Safe"]), len(pool["Unsafe"]))
    out: list[tuple[str, str, str]] = []
    for u, a in pool["Safe"][:take]:
        out.append((u, a, "Safe"))
    for u, a in pool["Unsafe"][:take]:
        out.append((u, a, "Unsafe"))
    rng.shuffle(out)
    return out


def predict_l0(model, tok, user_text: str, assistant_text: str,
               device: str, max_new_tokens: int) -> str:
    """Naive generate + regex extract, under response-moderation rendering."""
    import torch
    rendered = render_response_moderation(tok, user_text, assistant_text)
    full_ids = tok.encode(rendered, add_special_tokens=False)
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=full_tensor,
            attention_mask=torch.ones_like(full_tensor),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, full_tensor.shape[1]:].tolist()
    text = tok.decode(gen, skip_special_tokens=True)
    return extract_verdict(text)


def predict_l2(model, tok, user_text: str, assistant_text: str,
               device: str) -> str:
    """Teacher-force "Safety: " and take argmax over the 3 verdict token ids
    at the final logit position. Single forward pass."""
    import torch
    rendered = render_response_moderation(tok, user_text, assistant_text)
    forced_ids, _, verdict_ids = discover_forced_prefix(tok, rendered)
    verdict_token_ids = [verdict_ids[v] for v in VERDICT_LABELS]

    input_tensor = torch.tensor([forced_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_tensor)
    with torch.no_grad():
        out = model(input_ids=input_tensor, attention_mask=attn, use_cache=False)
    last_logits = out.logits[0, -1, :]
    selected = last_logits[verdict_token_ids]
    idx = int(selected.argmax().item())
    return VERDICT_LABELS[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--n", type=int, default=20,
                    help="Total samples (balanced: N/2 Safe + N/2 Unsafe). Default 20.")
    ap.add_argument("--max-new-tokens", type=int, default=24,
                    help="Upper bound for L0 generate. Response-moderation output "
                         "adds a 'Refusal:' line, so budget a few extra tokens.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or autodetect_device()
    dtype = pick_dtype_for(device)
    n_per_class = max(1, args.n // 2)
    n_total = n_per_class * 2

    print(f"[accuracy] model={args.model_id} device={device} dtype={dtype}")
    print(f"[accuracy] mode=RESPONSE_MODERATION (system='assess the last ASSISTANT's response')")
    print(f"[accuracy] requested n={args.n} -> balanced {n_per_class} Safe + {n_per_class} Unsafe")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype,
    ).to(device).eval()

    samples = load_labeled_samples(n_per_class=n_per_class, seed=args.seed)
    if len(samples) < n_total:
        print(f"[warn] only {len(samples)} samples available after balancing "
              f"(asked for {n_total}); continuing with what we have.")
    print(f"[accuracy] loaded {len(samples)} labeled samples "
          f"(Safe={sum(1 for *_,l in samples if l=='Safe')}, "
          f"Unsafe={sum(1 for *_,l in samples if l=='Unsafe')})")

    l0_correct = 0
    l2_correct = 0
    l0_l2_agree = 0
    confusion = {"L0": Counter(), "L2": Counter()}
    disagreements: list[tuple[int, str, str, str]] = []  # (idx, label, v_l0, v_l2)

    for i, (user_t, asst_t, label) in enumerate(samples):
        try:
            v0 = predict_l0(model, tok, user_t, asst_t, device, args.max_new_tokens)
        except Exception as e:
            print(f"  [{i}] L0 FAIL: {e!r}")
            continue
        try:
            v2 = predict_l2(model, tok, user_t, asst_t, device)
        except Exception as e:
            print(f"  [{i}] L2 FAIL: {e!r}")
            continue

        confusion["L0"][(label, v0)] += 1
        confusion["L2"][(label, v2)] += 1
        if v0 == label:
            l0_correct += 1
        if v2 == label:
            l2_correct += 1
        if v0 == v2:
            l0_l2_agree += 1
        else:
            disagreements.append((i, label, v0, v2))

        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}] "
                  f"L0_acc={l0_correct}/{i+1}  L2_acc={l2_correct}/{i+1}  "
                  f"agree={l0_l2_agree}/{i+1}")

    n = len(samples)
    print("\n[accuracy] === RESULTS ===")
    print(f"L0 accuracy vs labels:    {l0_correct}/{n} ({100*l0_correct/n:.1f}%)")
    print(f"L2 accuracy vs labels:    {l2_correct}/{n} ({100*l2_correct/n:.1f}%)")
    print(f"L0/L2 per-sample agree:   {l0_l2_agree}/{n} ({100*l0_l2_agree/n:.1f}%)")

    print("\nconfusion (truth -> predicted):")
    for path in ("L0", "L2"):
        print(f"  {path}:")
        for (truth, pred), c in sorted(confusion[path].items()):
            print(f"    {truth:>6} -> {pred:>14}  x{c}")

    if disagreements:
        print(f"\n{len(disagreements)} L0/L2 disagreements (first 10 shown):")
        for d in disagreements[:10]:
            print(f"  sample#{d[0]}  label={d[1]}  L0={d[2]}  L2={d[3]}")
    else:
        print("\n[accuracy] L0 and L2 agree on every sample — optimization is correct.")

    # Exit non-zero only if implementation disagrees with naive. Label-accuracy
    # gaps are informational (they depend on the model, not our optimization).
    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
