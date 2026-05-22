"""Shared Qwen3Guard-Gen helpers: input rendering, forced-prefix discovery,
verdict-token-id discovery.

Runtime-agnostic — no torch / onnxruntime import here. The multi-runtime CPU
benchmark (`bench_gen_cpu.py`) and its backends (`gen_backends.py`) all build
the same L2 forced-prefix input through these helpers, so the only thing that
varies across runtimes is the forward pass itself.
"""
from __future__ import annotations


# Qwen3Guard-Gen is a 3-way classifier. The verdict is the first
# information-bearing token after "Safety: ".
VERDICT_LABELS: tuple[str, ...] = ("Safe", "Unsafe", "Controversial")

# Selectable input templates (--template). Both render to the same shape and
# end at the assistant `<think>` block, so appending "Safety: " lands at the
# start of the verdict line either way; they differ only in policy-block length.
TEMPLATES: tuple[str, ...] = ("original", "test-200")

# `test-200`: a compressed safety policy. The built-in Qwen3Guard template adds
# ~296 fixed tokens; this one adds ~130, so a representative input (~68 user
# tokens) lands near a 200-token total. It is a latency-measurement template —
# its verdict accuracy is not validated against the fine-tuned original prompt.
# Split PRE/POST around the user content: PRE is the byte-identical prefix every
# call shares — the unit a system-prompt KV cache would precompute once.
_TEST_200_PRE = (
    "<|im_start|>user\n"
    "# Task: classify THE LAST USER query under the safety policy below.\n\n"
    "Safe = no prohibited content; Unsafe = contains prohibited content; "
    "Controversial = not explicit but exploitable in context.\n"
    "Unsafe categories: Violent; Non-violent Illegal Acts; Sexual Content; PII; "
    "Suicide & Self-Harm; Unethical Acts; Politically Sensitive Topics; "
    "Copyright Violation; Jailbreak.\n\n"
    "<BEGIN CONVERSATION>\n\nUSER: "
)
_TEST_200_POST = (
    "\n\n<END CONVERSATION>\n\n"
    "Answer with 'Safety: Safe', 'Safety: Unsafe', or 'Safety: Controversial'."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def render_prompt(tokenizer, user_text: str, template: str = "original") -> str:
    """Render the classifier input for `user_text`.

    `template="original"` applies the model's built-in Qwen3Guard chat template
    (the prompt it was fine-tuned on); `template="test-200"` uses the compressed
    policy above. Both end at the assistant `<think>` block, so appending
    "Safety: " lands cleanly at the start of the verdict line."""
    if template == "original":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if template == "test-200":
        return _TEST_200_PRE + user_text + _TEST_200_POST
    raise ValueError(f"unknown template {template!r}; choose from {TEMPLATES}")


def discover_forced_prefix(
    tokenizer, rendered_prompt: str,
) -> tuple[list[int], int, dict[str, int]]:
    """Given a rendered chat prompt, return (forced_ids, diff_pos, verdict_ids).

    `forced_ids` is the tokenization of `rendered_prompt + "Safety: "` — the
    model, forwarded over these tokens, predicts the first verdict subtoken at
    the last position. `verdict_ids` maps each verdict label to the token id at
    `diff_pos` (the first subtoken where Safe/Unsafe/Controversial diverge).

    BPE can merge "Safety: " with the preceding marker differently depending on
    what came before, so the divergence point is probed per prompt.
    """
    tokenized = {
        v: tokenizer.encode(rendered_prompt + f"Safety: {v}", add_special_tokens=False)
        for v in VERDICT_LABELS
    }
    min_len = min(len(t) for t in tokenized.values())
    diff_pos: int | None = None
    for i in range(min_len):
        if len({t[i] for t in tokenized.values()}) > 1:
            diff_pos = i
            break
    if diff_pos is None:
        raise RuntimeError(
            "forced-prefix probe failed: Safe/Unsafe/Controversial tokenized to "
            "the same prefix up to min_len — no divergence point."
        )
    verdict_ids = {v: tokenized[v][diff_pos] for v in VERDICT_LABELS}
    if len(set(verdict_ids.values())) != 3:
        raise RuntimeError(
            f"forced-prefix probe failed: verdict first-subtokens collide: {verdict_ids}"
        )
    return tokenized["Safe"][:diff_pos], diff_pos, verdict_ids


def build_forced_ids(tokenizer, user_text: str, template: str = "original") -> list[int]:
    """Token sequence for `render_prompt(user_text) + "Safety: "` — the L2
    forced-prefix input. One forward over these ids reads the verdict."""
    forced_ids, _, _ = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, user_text, template)
    )
    return forced_ids


def build_plain_ids(tokenizer, user_text: str, template: str = "original") -> list[int]:
    """Token sequence for the plain rendered prompt, no trailing "Safety: " —
    the L0 decode-loop input."""
    return tokenizer.encode(
        render_prompt(tokenizer, user_text, template), add_special_tokens=False
    )


def common_prefix(seqs: list[list[int]]) -> list[int]:
    """Longest leading token run shared by every sequence, capped one short of
    the shortest so each sequence keeps a non-empty suffix. The KV-cache path
    primes this shared prefix once and forwards only each suffix."""
    first = seqs[0]
    limit = min(len(s) for s in seqs) - 1
    p = 0
    while p < limit and all(s[p] == first[p] for s in seqs):
        p += 1
    return first[:p]


def discover_verdict_token_ids(tokenizer, template: str = "original") -> dict[str, int]:
    """Discover the 3 verdict token ids (Safe / Unsafe / Controversial).

    The ids are tokenizer-dependent but prompt-independent in practice (the
    "Safety: " boundary is fixed text); two probes with different leading
    characters guard against a BPE merge that would make them prompt-varying.
    """
    _, _, ids1 = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, "Representative probe content for verdict id discovery.", template))
    _, _, ids2 = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, "Another arbitrary probe 12345 !@# so BPE can differ.", template))
    if ids1 != ids2:
        raise RuntimeError(
            f"verdict token ids vary across prompts; cannot precompute a shared "
            f"readout. probe1={ids1} probe2={ids2}"
        )
    return ids1
