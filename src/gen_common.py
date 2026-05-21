"""Shared Qwen3Guard-Gen helpers: chat-template rendering, forced-prefix
discovery, verdict-token-id discovery.

Runtime-agnostic — no torch / onnxruntime import here. The multi-runtime CPU
benchmark (`bench_gen_cpu.py`) and its backends (`gen_backends.py`) all build
the same L2 forced-prefix input through these helpers, so the only thing that
varies across runtimes is the forward pass itself.
"""
from __future__ import annotations


# Qwen3Guard-Gen is a 3-way classifier. The verdict is the first
# information-bearing token after "Safety: ".
VERDICT_LABELS: tuple[str, ...] = ("Safe", "Unsafe", "Controversial")


def render_prompt(tokenizer, user_text: str) -> str:
    """Apply the chat template with add_generation_prompt=True so the rendered
    prompt ends at `<|im_start|>assistant\\n`. Appending "Safety: " then lands
    cleanly at the start of the assistant response."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


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


def build_forced_ids(tokenizer, user_text: str) -> list[int]:
    """Token sequence for `chat_template(user_text) + "Safety: "` — the L2
    forced-prefix input. One forward over these ids reads the verdict."""
    forced_ids, _, _ = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, user_text)
    )
    return forced_ids


def discover_verdict_token_ids(tokenizer) -> dict[str, int]:
    """Discover the 3 verdict token ids (Safe / Unsafe / Controversial).

    The ids are tokenizer-dependent but prompt-independent in practice (the
    "Safety: " boundary is fixed text); two probes with different leading
    characters guard against a BPE merge that would make them prompt-varying.
    """
    _, _, ids1 = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, "Representative probe content for verdict id discovery."))
    _, _, ids2 = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, "Another arbitrary probe 12345 !@# so BPE can differ."))
    if ids1 != ids2:
        raise RuntimeError(
            f"verdict token ids vary across prompts; cannot precompute a shared "
            f"readout. probe1={ids1} probe2={ids2}"
        )
    return ids1
