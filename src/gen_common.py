"""Qwen3Guard-Gen helpers: rendering, forced-prefix discovery, verdict-token-id
discovery. Runtime-agnostic — no torch / onnxruntime import here."""
from __future__ import annotations

from contract import Template, TEMPLATES


# Qwen3Guard-Gen is a 3-way classifier.
VERDICT_LABELS: tuple[str, ...] = ("Safe", "Unsafe", "Controversial")

# `test-200`: compressed safety policy (~130-token overhead vs the built-in
# Qwen3Guard ~296). PRE / POST split around the user content so the prefix
# (everything before user_text) is byte-identical per call — the unit a
# system-prompt KV cache precomputes once.
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


def render_prompt(tokenizer, user_text: str, template: Template = "original") -> str:
    """Classifier input. Ends at the assistant `<think>` block; appending
    `"Safety: "` lands at the start of the verdict line."""
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
    """Return (forced_ids, diff_pos, verdict_ids) for `rendered + "Safety: "`.
    diff_pos is the first index where Safe/Unsafe/Controversial diverge."""
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
        raise RuntimeError("forced-prefix probe: no divergence point.")
    verdict_ids = {v: tokenized[v][diff_pos] for v in VERDICT_LABELS}
    if len(set(verdict_ids.values())) != 3:
        raise RuntimeError(f"verdict first-subtokens collide: {verdict_ids}")
    return tokenized["Safe"][:diff_pos], diff_pos, verdict_ids


def build_forced_ids(
    tokenizer, user_text: str, template: Template = "original",
) -> list[int]:
    """L2 forced-prefix input: tokenize(render + "Safety: ")."""
    forced_ids, _, _ = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, user_text, template)
    )
    return forced_ids


def build_plain_ids(
    tokenizer, user_text: str, template: Template = "original",
) -> list[int]:
    """L0 decode-loop input: tokenize(render), no trailing "Safety: "."""
    return tokenizer.encode(
        render_prompt(tokenizer, user_text, template), add_special_tokens=False
    )


def common_prefix(seqs: list[list[int]]) -> list[int]:
    """Longest leading token run shared by every sequence, capped one short of
    the shortest so each sequence keeps a non-empty suffix."""
    first = seqs[0]
    limit = min(len(s) for s in seqs) - 1
    p = 0
    while p < limit and all(s[p] == first[p] for s in seqs):
        p += 1
    return first[:p]


def extract_verdict(generated_text: str) -> str:
    """Parse `Safety: <verdict>` from a Qwen3Guard-Gen response."""
    t = generated_text.strip()
    if t.startswith("Safety:"):
        after = t[len("Safety:"):].strip()
        return after.split()[0] if after else ""
    # Fallback: check more specific labels first so "Safe" doesn't shadow a
    # "Controversial" / "Unsafe" later in the text.
    for v in ("Controversial", "Unsafe", "Safe"):
        if v in t:
            return v
    return f"OTHER({t[:40]!r})"


def discover_verdict_token_ids(
    tokenizer, template: Template = "original",
) -> dict[str, int]:
    """Return {label: token_id}. Two probes with different leading chars guard
    against BPE merges that would make the ids prompt-varying."""
    _, _, ids1 = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, "Representative probe content for verdict id discovery.", template))
    _, _, ids2 = discover_forced_prefix(
        tokenizer, render_prompt(tokenizer, "Another arbitrary probe 12345 !@# so BPE can differ.", template))
    if ids1 != ids2:
        raise RuntimeError(f"verdict ids vary across prompts: {ids1} vs {ids2}")
    return ids1
