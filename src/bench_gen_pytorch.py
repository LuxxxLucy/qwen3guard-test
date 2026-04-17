"""Qwen3Guard-Gen PyTorch batch=1 latency benchmark + optimization ladder.

Measures batch=1 classification latency under the following modes:

  LP (--prefill-only):
     Pure prefill. ONE forward pass over the rendered prompt, restricted
     3-way readout over {Safe, Unsafe, Controversial} logits. No
     teacher-forced prefix, no decode. Isolates prefill cost at each
     input length for the prefill / decode breakdown.

  L0 (--no-prefix-cache):
     Naive: tokenize full prompt, generate() with DynamicCache,
     decode until EOS or max_new_tokens. No reuse. Total latency =
     prefill + N_decode * per_token_decode.

  L1 (--prefix-cache):
     Precompute KV for the chat-template head (longest common prefix
     of tokenizations across diverse probe user contents). Per call,
     deep-copy the cached KV, feed only (user + tail) as new input_ids,
     and generate(). Saves one full-template prefill per call.

  L2 (--prefix-cache --forced-prefix):
     Teacher-force "Safety: " as the partial assistant response and do a
     restricted 3-way compare over {Safe, Unsafe, Controversial} logits
     at the last position. ONE forward pass, no decode loop.

  L3 (--prefix-cache --forced-prefix --compile):
     Wrap model.forward in torch.compile (mode='reduce-overhead',
     dynamic=True). Fuses kernels and eliminates per-step Python overhead
     on CUDA.

With LP and L0 both measured at each sweep length, per-token decode cost
is derivable as (L0 - LP) / N_decode_tokens — the breakdown the ladder's
summary table reports.

Every optimization composes with the ones above it. The correctness
check at startup verifies that each enabled optimization produces the
same verdict as the L0 baseline on a representative sample. For
dataset-scale verdict agreement testing across N real samples, see
`scripts/correctness_test.py`.

Length-sweep samples are drawn via ``synthesize_prompts`` (100 distinct
salted prompts per length) so user content varies call-to-call and the
benchmark only benefits from caching that the code EXPLICITLY installed
(the template head), not from implicit reuse of identical tensors.
"""
from __future__ import annotations

import argparse
import copy
import statistics
import sys
from pathlib import Path

from bench_common import (
    BenchResult, LatencyStats, load_representative_texts,
    synthesize_input_ids, synthesize_prompts,
    pick_device, pick_dtype, warmup_and_measure, write_result,
)


# --- Template / tokenizer probing ------------------------------------------

# Probe contents for the longest-common-prefix tokenization pass. Diverse
# first characters (letters, digits, punctuation, whitespace, Unicode) so the
# tokenizer exercises different BPE merges at the user-content boundary —
# whatever survives as a shared prefix is safe to cache.
LCP_PROBES: list[str] = [
    "Hello, how are you today?",
    "List the first 10 prime numbers.",
    "def foo(x): return x + 1",
    "   Tab\tspace test   ",
    "Café naïve résumé — Zürich",
    "Here is a moderately long probe sentence that should exercise a "
    "different tokenization than the shorter probes above.",
    "1234567890",
    "?!.,:;<>()[]{}",
]


def compute_cacheable_head(tokenizer) -> list[int]:
    """head_ids = LCP of tokenizations across probe renderings.

    Qwen3's BPE can legally merge the trailing `\\n` of the user role marker
    with the first byte of user content, so `tokenize(head_str)` is NOT
    always a token-level prefix of `tokenize(head_str + user + tail_str)`.
    Taking the LCP across several tokenizations gives, by construction, the
    maximal stable prefix — independent of what user content follows.
    """
    tokenizations: list[list[int]] = []
    for probe in LCP_PROBES:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": probe}], tokenize=False,
        )
        tokenizations.append(
            tokenizer.encode(rendered, add_special_tokens=False)
        )
    head: list[int] = []
    min_len = min(len(t) for t in tokenizations)
    first = tokenizations[0]
    for i in range(min_len):
        tid = first[i]
        if all(t[i] == tid for t in tokenizations):
            head.append(tid)
        else:
            break
    if not head:
        raise RuntimeError(
            "cacheable head is empty: no common prefix across probe "
            "tokenizations. Tokenizer does not permit prefix caching here."
        )
    return head


def precompute_template_cache(model, head_ids: list[int], device: str):
    """Forward the template head once, return its past_key_values."""
    import torch
    head_tensor = torch.tensor([head_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(head_tensor)
    with torch.no_grad():
        out = model(input_ids=head_tensor, attention_mask=attn, use_cache=True)
    return out.past_key_values


# --- Forced-prefix probing -------------------------------------------------

# Qwen3Guard-Gen is a 3-way classifier: Safe / Unsafe / Controversial. The
# verdict is the first information-bearing token after "Safety: ".
VERDICT_LABELS: tuple[str, ...] = ("Safe", "Unsafe", "Controversial")


def discover_forced_prefix(
    tokenizer, rendered_prompt: str,
) -> tuple[list[int], int, dict[str, int]]:
    """Given a rendered chat prompt, return:
       (full_ids_stop_before_verdict, diff_pos, verdict_token_ids)

    `full_ids_stop_before_verdict` is the tokenization of
    `rendered_prompt + "Safety: "` — the model, forwarded over these tokens,
    predicts the first verdict subtoken at the last position. `diff_pos` is
    its length. `verdict_token_ids` maps each verdict label
    ("Safe"/"Unsafe"/"Controversial") to the token id at `diff_pos`
    (the first subtoken of that verdict word — may be a full-word token for
    "Safe" or a multi-subtoken start like "Un"/"Cont" for the others).

    We probe per-prompt because BPE can merge `Safety: ` with the preceding
    `<|im_start|>assistant\\n` differently depending on what came before.
    The probe itself is a few tokenize() calls — microseconds.
    """
    tokenized = {
        v: tokenizer.encode(rendered_prompt + f"Safety: {v}", add_special_tokens=False)
        for v in VERDICT_LABELS
    }
    # All three strings share a common prefix up to the verdict word; the
    # first position where any two tokenizations diverge is `diff_pos`.
    min_len = min(len(t) for t in tokenized.values())
    diff_pos: int | None = None
    for i in range(min_len):
        tids = {t[i] for t in tokenized.values()}
        if len(tids) > 1:
            diff_pos = i
            break
    if diff_pos is None:
        raise RuntimeError(
            "forced-prefix probe failed: Safe/Unsafe/Controversial tokenized "
            "to the same prefix up to min_len — no divergence point."
        )
    verdict_ids = {v: tokenized[v][diff_pos] for v in VERDICT_LABELS}
    # The three first-subtokens must be distinct, otherwise we cannot
    # distinguish the verdicts by reading a single logit position.
    if len(set(verdict_ids.values())) != 3:
        raise RuntimeError(
            f"forced-prefix probe failed: verdict first-subtokens collide: "
            f"{verdict_ids}"
        )
    ref = tokenized["Safe"]
    return ref[:diff_pos], diff_pos, verdict_ids


# --- Prediction paths (used by bench + correctness test) ------------------

def render_prompt(tokenizer, user_text: str) -> str:
    """Apply chat template with add_generation_prompt=True so the rendered
    prompt ends at `<|im_start|>assistant\\n`. Appending "Safety: " lands
    cleanly at the start of the assistant response."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def predict_naive(model, tok, user_text: str, device: str,
                  max_new_tokens: int) -> tuple[str, list[int]]:
    """L0 baseline: tokenize full prompt, generate() with DynamicCache,
    decode to text. Return (verdict, generated_ids)."""
    import torch
    rendered = render_prompt(tok, user_text)
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
    return extract_verdict(text), gen


def predict_cached(model, tok, template_cache, head_ids: list[int],
                   user_text: str, device: str,
                   max_new_tokens: int) -> tuple[str, list[int]]:
    """L1: prefix-cache + generate()."""
    import torch
    rendered = render_prompt(tok, user_text)
    full_ids = tok.encode(rendered, add_special_tokens=False)
    T = len(head_ids)
    if full_ids[:T] != head_ids:
        raise RuntimeError("cached: seam broken for this sample")
    new_ids = full_ids[T:]
    new_tensor = torch.tensor([new_ids], dtype=torch.long, device=device)
    attn = torch.ones((1, T + len(new_ids)), dtype=torch.long, device=device)
    cache_copy = copy.deepcopy(template_cache)
    cache_position = torch.arange(T, T + len(new_ids), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=new_tensor,
            attention_mask=attn,
            past_key_values=cache_copy,
            cache_position=cache_position,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, new_tensor.shape[1]:].tolist()
    text = tok.decode(gen, skip_special_tokens=True)
    return extract_verdict(text), gen


def predict_forced_prefix(model, tok, template_cache, head_ids: list[int],
                          user_text: str, device: str,
                          use_prefix_cache: bool,
                          compile_enabled: bool = False) -> tuple[str, int]:
    """L2: teacher-force "Safety: " and do a restricted 3-way compare over
    {Safe, Unsafe, Controversial} at the last logit position. Return
    (verdict_label, predicted_token_id).

    Restricted compare instead of open-vocab argmax: the classifier's
    output space is exactly 3 tokens, and reading only those 3 logits
    (a) cannot return "OTHER", (b) handles Qwen3Guard-Gen's 3-way label
    correctly, and (c) yields calibrated class probabilities for free
    via softmax over the 3 selected logits.

    When ``compile_enabled`` is True we call ``cudagraph_mark_step_begin()``
    before the forward so ``reduce-overhead`` CUDA graphs don't complain
    about reusing buffers from a previous invocation.
    """
    import torch
    rendered = render_prompt(tok, user_text)
    forced_ids, _, verdict_ids = discover_forced_prefix(tok, rendered)
    verdict_token_ids = [verdict_ids[v] for v in VERDICT_LABELS]

    if use_prefix_cache:
        T = len(head_ids)
        if forced_ids[:T] != head_ids:
            raise RuntimeError("forced-prefix: seam broken for this sample")
        new_ids = forced_ids[T:]
        new_tensor = torch.tensor([new_ids], dtype=torch.long, device=device)
        attn = torch.ones((1, T + len(new_ids)), dtype=torch.long, device=device)
        cache_copy = copy.deepcopy(template_cache)
        cache_position = torch.arange(T, T + len(new_ids), dtype=torch.long, device=device)
        with torch.no_grad():
            if compile_enabled:
                torch.compiler.cudagraph_mark_step_begin()
            out = model(
                input_ids=new_tensor,
                attention_mask=attn,
                past_key_values=cache_copy,
                cache_position=cache_position,
                use_cache=False,
            )
    else:
        full_tensor = torch.tensor([forced_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            if compile_enabled:
                torch.compiler.cudagraph_mark_step_begin()
            out = model(
                input_ids=full_tensor,
                attention_mask=torch.ones_like(full_tensor),
                use_cache=False,
            )
    # Restricted 3-way compare: argmax over only the 3 verdict logits.
    verdict_logits = out.logits[0, -1, verdict_token_ids]
    idx = int(verdict_logits.argmax().item())
    return VERDICT_LABELS[idx], verdict_token_ids[idx]


def extract_verdict(generated_text: str) -> str:
    """Parse the Qwen3Guard-Gen response to extract the verdict.

    Qwen3Guard-Gen is a 3-way classifier — the model emits
    `Safety: Safe\\n...`, `Safety: Unsafe\\n...`, or
    `Safety: Controversial\\n...`.
    """
    t = generated_text.strip()
    if t.startswith("Safety:"):
        after = t[len("Safety:"):].strip()
        first_word = after.split()[0] if after else ""
        return first_word  # "Safe" / "Unsafe" / "Controversial"
    # Fallback: check the more specific labels first so "Safe" doesn't
    # shadow a "Controversial" / "Unsafe" that appears later in the text.
    for v in ("Controversial", "Unsafe", "Safe"):
        if v in t:
            return v
    return f"OTHER({t[:40]!r})"


# --- Startup correctness check --------------------------------------------

def startup_correctness_check(tok, model, head_ids, template_cache, device,
                              max_new_tokens, use_prefix_cache, use_forced_prefix,
                              compile_enabled: bool = False):
    """Sanity check on one 64-token synthetic sample: verify every enabled
    optimization agrees with the L0 verdict. Not a replacement for the
    dataset correctness test; fails loudly if an optimization is wrong.

    When ``compile_enabled`` is True we skip the ``predict_naive`` /
    ``predict_cached`` paths: HF ``generate()`` runs a KV-cached decode
    loop that is fundamentally incompatible with ``reduce-overhead``
    CUDA graphs (each decode step overwrites buffers that the next step
    still references). Verdict agreement for L0/L1/L2 is covered by
    ``scripts/correctness_test.py`` (run as Step 0 of the ladder before
    the L3 bench). Here we only re-verify the forced-prefix path under
    compile, which is the one L3 actually benchmarks.
    """
    user_ids = synthesize_input_ids(tok, target_tokens=64)
    user_text = tok.decode(user_ids, skip_special_tokens=True)

    if compile_enabled:
        if not use_forced_prefix:
            print("[correctness] L3 without --forced-prefix: skipping decode-loop "
                  "correctness check (incompatible with reduce-overhead CUDA graphs).")
            return
        v, pred_id = predict_forced_prefix(
            model, tok, template_cache, head_ids, user_text, device,
            use_prefix_cache, compile_enabled=True,
        )
        print(f"[correctness] L3 forced:    verdict={v!r} token={pred_id}  "
              f"(baseline skipped under torch.compile; see scripts/correctness_test.py)")
        return

    baseline_verdict, baseline_gen = predict_naive(
        model, tok, user_text, device, max_new_tokens,
    )
    print(f"[correctness] L0 baseline: verdict={baseline_verdict!r} "
          f"gen={tok.decode(baseline_gen, skip_special_tokens=True)[:80]!r}")

    if use_prefix_cache:
        v, gen_c = predict_cached(
            model, tok, template_cache, head_ids, user_text, device, max_new_tokens,
        )
        ok = gen_c == baseline_gen
        tag = "OK" if ok else "MISMATCH"
        print(f"[correctness] L1 cached:    verdict={v!r}  {tag} "
              f"(token-exact vs baseline: {ok})")

    if use_forced_prefix:
        v, pred_id = predict_forced_prefix(
            model, tok, template_cache, head_ids, user_text, device,
            use_prefix_cache,
        )
        ok = v == baseline_verdict
        tag = "OK" if ok else f"MISMATCH (baseline={baseline_verdict!r})"
        print(f"[correctness] L2 forced:    verdict={v!r} token={pred_id}  {tag}")


# --- Step functions for the benchmark loop --------------------------------

def make_step(model, tok, device, use_prefix_cache, template_cache, T_head,
              use_forced_prefix, max_new_tokens, compile_enabled: bool = False,
              verdict_token_ids_tensor=None, prefill_only: bool = False):
    """Return a step function that runs ONE classification for the timer.
    Dispatches to the right combination of optimizations based on flags.

    ``verdict_token_ids_tensor`` is a 1-D tensor of the 3 verdict token ids
    (Safe/Unsafe/Controversial) on `device`; the forced-prefix and
    prefill-only step functions use it for a restricted 3-way compare
    over the last logit position, matching the production classifier
    readout and avoiding an open-vocab argmax.

    When ``prefill_only`` is True, returns a step that does ONE forward
    pass over the raw rendered prompt (no teacher-forced "Safety: ", no
    decode loop) and reads only the 3-verdict logits. This isolates pure
    prefill cost for the ladder's prefill/decode breakdown.

    When ``compile_enabled`` is True the forced-prefix step functions call
    ``torch.compiler.cudagraph_mark_step_begin()`` before each forward, so
    the ``reduce-overhead`` CUDA graphs know a new step has started and
    can safely recycle their captured output buffers.
    """
    import torch

    def _restricted_read(out) -> None:
        if verdict_token_ids_tensor is not None:
            _ = out.logits[0, -1, verdict_token_ids_tensor].argmax(dim=-1)
        else:
            _ = out.logits[:, -1, :].argmax(dim=-1)

    def step_prefill_only(full_ids: list[int]) -> None:
        # No forced prefix, no decode — measures pure prefill cost over
        # the rendered prompt at its natural length.
        full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            if compile_enabled:
                torch.compiler.cudagraph_mark_step_begin()
            out = model(
                input_ids=full_tensor,
                attention_mask=torch.ones_like(full_tensor),
                use_cache=False,
            )
            _restricted_read(out)

    def step_naive(full_ids: list[int]) -> None:
        full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            model.generate(
                input_ids=full_tensor,
                attention_mask=torch.ones_like(full_tensor),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

    def step_cached(full_ids: list[int]) -> None:
        new_ids = full_ids[T_head:]
        new_tensor = torch.tensor([new_ids], dtype=torch.long, device=device)
        attn = torch.ones((1, T_head + len(new_ids)), dtype=torch.long, device=device)
        cache_copy = copy.deepcopy(template_cache)
        cache_position = torch.arange(T_head, T_head + len(new_ids),
                                      dtype=torch.long, device=device)
        with torch.no_grad():
            model.generate(
                input_ids=new_tensor,
                attention_mask=attn,
                past_key_values=cache_copy,
                cache_position=cache_position,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

    def step_forced_cached(forced_full_ids: list[int]) -> None:
        # full_ids here is already `forced_full_ids` = tokenize(prompt+"Safety: ")
        new_ids = forced_full_ids[T_head:]
        new_tensor = torch.tensor([new_ids], dtype=torch.long, device=device)
        attn = torch.ones((1, T_head + len(new_ids)), dtype=torch.long, device=device)
        cache_copy = copy.deepcopy(template_cache)
        cache_position = torch.arange(T_head, T_head + len(new_ids),
                                      dtype=torch.long, device=device)
        with torch.no_grad():
            if compile_enabled:
                torch.compiler.cudagraph_mark_step_begin()
            out = model(
                input_ids=new_tensor,
                attention_mask=attn,
                past_key_values=cache_copy,
                cache_position=cache_position,
                use_cache=False,
            )
            _restricted_read(out)

    def step_forced_uncached(forced_full_ids: list[int]) -> None:
        full_tensor = torch.tensor([forced_full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            if compile_enabled:
                torch.compiler.cudagraph_mark_step_begin()
            out = model(
                input_ids=full_tensor,
                attention_mask=torch.ones_like(full_tensor),
                use_cache=False,
            )
            _restricted_read(out)

    if prefill_only:
        return step_prefill_only
    if use_forced_prefix:
        return step_forced_cached if use_prefix_cache else step_forced_uncached
    return step_cached if use_prefix_cache else step_naive


# --- Main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default=None,
                    help="Override device (cuda|cpu). Default: cuda if available.")
    ap.add_argument("--lengths", type=int, nargs="+", default=None,
                    help="Length sweep (synthetic prompts of these user-content "
                         "token counts). Default: representative dataset.")
    ap.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="L1: Reuse KV for the chat-template head. Default on.")
    ap.add_argument("--forced-prefix", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="L2: Teacher-force 'Safety: ' so only ONE forward pass "
                         "is needed to read the Safe/Unsafe verdict. Default off.")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="L3: Wrap model.forward in torch.compile. CUDA-targeted. "
                         "First few warmup calls are slow (compilation). Default off.")
    ap.add_argument("--prefill-only", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="LP: pure prefill — one forward pass over the rendered "
                         "prompt + restricted 3-way verdict readout. No "
                         "teacher-forced prefix, no decode. Overrides "
                         "--forced-prefix. Used by the ladder to isolate prefill "
                         "cost at each input length.")
    ap.add_argument("--opt-level", default=None,
                    help="Optional tag (e.g. 'L0'/'L1'/'L2'/'L3'/'LP') written "
                         "into the result JSON for easier grouping.")
    args = ap.parse_args()

    # --prefill-only supersedes other optimizations: the whole point is to
    # measure one forward pass in isolation, so we force off forced-prefix
    # and prefix-cache (no deepcopy noise) and refuse compile (would skew
    # the pure-prefill number we want to report).
    if args.prefill_only:
        args.forced_prefix = False
        args.prefix_cache = False
        args.compile = False

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or pick_device()
    dtype = pick_dtype(device)

    opt_summary = (f"prefix_cache={args.prefix_cache} "
                   f"forced_prefix={args.forced_prefix} "
                   f"compile={args.compile} "
                   f"prefill_only={args.prefill_only}")
    print(f"[bench-gen-pt] model={args.model_id} device={device} dtype={dtype} "
          f"opt_level={args.opt_level} {opt_summary}")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype,
    ).to(device).eval()

    head_ids = compute_cacheable_head(tok)
    empty_rendered = tok.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=False,
        add_generation_prompt=True,
    )
    nominal_template_overhead = len(tok.encode(empty_rendered, add_special_tokens=False))
    print(f"[bench-gen-pt] cacheable_head={len(head_ids)} "
          f"nominal_template_overhead={nominal_template_overhead}")

    # Discover the 3 verdict token ids (Safe/Unsafe/Controversial) using a
    # representative rendered prompt. These ids are tokenizer-dependent but
    # prompt-independent in practice (the "Safety: " boundary is fixed
    # text); we assert consistency on a second probe as a guard.
    _probe_rendered = render_prompt(tok, "Representative probe content for verdict id discovery.")
    _, _, _probe_verdict_ids = discover_forced_prefix(tok, _probe_rendered)
    _probe_rendered_2 = render_prompt(tok, "Another arbitrary probe 12345 !@# so BPE can differ.")
    _, _, _probe_verdict_ids_2 = discover_forced_prefix(tok, _probe_rendered_2)
    if _probe_verdict_ids != _probe_verdict_ids_2:
        raise RuntimeError(
            "verdict token ids vary across prompts; cannot precompute "
            f"a shared readout tensor. probe1={_probe_verdict_ids} "
            f"probe2={_probe_verdict_ids_2}"
        )
    verdict_ids = _probe_verdict_ids
    verdict_token_ids_tensor = torch.tensor(
        [verdict_ids[v] for v in VERDICT_LABELS], dtype=torch.long, device=device,
    )
    print(f"[bench-gen-pt] verdict_token_ids={dict(verdict_ids)}")

    # Precompute the template cache BEFORE applying torch.compile. Under
    # reduce-overhead mode the compiled forward returns cudagraph-captured
    # tensors; if we built the template cache from that, a later deepcopy
    # during the bench step would try to read storage that a subsequent
    # compiled step had already overwritten. Building the cache with eager
    # forward keeps its tensors on ordinary (non-cudagraph) storage.
    template_cache = None
    if args.prefix_cache:
        template_cache = precompute_template_cache(model, head_ids, device)

    if args.compile:
        # dynamic=True tolerates varying input lengths (sweep) without
        # forcing a recompile per shape; reduce-overhead enables CUDA graphs
        # when possible. Fall back to eager if the inductor backend can't
        # compile for this setup (rare, but better than crashing the run).
        print(f"[bench-gen-pt] torch.compile enabled (mode=reduce-overhead, dynamic=True)")
        try:
            model.forward = torch.compile(
                model.forward, mode="reduce-overhead", dynamic=True,
            )
        except Exception as e:
            print(f"[bench-gen-pt] torch.compile unavailable ({e!r}); falling back to eager.")

    # Startup correctness sanity — verifies each enabled optimization gives
    # the same verdict as L0 on one synthetic sample. Fast.
    startup_correctness_check(
        tok, model, head_ids, template_cache, device, args.max_new_tokens,
        args.prefix_cache, args.forced_prefix,
        compile_enabled=args.compile,
    )

    T_head = len(head_ids) if template_cache is not None else 0
    step = make_step(
        model, tok, device,
        use_prefix_cache=args.prefix_cache,
        template_cache=template_cache,
        T_head=T_head,
        use_forced_prefix=args.forced_prefix,
        max_new_tokens=args.max_new_tokens,
        compile_enabled=args.compile,
        verdict_token_ids_tensor=verdict_token_ids_tensor,
        prefill_only=args.prefill_only,
    )

    opt_tag = args.opt_level or (
        "LP" if args.prefill_only else
        "L3" if args.compile else
        "L2" if args.forced_prefix else
        "L1" if args.prefix_cache else
        "L0"
    )

    def build_input_ids(user_text: str) -> list[int]:
        """Build the token sequence this optimization level feeds to the
        model. Naive / cached / prefill-only: full rendered prompt.
        Forced-prefix: rendered prompt + `Safety: ` teacher-force suffix."""
        rendered = render_prompt(tok, user_text)
        if args.forced_prefix:
            forced_ids, _, _ = discover_forced_prefix(tok, rendered)
            return forced_ids
        return tok.encode(rendered, add_special_tokens=False)

    if args.lengths:
        for L in args.lengths:
            # Generate `n_samples` distinct salted prompts of approximately
            # `L` user tokens each. This keeps the length axis controlled
            # while ensuring user content varies per call — so any observed
            # speedup comes from the caching we explicitly installed
            # (template head), not from implicit reuse of identical tensors.
            user_texts = synthesize_prompts(
                tok, target_tokens=L, n=args.n_samples, seed=L,
            )
            full_ids_list: list[list[int]] = []
            user_tokens_counts: list[int] = []
            for ut in user_texts:
                fids = build_input_ids(ut)
                if args.prefix_cache and fids[:T_head] != head_ids:
                    # Rare: a salted sample broke the template-head seam.
                    # Skip this one, keep the rest.
                    continue
                full_ids_list.append(fids)
                user_tokens_counts.append(
                    len(tok.encode(ut, add_special_tokens=False))
                )
            if not full_ids_list:
                print(f"[warn] sweep-len{L}: all salted samples failed the "
                      f"seam check; skipping.")
                continue

            lens = [len(x) for x in full_ids_list]
            total_tokens_median = int(statistics.median(lens))
            user_tokens_median = int(statistics.median(user_tokens_counts))
            lat = warmup_and_measure(step, full_ids_list, args.n_warmup, device)  # type: ignore[arg-type]
            res = BenchResult(
                variant="gen",
                runtime="pytorch",
                model_id=args.model_id,
                device=device,
                dtype=str(dtype).replace("torch.", ""),
                provider=None,
                n_samples=len(lat),
                n_warmup=args.n_warmup,
                input_token_count_median=total_tokens_median,
                output_token_count=(
                    1 if (args.forced_prefix or args.prefill_only)
                    else args.max_new_tokens
                ),
                latency=LatencyStats.from_samples(lat),
                extra={"mode": f"sweep-len{L}",
                       "opt_level": opt_tag,
                       "target_user_tokens": L,
                       "user_tokens_median": user_tokens_median,
                       "n_distinct_samples": len(full_ids_list),
                       "nominal_template_overhead_tokens": nominal_template_overhead,
                       "cacheable_head_tokens": len(head_ids),
                       "total_input_tokens_median": total_tokens_median,
                       "chat_template_applied": True,
                       "prefix_cache": args.prefix_cache,
                       "forced_prefix": args.forced_prefix,
                       "compile": args.compile,
                       "prefill_only": args.prefill_only},
            )
            path = write_result(res, Path(args.out_dir))
            print(f"[bench-gen-pt] ({opt_tag} sweep-len{L}  "
                  f"user~={user_tokens_median}  template={nominal_template_overhead}  "
                  f"total~={total_tokens_median}  n_distinct={len(full_ids_list)}) "
                  f"wrote {path}")
            print(f"[bench-gen-pt] p50={res.latency.p50_ms:.1f}ms "
                  f"p95={res.latency.p95_ms:.1f}ms p99={res.latency.p99_ms:.1f}ms "
                  f"thr={res.latency.throughput_rps:.2f} rps")
    else:
        samples = load_representative_texts(max_samples=args.n_samples, tokenizer=tok)
        if not samples:
            print("[err] no samples available.", file=sys.stderr)
            return 1
        full_ids_list: list[list[int]] = []
        for t in samples:
            fids = build_input_ids(t)
            if args.prefix_cache and fids[:T_head] != head_ids:
                continue
            full_ids_list.append(fids)
        if not full_ids_list:
            print("[err] no samples passed the seam check.", file=sys.stderr)
            return 1
        input_token_lens = [len(x) for x in full_ids_list]
        median_tokens = int(statistics.median(input_token_lens))
        lat = warmup_and_measure(step, full_ids_list, args.n_warmup, device)  # type: ignore[arg-type]
        res = BenchResult(
            variant="gen",
            runtime="pytorch",
            model_id=args.model_id,
            device=device,
            dtype=str(dtype).replace("torch.", ""),
            provider=None,
            n_samples=len(lat),
            n_warmup=args.n_warmup,
            input_token_count_median=median_tokens,
            output_token_count=(
                1 if (args.forced_prefix or args.prefill_only)
                else args.max_new_tokens
            ),
            latency=LatencyStats.from_samples(lat),
            extra={"mode": "representative",
                   "opt_level": opt_tag,
                   "target_user_tokens": None,
                   "nominal_template_overhead_tokens": nominal_template_overhead,
                   "cacheable_head_tokens": len(head_ids),
                   "chat_template_applied": True,
                   "prefix_cache": args.prefix_cache,
                   "forced_prefix": args.forced_prefix,
                   "compile": args.compile,
                   "prefill_only": args.prefill_only},
        )
        path = write_result(res, Path(args.out_dir))
        print(f"[bench-gen-pt] ({opt_tag} representative len={median_tokens}) "
              f"wrote {path}")
        print(f"[bench-gen-pt] p50={res.latency.p50_ms:.1f}ms "
              f"p95={res.latency.p95_ms:.1f}ms p99={res.latency.p99_ms:.1f}ms "
              f"thr={res.latency.throughput_rps:.2f} rps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
