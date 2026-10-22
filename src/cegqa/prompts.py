"""Prompt construction (shared by prompting baselines, LoRA SFT and inference) and answer parsing."""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a question answering assistant. Answer the question using only the passage. "
    "Reply with the shortest span copied from the passage that answers the question, "
    "with no explanation."
)


def truncate_context(tokenizer, context: str, max_tokens: int, answers: list[str] | None = None) -> tuple[str, bool]:
    """Truncate `context` to at most `max_tokens` tokenizer tokens.

    Evaluation keeps the head of the passage. For training, pass `answers`: if the first
    answer occurrence falls outside the head window, the window is moved to contain it so
    that every training target stays grounded in the visible passage.
    Returns (text, was_truncated).
    """
    enc = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if len(offsets) <= max_tokens:
        return context, False
    start_tok = 0
    if answers:
        low = context.lower()
        pos = min((low.find(a.lower()) for a in answers if low.find(a.lower()) >= 0), default=-1)
        if pos >= 0 and pos >= offsets[max_tokens - 1][1] - 1:
            ans_tok = next(i for i, (s, e) in enumerate(offsets) if e > pos)
            start_tok = max(0, min(ans_tok - max_tokens // 2, len(offsets) - max_tokens))
    end_tok = start_tok + max_tokens
    return context[offsets[start_tok][0]: offsets[end_tok - 1][1]], True


def user_message(context: str, question: str) -> str:
    return f"Passage:\n{context}\n\nQuestion: {question}"


def build_messages(context: str, question: str, fewshot: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demo in fewshot or []:
        msgs.append({"role": "user", "content": user_message(demo["context"], demo["question"])})
        msgs.append({"role": "assistant", "content": demo["answers"][0]})
    msgs.append({"role": "user", "content": user_message(context, question)})
    return msgs


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Chat-template the conversation up to the assistant turn (Qwen3 thinking disabled)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_prompt(tokenizer, ex: dict, max_context_tokens: int, fewshot: list[dict] | None = None,
                 for_training: bool = False) -> tuple[str, bool]:
    ctx, truncated = truncate_context(
        tokenizer, ex["context"], max_context_tokens, ex["answers"] if for_training else None
    )
    return render_prompt(tokenizer, build_messages(ctx, ex["question"], fewshot)), truncated


_THINK = re.compile(r"<think>.*?</think>", re.S)
_PREFIX = re.compile(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:：]\s*", re.I)


def extract_answer(text: str) -> str:
    """Turn a raw generation into a short answer string."""
    text = _THINK.sub("", text).replace("<think>", "").replace("</think>", "").strip()
    if not text:
        return ""
    line = text.split("\n")[0].strip()
    line = _PREFIX.sub("", line).strip()
    line = line.strip("\"'`*").strip()
    if line.endswith(".") and not re.search(r"\b[A-Z]\.$", line):
        line = line[:-1].rstrip()
    return line
