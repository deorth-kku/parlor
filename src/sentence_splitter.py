from __future__ import annotations

import re

ASCII_SENTENCE_ENDINGS = ".!?"
CJK_SENTENCE_ENDINGS = "。！？"
SENTENCE_ENDINGS = ASCII_SENTENCE_ENDINGS + CJK_SENTENCE_ENDINGS

COMMON_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
}

LETTER_DOT_ABBREVIATION_RE = re.compile(r"(?:\b[A-Za-z]\.){2,}$")
TRAILING_ALPHA_DOT_RE = re.compile(r"([A-Za-z.]+)$")


def _next_significant_index(text: str, start: int) -> int:
    i = start
    while i < len(text) and text[i] in "\"')]} \t\r\n":
        i += 1
    return i


def _looks_like_abbreviation(text: str, period_index: int) -> bool:
    if text[period_index] != ".":
        return False

    previous = text[: period_index + 1].rstrip()
    if not previous:
        return False

    next_index = _next_significant_index(text, period_index + 1)
    following = text[next_index:]

    if period_index > 0 and text[period_index - 1].isdigit() and following[:1].isdigit():
        return True

    token_match = TRAILING_ALPHA_DOT_RE.search(previous)
    if not token_match:
        return False

    token = token_match.group(1)
    token_lower = token.lower()
    if token_lower in COMMON_ABBREVIATIONS:
        return True

    if LETTER_DOT_ABBREVIATION_RE.fullmatch(token):
        return True

    if len(token) == 2 and token[0].isalpha() and following[:1].isalpha():
        return True

    return len(token) <= 4 and token[:-1].isalpha() and token[0].isupper()


def append_sentence_buffer(buffer: str, chunk: str) -> tuple[str, list[str]]:
    text = buffer + chunk
    sentences: list[str] = []
    start = 0
    i = 0

    while i < len(text):
        ch = text[i]
        if ch not in SENTENCE_ENDINGS:
            i += 1
            continue

        if ch == "." and i + 1 < len(text) and text[i + 1] == ".":
            i += 1
            continue

        next_index = _next_significant_index(text, i + 1)
        if _looks_like_abbreviation(text, i):
            i += 1
            continue

        sentence = text[start : i + 1].strip()
        if sentence:
            sentences.append(sentence)
        start = next_index
        i = start

    return text[start:], sentences
