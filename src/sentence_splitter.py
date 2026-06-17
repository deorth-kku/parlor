from __future__ import annotations

import re

ASCII_SENTENCE_ENDINGS = ".!?"
CJK_SENTENCE_ENDINGS = "。！？"
SENTENCE_ENDINGS = ASCII_SENTENCE_ENDINGS + CJK_SENTENCE_ENDINGS
SOFT_BOUNDARIES = ",;，；、"
OPENING_TOKENS = "\"'([{"
CLOSING_TOKENS = "\"')]} \t\r\n"
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
ABBREVIATION_PATTERN_RE = re.compile(r"(?:[A-Za-z]\.){2,}$")
WORD_OR_CJK_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
DISCOURSE_STARTERS = {
    "ah",
    "alright",
    "anyway",
    "basically",
    "cool",
    "eh",
    "hey",
    "hmm",
    "honestly",
    "look",
    "maybe",
    "no",
    "now",
    "oh",
    "okay",
    "right",
    "so",
    "uh",
    "um",
    "well",
    "yeah",
    "yep",
    "yes",
}
VOWELS = set("aeiouAEIOU")

MIN_SOFT_SPLIT_WORDS = 8
MIN_SOFT_SPLIT_CHARS = 48
MIN_SOFT_SPLIT_TAIL_CHARS = 12
SOFT_BOUNDARY_REPEAT_THRESHOLD = 2


def _next_significant_index(text: str, start: int) -> int:
    i = start
    while i < len(text) and text[i] in CLOSING_TOKENS:
        i += 1
    return i


def _previous_significant_index(text: str, start: int) -> int:
    i = start
    while i >= 0 and text[i].isspace():
        i -= 1
    return i


def _extract_trailing_period_token(text: str, period_index: int) -> str:
    start = period_index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    return text[start : period_index + 1]


def _prefix_word_count(text: str, end: int) -> int:
    return len(WORD_OR_CJK_RE.findall(text[:end]))


def _tail_word_count(text: str, start: int, end: int) -> int:
    return len(WORD_OR_CJK_RE.findall(text[start:end]))


def _last_word_before(text: str, end: int) -> str:
    matches = list(LATIN_WORD_RE.finditer(text[:end]))
    if not matches:
        return ""
    return matches[-1].group(0)


def _first_word_after(text: str, start: int) -> str:
    match = LATIN_WORD_RE.search(text, start)
    if not match:
        return ""
    return match.group(0)


def _soft_boundary_counts(text: str, start: int, end: int) -> tuple[int, int]:
    commas = 0
    semicolons = 0
    for ch in text[start:end]:
        if ch in ",，":
            commas += 1
        elif ch in ";；":
            semicolons += 1
    return commas, semicolons


def _looks_like_abbreviation(text: str, period_index: int) -> bool:
    if text[period_index] != ".":
        return False

    previous_index = _previous_significant_index(text, period_index - 1)
    next_index = _next_significant_index(text, period_index + 1)

    if previous_index >= 0 and text[previous_index].isdigit() and next_index < len(text) and text[next_index].isdigit():
        return True

    token = _extract_trailing_period_token(text, period_index)
    if not token:
        return False

    bare = token[:-1]
    if not bare:
        return False

    abbreviation_like = False
    if ABBREVIATION_PATTERN_RE.fullmatch(token):
        abbreviation_like = True
    elif bare.isalpha():
        if len(bare) == 1:
            abbreviation_like = True
        elif bare.isupper() and len(bare) <= 3:
            abbreviation_like = True

    if next_index >= len(text):
        return abbreviation_like

    next_char = text[next_index]
    following_word = _first_word_after(text, next_index)

    if next_char.islower():
        if abbreviation_like:
            return True
        if bare.isalpha() and len(bare) <= 4:
            return True

    if following_word and following_word[0].islower():
        if abbreviation_like:
            return True
        if bare.isalpha() and len(bare) <= 4:
            return True

    if (
        bare.isalpha()
        and 1 < len(bare) <= 3
        and bare[0].isupper()
        and not any(ch in VOWELS for ch in bare[1:])
        and following_word
        and following_word[0].isupper()
    ):
        return True

    return abbreviation_like and bool(following_word) and following_word[0].isupper()


def _should_split_on_soft_boundary(text: str, start: int, boundary_index: int) -> bool:
    if text[boundary_index] not in SOFT_BOUNDARIES:
        return False

    prefix = text[start : boundary_index + 1]
    prefix_stripped = prefix.strip()
    if not prefix_stripped:
        return False

    word_count = _prefix_word_count(prefix_stripped, len(prefix_stripped))
    if word_count < MIN_SOFT_SPLIT_WORDS or len(prefix_stripped) < MIN_SOFT_SPLIT_CHARS:
        return False

    last_word = _last_word_before(text, boundary_index)
    if last_word and last_word.lower() in DISCOURSE_STARTERS and word_count <= MIN_SOFT_SPLIT_WORDS + 1:
        return False

    next_index = _next_significant_index(text, boundary_index + 1)
    if next_index >= len(text):
        return False

    next_char = text[next_index]
    if next_char in OPENING_TOKENS:
        return False

    tail_chars = len(text[next_index:].lstrip())
    if tail_chars < MIN_SOFT_SPLIT_TAIL_CHARS:
        return False

    tail_word_count = _tail_word_count(text, next_index, len(text))
    if tail_word_count < 3:
        return False

    commas, semicolons = _soft_boundary_counts(text, start, boundary_index + 1)

    if text[boundary_index] in ";；":
        return semicolons >= SOFT_BOUNDARY_REPEAT_THRESHOLD

    if semicolons > 0:
        return False

    return commas >= SOFT_BOUNDARY_REPEAT_THRESHOLD


def append_sentence_buffer(buffer: str, chunk: str) -> tuple[str, list[str]]:
    text = buffer + chunk
    sentences: list[str] = []
    start = 0
    i = 0

    while i < len(text):
        ch = text[i]

        if ch in SOFT_BOUNDARIES and _should_split_on_soft_boundary(text, start, i):
            next_index = _next_significant_index(text, i + 1)
            sentence = text[start : i + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = next_index
            i = start
            continue

        if ch not in SENTENCE_ENDINGS:
            i += 1
            continue

        if ch == "." and i + 1 < len(text) and text[i + 1] == ".":
            i += 1
            continue

        next_index = _next_significant_index(text, i + 1)
        if ch == "." and _looks_like_abbreviation(text, i):
            i += 1
            continue

        sentence = text[start : i + 1].strip()
        if sentence:
            sentences.append(sentence)
        start = next_index
        i = start

    return text[start:], sentences
