"""Helpers for per-turn TTS language instructions."""

from __future__ import annotations


def build_language_instruction(language: str, voice: str) -> str:
    return (
        f"TTS language selected by user: {language}. "
        f"TTS voice selected by user: {voice}. "
        f"Respond in {language} unless the user explicitly asks for a different language."
    )
