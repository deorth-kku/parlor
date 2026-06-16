"""Helpers for per-turn TTS language instructions."""

from __future__ import annotations

from voices_catalog import VOICE_CATALOG


def build_language_instruction(language: str, voice: str) -> str:
    label = str(VOICE_CATALOG.languages[language]["label"])
    return (
        f"TTS language selected by user: {language} ({label}). "
        f"TTS voice selected by user: {voice}. "
        f"Respond in {label} unless the user explicitly asks for a different language."
    )
