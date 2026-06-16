"""Voice catalog derived from kokoro-onnx VOICES.md."""

from __future__ import annotations

from dataclasses import dataclass
from voices_data import LANGUAGES

@dataclass(frozen=True, slots=True)
class VoiceCatalog:
    languages: dict[str, dict[str, object]]
    default_language: str
    default_voices: dict[str, str]

def load_voice_catalog(voices_md_path: str | Path | None = None) -> VoiceCatalog:
    languages = LANGUAGES
    return VoiceCatalog(
        languages=languages,
        default_language="en",
        default_voices={k: v["default_voice"] for k, v in languages.items()},
    )

VOICE_CATALOG = load_voice_catalog()
