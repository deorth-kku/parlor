"""Voice catalog derived from kokoro-onnx VOICES.md."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VoiceCatalog:
    languages: dict[str, dict[str, object]]
    default_language: str
    default_voices: dict[str, str]


_SECTION_RE = re.compile(r"^###\s+(?P<title>.+?)\s*$")
_VOICE_RE = re.compile(r"^\|\s*(?P<cell>[^|]+?)\s*\|")


def _language_key(title: str) -> str | None:
    mapping = {
        "American English": "en",
        "British English": "en",
        "Japanese": "ja",
        "Mandarin Chinese": "zh",
    }
    return mapping.get(title)


def load_voice_catalog(voices_md_path: str | Path | None = None) -> VoiceCatalog:
    path = Path(voices_md_path) if voices_md_path else Path(__file__).resolve().parent.parent / "vender" / "kokoro-onnx" / "VOICES.md"
    sections: dict[str, list[str]] = {"en": [], "ja": [], "zh": []}
    current_language: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        section_match = _SECTION_RE.match(raw_line)
        if section_match:
            current_language = _language_key(section_match.group("title"))
            continue

        voice_match = _VOICE_RE.match(raw_line)
        if not voice_match or current_language not in sections:
            continue
        voice = voice_match.group("cell").replace("**", "").replace("\\_", "_").strip()
        if voice not in sections[current_language]:
            sections[current_language].append(voice)

    languages = {
        "en": {
            "label": "English",
            "voices": sections["en"],
            "default_voice": "af_heart",
        },
        "ja": {
            "label": "Japanese",
            "voices": sections["ja"],
            "default_voice": "jf_alpha",
        },
        "zh": {
            "label": "Chinese",
            "voices": sections["zh"],
            "default_voice": "zf_xiaoxiao",
        },
    }

    return VoiceCatalog(
        languages=languages,
        default_language="en",
        default_voices={k: v["default_voice"] for k, v in languages.items()},
    )


VOICE_CATALOG = load_voice_catalog()
