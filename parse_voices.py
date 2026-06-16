
import re
from pathlib import Path

def _language_key(title: str) -> str | None:
    mapping = {
        "American English": "en",
        "British English": "en",
        "Japanese": "ja",
        "Mandarin Chinese": "zh",
    }
    return mapping.get(title)

_SECTION_RE = re.compile(r"^###\s+(?P<title>.+?)\s*$")
_VOICE_RE = re.compile(r"^\|\s*(?P<cell>[^|]+?)\s*\|")

def parse_voices():
    path = Path("vender/kokoro-onnx/VOICES.md")
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
        if voice not in sections[current_language] and voice not in ["Name", "----"]:
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
    return languages

if __name__ == "__main__":
    data = parse_voices()
    with open("src/voices_data.py", "w", encoding="utf-8") as f:
        f.write("LANGUAGES = ")
        f.write(str(data))
