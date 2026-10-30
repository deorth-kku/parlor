"""Remote TTS client for the llama.cpp server's ``POST /tts`` endpoint.

Speech is generated on a remote llama.cpp server and streamed back as raw
float32 little-endian mono PCM (``response_format=pcm`` + ``stream=true``).
The sample rate is carried in the response ``Content-Type`` header, e.g.
``audio/pcm;rate=24000;encoding=float;bits=32``.

Voice selection is driven by the ``tts.voices`` config value:

* a **dict** ``{name: ref_audio_path}`` -> "ref" mode: the referenced audio
  file is base64-encoded and sent as ``speaker_ref_b64`` (no ``voice`` field).
* a **list** ``[name, ...]`` -> "name" mode: the selected name is sent in the
  ``voice`` field and no reference file is used.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from config import DEFAULT_CONFIG_PATH, Config

# Matches the sample rate in a PCM Content-Type header, e.g. "...;rate=24000;...".
_RATE_RE = re.compile(r"rate=(\d+)")

# Fallback sample rate when the header does not carry one.
_FALLBACK_SAMPLE_RATE = 24000


class RemoteTTS:
    """Thin async client around the llama.cpp ``/tts`` endpoint."""

    def __init__(self, config: Config):
        self._config = config
        self._base_url = str(config.get("tts.base_url", "http://127.0.0.1:9931")).rstrip("/")
        self._model = str(config.get("tts.model") or "")
        self._timeout = float(config.get("tts.timeout", 120.0))
        params = config.get("tts.params")
        self._params: dict[str, Any] = params if isinstance(params, dict) else {}
        self._voices: Any = config.get("tts.voices")
        # Relative reference-audio paths resolve against the config directory.
        self._ref_dir = DEFAULT_CONFIG_PATH.parent
        self._ref_cache: dict[str, str] = {}
        self._client = httpx.AsyncClient(timeout=self._timeout)

    def freeze_config(self) -> None:
        """Snapshot the current config data.

        After a hot reload replaces the shared ``Config`` singleton's data in
        place, retired clients keep reading their pre-reload values (e.g.
        languages and defaults) instead of mixing new and old config.
        """
        self._config = Config(self._config.data)

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    def voice_mode(self) -> str:
        """``"ref"`` when ``voices`` is a dict, ``"name"`` when it is a list."""
        return "ref" if isinstance(self._voices, dict) else "name"

    def voice_names(self) -> list[str]:
        """The selectable voice names, in config order."""
        if isinstance(self._voices, dict):
            return list(self._voices.keys())
        if isinstance(self._voices, list):
            return [str(v) for v in self._voices]
        return []

    def validate(self) -> None:
        """Fail fast if configured reference audio files are missing."""
        if self.voice_mode() != "ref":
            return
        for name, ref in self._voices.items():
            path = Path(ref)
            if not path.is_absolute():
                path = self._ref_dir / path
            if not path.is_file():
                raise FileNotFoundError(
                    f"TTS reference audio for voice {name!r} not found: {path}"
                )

    def languages(self) -> list[dict[str, str]]:
        """The available languages as ``[{"code", "label"}, ...]``."""
        raw = self._config.get("tts.languages") or []
        result: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                code = str(item.get("code", ""))
                if code:
                    result.append({"code": code, "label": str(item.get("label", code))})
            else:
                code = str(item)
                if code:
                    result.append({"code": code, "label": code})
        return result

    @property
    def default_language(self) -> str:
        langs = self.languages()
        configured = self._config.get("tts.default_language")
        if configured and any(l["code"] == configured for l in langs):
            return str(configured)
        return langs[0]["code"] if langs else "en"

    @property
    def default_voice(self) -> str:
        names = self.voice_names()
        configured = self._config.get("tts.default_voice")
        if configured and configured in names:
            return str(configured)
        return names[0] if names else ""

    def _speaker_ref_b64(self, voice: str) -> str:
        """Base64-encode the reference audio file for ``voice`` (cached)."""
        cached = self._ref_cache.get(voice)
        if cached is not None:
            return cached
        path = Path(self._voices[voice])
        if not path.is_absolute():
            path = self._ref_dir / path
        cached = base64.b64encode(path.read_bytes()).decode("ascii")
        self._ref_cache[voice] = cached
        return cached

    @staticmethod
    def _parse_sample_rate(content_type: str) -> int:
        match = _RATE_RE.search(content_type)
        if match:
            return int(match.group(1))
        return _FALLBACK_SAMPLE_RATE

    async def stream_pcm(
        self, text: str, language: str, voice: str
    ) -> AsyncIterator[tuple[int | None, bytes]]:
        """Stream PCM for ``text`` from the remote server.

        Yields ``(sample_rate, chunk)`` where ``sample_rate`` is set on the
        first chunk and ``None`` afterwards. Every ``chunk`` is 4-byte aligned
        (a whole number of float32 samples) so the client can decode it safely.
        """
        if not voice:
            raise ValueError("No TTS voice configured (tts.voices is empty)")
        # Extra params go in first so the protocol fields always win.
        payload: dict[str, Any] = dict(self._params)
        payload.update(
            {
                "input": text,
                "lang": language,
                "response_format": "pcm",
                "stream": True,
            }
        )
        if self._model:
            payload["model"] = self._model
        if self.voice_mode() == "ref":
            payload["speaker_ref_b64"] = self._speaker_ref_b64(voice)
        else:
            payload["voice"] = voice

        async with self._client.stream("POST", f"{self._base_url}/tts", json=payload) as resp:
            resp.raise_for_status()
            sample_rate = self._parse_sample_rate(resp.headers.get("content-type", ""))
            carry = b""
            first = True
            async for chunk in resp.aiter_bytes():
                carry += chunk
                aligned_len = len(carry) - (len(carry) % 4)
                if aligned_len == 0:
                    continue
                to_send = carry[:aligned_len]
                carry = carry[aligned_len:]
                yield (sample_rate if first else None), to_send
                first = False
            if first:
                print(
                    "TTS warning: remote returned no audio "
                    f"(text={text[:100]!r}, language={language!r}, voice={voice!r})"
                )

    async def aclose(self) -> None:
        await self._client.aclose()
