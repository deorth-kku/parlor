"""Tests for the remote TTS client (tts_remote.py)."""

from __future__ import annotations

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from config import Config
from tts_remote import RemoteTTS


def _make_config(
    voices,
    languages=None,
    default_language="en",
    default_voice="",
):
    data = {
        "tts": {
            "base_url": "http://127.0.0.1:9931",
            "voices": voices,
            "languages": languages
            or [
                {"code": "en", "label": "English"},
                {"code": "zh", "label": "Chinese"},
            ],
            "default_language": default_language,
            "default_voice": default_voice,
        }
    }
    return Config(data)


class RemoteTTSTestsBase(unittest.TestCase):
    """Base class that closes the httpx client created by each test."""

    def setUp(self):
        self._tts = None

    def tearDown(self):
        if self._tts is not None:
            asyncio.run(self._tts.aclose())

    def make_tts(self, config):
        self._tts = RemoteTTS(config)
        return self._tts


class RemoteTTSModeTests(RemoteTTSTestsBase):
    def test_dict_mode_is_ref(self):
        tts = self.make_tts(_make_config({"alice": "/tmp/x.wav"}))
        self.assertEqual(tts.voice_mode(), "ref")
        self.assertEqual(tts.voice_names(), ["alice"])

    def test_list_mode_is_name(self):
        tts = self.make_tts(_make_config(["female", "male"]))
        self.assertEqual(tts.voice_mode(), "name")
        self.assertEqual(tts.voice_names(), ["female", "male"])

    def test_empty_voices(self):
        tts = self.make_tts(_make_config({}))
        self.assertEqual(tts.voice_names(), [])
        self.assertEqual(tts.default_voice, "")


class RemoteTTSRefTests(RemoteTTSTestsBase):
    def test_speaker_ref_b64(self):
        payload = b"synthetic-audio-bytes"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(payload)
            path = fh.name
        try:
            tts = self.make_tts(_make_config({"alice": path}))
            b64 = tts._speaker_ref_b64("alice")
            self.assertEqual(base64.b64decode(b64), payload)
            # Cached on the second call.
            self.assertEqual(tts._speaker_ref_b64("alice"), b64)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_validate_missing_ref_raises(self):
        tts = self.make_tts(_make_config({"alice": "/tmp/does-not-exist.wav"}))
        with self.assertRaises(FileNotFoundError):
            tts.validate()


class RemoteTTSLanguageTests(RemoteTTSTestsBase):
    def test_languages_and_defaults(self):
        tts = self.make_tts(
            _make_config({"alice": "/tmp/x.wav"}, default_language="zh", default_voice="alice")
        )
        self.assertEqual(tts.languages()[0]["code"], "en")
        self.assertEqual(tts.default_language, "zh")
        self.assertEqual(tts.default_voice, "alice")

    def test_default_falls_back_to_first(self):
        tts = self.make_tts(
            _make_config({"alice": "/tmp/x.wav"}, default_language="fr", default_voice="nope")
        )
        self.assertEqual(tts.default_language, "en")
        self.assertEqual(tts.default_voice, "alice")


class ParseSampleRateTests(unittest.TestCase):
    def test_parses_rate(self):
        self.assertEqual(
            RemoteTTS._parse_sample_rate("audio/pcm;rate=24000;encoding=float;bits=32"), 24000
        )

    def test_fallback(self):
        self.assertEqual(RemoteTTS._parse_sample_rate("audio/pcm"), 24000)


if __name__ == "__main__":
    unittest.main()
