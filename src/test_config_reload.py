"""Tests for the online config reload (ConfigManager)."""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from config_reload import ConfigManager, ReloadResult
from openai_backend import OpenAICompatibleBackend
from tts_remote import RemoteTTS


def make_backend(data: dict) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend.from_config(Config(data))


def make_tts(data: dict) -> RemoteTTS:
    return RemoteTTS(Config(data))


OLD_OPENAI = {"openai": {"base_url": "http://old.example.test/v1", "api_key": "test-key"}}
OLD_TTS = {"tts": {"base_url": "http://old.example.test:9931", "voices": ["alice"]}}
NEW_OPENAI = {"openai": {"base_url": "http://new.example.test/v1", "api_key": "test-key-2"}}
NEW_TTS = {"tts": {"base_url": "http://new.example.test:9931", "voices": ["bob"]}}


class ConfigManagerTests(unittest.TestCase):
    def _write_config(self, tmp: str, data: object) -> Path:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _make_manager(
        self, path: Path, grace_seconds: float = 30.0
    ) -> tuple[ConfigManager, OpenAICompatibleBackend, RemoteTTS]:
        backend = make_backend(OLD_OPENAI)
        tts = make_tts(OLD_TTS)
        manager = ConfigManager(path, backend=backend, tts=tts, grace_seconds=grace_seconds)
        return manager, backend, tts

    def test_apply_swaps_clients_on_valid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(tmp, {**NEW_OPENAI, **NEW_TTS})
            manager, old_backend, old_tts = self._make_manager(path)
            try:
                result = asyncio.run(manager.apply())
                self.assertIsInstance(result, ReloadResult)
                self.assertTrue(result.ok)
                self.assertIsNotNone(result.config)
                self.assertIsNot(manager.backend, old_backend)
                self.assertIsNot(manager.tts, old_tts)
                self.assertEqual(manager.backend._config.base_url, "http://new.example.test/v1")
                self.assertEqual(manager.tts.base_url, "http://new.example.test:9931")
                self.assertEqual(manager.tts.voice_names(), ["bob"])
                # Old clients are retired (deferred close), not closed yet.
                self.assertEqual(len(manager._retired), 2)
                self.assertFalse(old_backend._client.is_closed)
                self.assertFalse(old_tts._client.is_closed)
            finally:
                asyncio.run(old_backend.aclose())
                asyncio.run(old_tts.aclose())
                if manager.backend is not None:
                    asyncio.run(manager.backend.aclose())
                if manager.tts is not None:
                    asyncio.run(manager.tts.aclose())

    def test_apply_keeps_previous_on_missing_tts_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Ref mode with a reference file that does not exist.
            path = self._write_config(
                tmp, {"tts": {"voices": {"alice": "missing_ref_audio.wav"}}}
            )
            manager, old_backend, old_tts = self._make_manager(path)
            try:
                result = asyncio.run(manager.apply())
                self.assertFalse(result.ok)
                self.assertIn("keeping previous", result.message)
                self.assertIs(manager.backend, old_backend)
                self.assertIs(manager.tts, old_tts)
            finally:
                asyncio.run(old_backend.aclose())
                asyncio.run(old_tts.aclose())

    def test_apply_keeps_previous_on_bad_numeric_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A non-numeric timeout makes backend construction fail; the
            # previous clients must be kept and no exception may escape.
            path = self._write_config(tmp, {"openai": {"timeout": "fast"}})
            manager, old_backend, old_tts = self._make_manager(path)
            try:
                result = asyncio.run(manager.apply())
                self.assertFalse(result.ok)
                self.assertIn("keeping previous", result.message)
                self.assertIs(manager.backend, old_backend)
                self.assertIs(manager.tts, old_tts)
            finally:
                asyncio.run(old_backend.aclose())
                asyncio.run(old_tts.aclose())

    def test_apply_keeps_previous_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            manager, old_backend, old_tts = self._make_manager(path)
            try:
                result = asyncio.run(manager.apply())
                self.assertFalse(result.ok)
                self.assertIn("Failed to load config", result.message)
                self.assertIs(manager.backend, old_backend)
                self.assertIs(manager.tts, old_tts)
            finally:
                asyncio.run(old_backend.aclose())
                asyncio.run(old_tts.aclose())

    def test_apply_with_missing_file_uses_defaults(self):
        # A missing config file is not an error: built-in defaults apply.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "does-not-exist.json"
            manager, old_backend, old_tts = self._make_manager(path)
            try:
                result = asyncio.run(manager.apply())
                self.assertTrue(result.ok)
                self.assertIsNot(manager.backend, old_backend)
                self.assertEqual(manager.backend._config.base_url, "http://127.0.0.1:8080/v1")
                self.assertEqual(manager.tts.base_url, "http://127.0.0.1:9931")
            finally:
                asyncio.run(old_backend.aclose())
                asyncio.run(old_tts.aclose())
                if manager.backend is not None:
                    asyncio.run(manager.backend.aclose())
                if manager.tts is not None:
                    asyncio.run(manager.tts.aclose())

    def test_retired_tts_keeps_old_config_after_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data = {
                **OLD_OPENAI,
                "tts": {
                    "base_url": "http://old.example.test:9931",
                    "voices": ["alice"],
                    "languages": [{"code": "en", "label": "English"}],
                },
            }
            new_data = {
                **NEW_OPENAI,
                "tts": {
                    "base_url": "http://new.example.test:9931",
                    "voices": ["bob"],
                    "languages": [{"code": "ja", "label": "Japanese"}],
                },
            }
            path = self._write_config(tmp, new_data)
            # The old TTS shares one Config object, as it does with the
            # server's module-level singleton.
            shared = Config(old_data)
            backend = make_backend(old_data)
            tts = RemoteTTS(shared)
            manager = ConfigManager(path, backend=backend, tts=tts)
            try:
                result = asyncio.run(manager.apply())
                self.assertTrue(result.ok)
                # Simulate the server replacing the singleton's data in place.
                shared.replace(new_data)
                # The retired client must still read its pre-reload config.
                self.assertEqual([l["code"] for l in tts.languages()], ["en"])
                self.assertEqual(tts.default_language, "en")
            finally:
                asyncio.run(backend.aclose())
                asyncio.run(tts.aclose())
                if manager.backend is not None:
                    asyncio.run(manager.backend.aclose())
                if manager.tts is not None:
                    asyncio.run(manager.tts.aclose())

    def test_retire_loop_closes_old_clients_after_grace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(tmp, {**NEW_OPENAI, **NEW_TTS})
            manager, old_backend, old_tts = self._make_manager(path, grace_seconds=0.1)
            try:
                asyncio.run(self._run_retire_cycle(manager, old_backend, old_tts))
            finally:
                if manager.backend is not None:
                    asyncio.run(manager.backend.aclose())
                if manager.tts is not None:
                    asyncio.run(manager.tts.aclose())

    async def _run_retire_cycle(
        self,
        manager: ConfigManager,
        old_backend: OpenAICompatibleBackend,
        old_tts: RemoteTTS,
    ) -> None:
        await manager.start()
        try:
            result = await manager.apply()
            self.assertTrue(result.ok)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if old_backend._client.is_closed and old_tts._client.is_closed:
                    break
                await asyncio.sleep(0.05)
            self.assertTrue(old_backend._client.is_closed)
            self.assertTrue(old_tts._client.is_closed)
        finally:
            await manager.stop()
            self.assertIsNone(manager._retire_task)

    def test_stop_without_start_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            manager, old_backend, old_tts = self._make_manager(path)
            try:
                asyncio.run(manager.stop())
            finally:
                asyncio.run(old_backend.aclose())
                asyncio.run(old_tts.aclose())


if __name__ == "__main__":
    unittest.main()
