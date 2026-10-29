import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as config_module
from config import Config, _deep_merge, load_config, reload
from openai_backend import OpenAICompatibleBackend


class DeepMergeTests(unittest.TestCase):
    def test_override_merges_nested_sections(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"c": 3, "d": 4}}
        self.assertEqual(_deep_merge(base, override), {"a": 1, "b": {"c": 3, "d": 4}})

    def test_base_is_not_mutated(self):
        base = {"b": {"c": 2}}
        _deep_merge(base, {"b": {"c": 3}})
        self.assertEqual(base, {"b": {"c": 2}})


class ConfigAccessTests(unittest.TestCase):
    def test_get_dotted_path(self):
        cfg = Config({"openai": {"base_url": "http://x"}})
        self.assertEqual(cfg.get("openai.base_url"), "http://x")
        self.assertIsNone(cfg.get("openai.missing"))
        self.assertEqual(cfg.get("openai.missing", "dflt"), "dflt")

    def test_section_returns_dict_or_empty(self):
        cfg = Config({"server": {"port": 9000}})
        self.assertEqual(cfg.section("server"), {"port": 9000})
        self.assertEqual(cfg.section("nope"), {})


class LoadConfigTests(unittest.TestCase):
    def test_missing_file_uses_defaults(self):
        cfg = load_config(Path("/nonexistent/config.json"))
        self.assertEqual(cfg.get("server.port"), 8000)
        self.assertEqual(cfg.get("openai.base_url"), "http://127.0.0.1:8080/v1")

    def test_file_overrides_defaults_and_keeps_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"openai": {"base_url": "http://custom", "timeout": 5}}),
                encoding="utf-8",
            )
            cfg = load_config(path)
            self.assertEqual(cfg.get("openai.base_url"), "http://custom")
            self.assertEqual(cfg.get("openai.timeout"), 5)
            # Unspecified keys fall back to built-in defaults.
            self.assertEqual(cfg.get("openai.max_tokens"), 4096)

    def test_non_object_root_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


class ReloadTests(unittest.TestCase):
    def test_reload_updates_singleton_in_place(self):
        original = config_module.config
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps({"server": {"port": 9999}}), encoding="utf-8")
                returned = reload(path)
            self.assertIs(returned, original)
            self.assertEqual(original.get("server.port"), 9999)
        finally:
            # Restore the default singleton so other tests are unaffected.
            reload(None)


class BackendFromConfigTests(unittest.TestCase):
    def test_from_config_reads_openai_section(self):
        cfg = Config(
            {
                "openai": {
                    "base_url": "http://example.test/v1/",
                    "api_key": "  secret  ",
                    "model": "  some-model  ",
                    "timeout": 30,
                    "temperature": 0.5,
                    "max_tokens": 100,
                }
            }
        )
        backend = OpenAICompatibleBackend.from_config(cfg)
        self.assertEqual(backend._config.base_url, "http://example.test/v1/")
        self.assertEqual(backend._config.api_key, "secret")
        self.assertEqual(backend._config.model, "some-model")
        self.assertEqual(backend._config.timeout, 30.0)
        self.assertEqual(backend._config.temperature, 0.5)
        self.assertEqual(backend._config.max_tokens, 100)

    def test_from_config_empty_values_become_none(self):
        cfg = Config({"openai": {"api_key": "", "model": "", "temperature": 0}})
        backend = OpenAICompatibleBackend.from_config(cfg)
        self.assertIsNone(backend._config.api_key)
        self.assertIsNone(backend._config.model)
        self.assertIsNone(backend._config.temperature)


if __name__ == "__main__":
    unittest.main()
