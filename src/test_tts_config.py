import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voices_catalog import load_voice_catalog
from tts import ONNXBackend
from tts_prompt import build_language_instruction


class VoiceCatalogTests(unittest.TestCase):
    def test_only_expected_languages_are_exposed(self):
        catalog = load_voice_catalog()
        self.assertEqual(set(catalog.languages.keys()), {"en", "ja", "zh"})

    def test_english_contains_us_and_uk_voices(self):
        catalog = load_voice_catalog()
        voices = catalog.languages["en"]["voices"]
        self.assertIn("af_heart", voices)
        self.assertIn("bf_emma", voices)

    def test_japanese_and_chinese_match_expected_codes(self):
        catalog = load_voice_catalog()
        self.assertIn("jf_alpha", catalog.languages["ja"]["voices"])
        self.assertIn("zf_xiaoxiao", catalog.languages["zh"]["voices"])


class RoutingTests(unittest.TestCase):
    def test_english_voice_prefix_routing(self):
        self.assertEqual(ONNXBackend._en_lang_for_voice("af_heart"), "en-us")
        self.assertEqual(ONNXBackend._en_lang_for_voice("bf_emma"), "en-gb")

    def test_voice_validation_rejects_mismatch(self):
        with self.assertRaises(ValueError):
            ONNXBackend._validate_voice("ja", "af_heart")

    def test_language_instruction_contains_selection(self):
        instruction = build_language_instruction("zh", "zf_xiaoxiao")
        self.assertIn("TTS language selected by user: zh", instruction)
        self.assertIn("TTS voice selected by user: zf_xiaoxiao", instruction)


if __name__ == "__main__":
    unittest.main()
