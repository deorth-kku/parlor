import unittest
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voices_catalog import load_voice_catalog
from tts import ONNXBackend, _build_provider_config, _memory_arena_shrink_target, _provider_name_from_env
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


class RuntimeConfigTests(unittest.TestCase):
    def test_provider_defaults_to_cpu_without_env(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_provider_name_from_env(), "CPUExecutionProvider")

    def test_cuda_provider_options_default_to_same_as_requested(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            providers = _build_provider_config("CUDAExecutionProvider")
        self.assertEqual(providers, [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})])

    def test_cuda_provider_options_include_optional_env_overrides(self):
        with mock.patch.dict(
            "os.environ",
            {
                "ORT_CUDA_ARENA_EXTEND_STRATEGY": "kNextPowerOfTwo",
                "ORT_CUDA_GPU_MEM_LIMIT": "2147483648",
                "ORT_CUDA_USE_EP_LEVEL_UNIFIED_STREAM": "1",
            },
            clear=True,
        ):
            providers = _build_provider_config("CUDAExecutionProvider")
        self.assertEqual(
            providers,
            [
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": "2147483648",
                        "use_ep_level_unified_stream": "1",
                    },
                )
            ],
        )

    def test_memory_arena_shrink_target_matches_provider(self):
        self.assertEqual(_memory_arena_shrink_target("CPUExecutionProvider"), "cpu:0")
        self.assertEqual(_memory_arena_shrink_target("CUDAExecutionProvider"), "gpu:0")


if __name__ == "__main__":
    unittest.main()
