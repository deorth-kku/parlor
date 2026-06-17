import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai_backend import (
    _cleanup_model_output,
    _is_degenerate_model_output,
    _model_prefers_prompt_structured_output,
    _append_instruction_to_user_content,
    _parse_turn_output,
)


class OpenAIBackendParsingTests(unittest.TestCase):
    def test_parse_json_payload_with_wrappers(self):
        raw = '<|im_start|>assistant\n<think>\n\n</think>\n```json\n{"transcription":"hi","response":"hello"}\n```'
        parsed = _parse_turn_output(raw)
        self.assertEqual(parsed["transcription"], "hi")
        self.assertEqual(parsed["response"], "hello")

    def test_parse_plain_text_fallback(self):
        parsed = _parse_turn_output("Just answer naturally.")
        self.assertEqual(parsed["transcription"], "")
        self.assertEqual(parsed["response"], "Just answer naturally.")

    def test_cleanup_removes_thinking_wrappers(self):
        cleaned = _cleanup_model_output('<|im_start|>assistant\n<think>hidden</think>\n{"response":"ok"}')
        self.assertEqual(cleaned, '{"response":"ok"}')

    def test_cleanup_drops_role_labels(self):
        self.assertEqual(_cleanup_model_output("user"), "")
        self.assertEqual(_cleanup_model_output("<|start_header_id|>assistant<|end_header_id|>\nhello"), "hello")

    def test_qwen_models_prefer_prompt_mode(self):
        self.assertTrue(_model_prefers_prompt_structured_output("Qwen3-Omni-30B-A3B-Instruct"))
        self.assertFalse(_model_prefers_prompt_structured_output("gemma-4-12b-it"))

    def test_degenerate_output_detection(self):
        self.assertTrue(_is_degenerate_model_output("<think>\n\n</think>\n\nuser\n"))
        self.assertFalse(_is_degenerate_model_output("Hello there."))

    def test_instruction_appends_to_multimodal_user_content(self):
        content = [{"type": "text", "text": "Hello"}]
        updated = _append_instruction_to_user_content(content, "Return JSON.")
        self.assertEqual(updated[-1], {"type": "text", "text": "Return JSON."})


if __name__ == "__main__":
    unittest.main()
