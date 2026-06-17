import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentence_splitter import append_sentence_buffer


class SentenceSplitterTests(unittest.TestCase):
    def test_keeps_country_abbreviation_inside_sentence(self):
        text = (
            "China is currently developing its own aircraft carriers, "
            "but they are generally smaller than the U.S. supercarriers. "
            "While they are expanding their naval capabilities, the U.S. "
            "currently maintains the largest and most advanced ships in the fleet."
        )
        buffer = ""
        sentences: list[str] = []
        for chunk in [
            "China is currently developing its own aircraft carriers, but they are ",
            "generally smaller than the U.S. supercarriers. While they are expanding ",
            "their naval capabilities, the U.S. currently maintains the largest and most advanced ships in the fleet.",
        ]:
            buffer, new_sentences = append_sentence_buffer(buffer, chunk)
            sentences.extend(new_sentences)

        self.assertEqual(buffer, "")
        self.assertEqual(
            sentences,
            [
                "China is currently developing its own aircraft carriers, but they are generally smaller than the U.S. supercarriers.",
                "While they are expanding their naval capabilities, the U.S. currently maintains the largest and most advanced ships in the fleet.",
            ],
        )

    def test_keeps_pattern_based_abbreviation_without_wordlist(self):
        buffer, sentences = append_sentence_buffer(
            "",
            "Several options exist, e.g. apples and oranges. Another sentence.",
        )
        self.assertEqual(
            sentences,
            [
                "Several options exist, e.g. apples and oranges.",
                "Another sentence.",
            ],
        )
        self.assertEqual(buffer, "")

    def test_keeps_decimal_numbers_inside_sentence(self):
        buffer, sentences = append_sentence_buffer("", "Version 3.14 is stable. Next sentence.")
        self.assertEqual(sentences, ["Version 3.14 is stable.", "Next sentence."])
        self.assertEqual(buffer, "")

    def test_waits_for_more_text_after_trailing_abbreviation(self):
        buffer, sentences = append_sentence_buffer("", "The U.S.")
        self.assertEqual(sentences, [])
        self.assertEqual(buffer, "The U.S.")

        buffer, sentences = append_sentence_buffer(buffer, " Navy is large.")
        self.assertEqual(sentences, ["The U.S. Navy is large."])
        self.assertEqual(buffer, "")

    def test_does_not_split_short_opening_comma_phrase(self):
        buffer, sentences = append_sentence_buffer("", "Yeah, that makes sense.")
        self.assertEqual(sentences, ["Yeah, that makes sense."])
        self.assertEqual(buffer, "")

    def test_splits_long_semicolon_chain_early(self):
        first_chunk = (
            "We can keep the current fallback for now; it is stable in production, "
            "easy to reason about, and already monitored; meanwhile we can test the "
        )
        buffer, sentences = append_sentence_buffer("", first_chunk)
        self.assertEqual(
            sentences,
            [
                "We can keep the current fallback for now; it is stable in production, easy to reason about, and already monitored;",
            ],
        )
        self.assertEqual(
            buffer,
            "meanwhile we can test the ",
        )

    def test_splits_long_comma_heavy_sentence_early(self):
        first_chunk = (
            "The fix is straightforward, we update the parser, we add coverage for "
        )
        buffer, sentences = append_sentence_buffer("", first_chunk)
        self.assertEqual(
            sentences,
            [
                "The fix is straightforward, we update the parser,",
            ],
        )
        self.assertEqual(
            buffer,
            "we add coverage for ",
        )


if __name__ == "__main__":
    unittest.main()
