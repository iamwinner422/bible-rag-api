import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import get_reference_verses  # noqa: E402
from reference import parse_reference  # noqa: E402


class ParseReferenceTests(unittest.TestCase):
    def test_existing_single_verse_and_range(self):
        self.assertEqual(parse_reference("Jean 3:16").verse_start, 16)
        self.assertEqual(parse_reference("Jean 3:16-18").verse_end, 18)

    def test_parses_comma_separated_verses_and_ranges(self):
        reference = parse_reference("Jean 3:16, 18-20")

        self.assertIsNotNone(reference)
        self.assertEqual(reference.verse_start, 16)
        self.assertEqual(reference.verse_end, 16)
        self.assertEqual(reference.verse_ranges, [(16, 16), (18, 20)])

    def test_parses_chapter_ranges(self):
        reference = parse_reference("Jean 3-4")

        self.assertIsNotNone(reference)
        self.assertEqual(reference.chapter, 3)
        self.assertEqual(reference.chapter_end, 4)
        self.assertIsNone(parse_reference("Jean 4-3"))
        self.assertIsNone(parse_reference("Jean 3-4:16"))

    def test_ignores_trailing_sentence_punctuation(self):
        reference = parse_reference("Jean 3:16-18.")

        self.assertIsNotNone(reference)
        self.assertEqual(reference.verse_start, 16)
        self.assertEqual(reference.verse_end, 18)

    def test_reference_lookup_selects_only_requested_verses(self):
        reference = parse_reference("Jean 3:16,18-20")
        metadata = [
            {"book": "Jean", "chapter": 3, "start": verse, "end": verse}
            for verse in (15, 16, 17, 18, 19, 20, 21)
        ]

        self.assertEqual(
            [verse["start"] for verse in get_reference_verses(reference, metadata)],
            [16, 18, 19, 20],
        )

    def test_chapter_range_lookup_includes_all_verses_in_range(self):
        reference = parse_reference("Jean 3-4.")
        metadata = [
            {"book": "Jean", "chapter": chapter, "start": verse, "end": verse}
            for chapter, verse in ((4, 1), (3, 2), (2, 1), (3, 1), (5, 1))
        ]

        self.assertEqual(
            [
                (verse["chapter"], verse["start"])
                for verse in get_reference_verses(reference, metadata)
            ],
            [(3, 1), (3, 2), (4, 1)],
        )


if __name__ == "__main__":
    unittest.main()
