import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402


class FakeEngine:
    def __init__(self, hits):
        self.hits = hits
        self.requested_sizes = []

    def search(self, query, size):
        self.requested_sizes.append(size)
        return self.hits


def make_verses(count, book="Jean", chapter=3):
    return [
        {
            "book": book,
            "chapter": chapter,
            "start": verse,
            "end": verse,
            "text": f"Verset {verse}",
        }
        for verse in range(1, count + 1)
    ]


class SearchPaginationTests(unittest.TestCase):
    def test_semantic_results_are_capped_across_pages(self):
        meta = make_verses(25)
        hits = [
            SimpleNamespace(idx=i, score=0.9, semantic=0.8, keyword=0.0)
            for i in range(len(meta))
        ]
        engine = FakeEngine(hits)

        with patch.dict(main.state, {"engine": engine, "meta": meta}):
            page_one = main.search("la foi", page=1, pageSize=10)
            page_two = main.search("la foi", page=2, pageSize=10)

        self.assertEqual(engine.requested_sizes, [16, 16])
        self.assertEqual(page_one.total, 16)
        self.assertEqual(page_one.totalPages, 2)
        self.assertEqual(len(page_one.results), 10)
        self.assertEqual(page_two.total, 16)
        self.assertEqual(len(page_two.results), 8)

    def test_reference_search_is_not_capped_at_16_verses(self):
        meta = make_verses(20)
        engine = FakeEngine([])

        with patch.dict(main.state, {"engine": engine, "meta": meta}):
            response = main.search("Jean 3", page=1, pageSize=20)

        self.assertEqual(response.mode, "reference")
        self.assertEqual(response.total, 20)
        self.assertEqual(len(response.results), 20)
        self.assertEqual(engine.requested_sizes, [])


if __name__ == "__main__":
    unittest.main()
