from types import SimpleNamespace

from app.services.search_service import SearchService


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


def test_semantic_results_are_capped_across_pages():
    meta = make_verses(25)
    hits = [
        SimpleNamespace(idx=i, score=0.9, semantic=0.8, keyword=0.0)
        for i in range(len(meta))
    ]
    engine = FakeEngine(hits)
    service = SearchService(engine, meta)

    page_one = service.search("la foi", page=1, page_size=10)
    page_two = service.search("la foi", page=2, page_size=10)

    assert engine.requested_sizes == [16, 16]
    assert page_one.total == 16
    assert page_one.totalPages == 2
    assert len(page_one.results) == 10
    assert page_two.total == 16
    assert len(page_two.results) == 6


def test_reference_search_is_not_capped_at_16_verses():
    meta = make_verses(20)
    engine = FakeEngine([])
    service = SearchService(engine, meta)

    response = service.search("Jean 3", page=1, page_size=20)

    assert response.mode == "reference"
    assert response.total == 20
    assert len(response.results) == 20
    assert engine.requested_sizes == []
