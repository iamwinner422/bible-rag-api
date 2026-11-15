import gzip
import json
from typing import Any

import numpy as np

from app.api.schemas import PaginatedResponse, VerseResult
from app.config import MAX_SEMANTIC_RESULTS, OUTPUTS_DIR
from app.references import get_reference_verses, parse_reference
from app.search.engine import Hit, SearchConfig, SearchEngine


class SearchService:
    def __init__(self, engine: SearchEngine, meta: list[dict[str, Any]]) -> None:
        self.engine = engine
        self.meta = meta
        self.verse_count = sum(
            1 for passage in meta if passage.get("start") == passage.get("end")
        )

    def search(self, query: str, page: int, page_size: int) -> PaginatedResponse:
        query = query.strip()
        reference = parse_reference(query)

        if reference is not None:
            passages = get_reference_verses(reference, self.meta)
            mode = "reference"
            results = [self._reference_result(passage) for passage in passages]
        else:
            hits = self.engine.search(query, size=MAX_SEMANTIC_RESULTS)
            hits = hits[:MAX_SEMANTIC_RESULTS]
            mode = "semantic"
            results = [self._semantic_result(hit) for hit in hits]

        total = len(results)
        total_pages = max(1, (total + page_size - 1) // page_size)
        start = (page - 1) * page_size
        return PaginatedResponse(
            query=query,
            mode=mode,
            page=page,
            pageSize=page_size,
            total=total,
            totalPages=total_pages,
            results=results[start : start + page_size],
        )

    def _reference_result(self, passage: dict[str, Any]) -> VerseResult:
        return VerseResult(
            reference=self._format_reference(passage),
            text=passage["text"],
            score=1.0,
            semantic_score=None,
            mots_cles=False,
        )

    def _semantic_result(self, hit: Hit) -> VerseResult:
        passage = self.meta[hit.idx]
        return VerseResult(
            reference=self._format_reference(passage),
            text=passage["text"],
            score=round(hit.score, 3),
            semantic_score=round(hit.semantic, 3),
            mots_cles=bool(hit.keyword > 0),
        )

    @staticmethod
    def _format_reference(passage: dict[str, Any]) -> str:
        start = passage["start"]
        end = passage["end"]
        span = str(start) if start == end else f"{start}-{end}"
        return f"{passage['book']} {passage['chapter']}:{span} (LSG)"


def load_search_service() -> SearchService:
    with gzip.open(OUTPUTS_DIR / "bible_emb.npy.gz", "rb") as file:
        embeddings = np.load(file).astype("float32")
    with gzip.open(
        OUTPUTS_DIR / "bible_meta.json.gz",
        "rt",
        encoding="utf-8",
    ) as file:
        meta = json.load(file)
    engine = SearchEngine(embeddings, meta, config=SearchConfig())
    return SearchService(engine, meta)
