from typing import Literal

from pydantic import BaseModel


class VerseResult(BaseModel):
    reference: str
    text: str
    score: float
    semantic_score: float | None = None
    mots_cles: bool = False


class PaginatedResponse(BaseModel):
    query: str
    mode: Literal["semantic", "reference"]
    page: int
    pageSize: int
    total: int
    totalPages: int
    results: list[VerseResult]
