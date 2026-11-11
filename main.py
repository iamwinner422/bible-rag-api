import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from search_engine import SearchConfig, SearchEngine
from reference import parse_reference, BibleReference

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MAX_PAGE_SIZE = 50

log = logging.getLogger("uvicorn.error")

state = {"engine": None, "meta": None, "error": None}

def load_index():
    try:
        log.info("Loading search index...")
        emb = np.load(os.path.join(OUTPUTS, "bible_emb.npy")).astype("float32")
        with open(os.path.join(OUTPUTS, "bible_meta.json"), encoding="utf-8") as file:
            meta = json.load(file)
        state["engine"] = SearchEngine(emb, meta, config=SearchConfig())
        state["meta"] = meta
        log.info("Search index ready: %d passages", len(meta))
    except Exception as exc:
        state["error"] = repr(exc)
        log.exception("Failed to load search index")

@asynccontextmanager
async def lifespan(app):
    load_index()
    yield

app = FastAPI(
    title="AI Bible Search API",
    description="Hybrid (semantic + keyword) search in the Louis Segond 1910 French Bible.",
    version="2.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

class VerseResult(BaseModel):
    reference: str
    text: str
    score: float
    semantic_score: Optional[float] = None
    mots_cles: bool = False

class PaginatedResponse(BaseModel):
    query: str
    mode: str                          # "semantic" or "reference"
    page: int
    pageSize: int
    total: int
    totalPages: int
    results: list[VerseResult]

def not_ready():
    if state["error"]:
        return JSONResponse({"status": "error", "detail": state["error"]}, status_code=500)
    return JSONResponse(
        {"status": "loading", "detail": "The index is loading; please try again in a few seconds."},
        status_code=503,
        headers={"Retry-After": "10"},
    )

def get_reference_verses(ref: BibleReference, meta: list) -> list[dict]:
    """Return the verses corresponding to a reference, in order."""
    results = []
    for p in meta:
        if p["book"] != ref.book:
            continue
        if p["chapter"] != ref.chapter:
            continue
        # On ne prend que les versets isolés (pas les passages groupés)
        if p.get("start") != p.get("end"):
            continue

        verse_num = p["start"]
        if ref.verse_start is not None:
            if verse_num < ref.verse_start:
                continue
            if ref.verse_end is not None and verse_num > ref.verse_end:
                continue

        results.append(p)

    # Tri par numéro de verset
    results.sort(key=lambda x: x["start"])
    return results

@app.get("/")
def health():
    if state["error"]:
        return {"status": "error", "detail": state["error"]}
    if state["engine"] is None:
        return {"status": "loading"}
    return {"status": "ok", "passages": len(state["meta"])}

@app.get("/search", response_model=PaginatedResponse)
def search(
    query: str = Query(..., min_length=2, max_length=300),
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    pageSize: int = Query(20, ge=1, le=MAX_PAGE_SIZE, description="Number of results per page"),
):
    engine, meta = state["engine"], state["meta"]
    if engine is None:
        return not_ready()

    query = query.strip()
    ref = parse_reference(query)

    # ---------- Reference Mode ----------
    if ref is not None:
        all_verses = get_reference_verses(ref, meta)
        total = len(all_verses)
        total_pages = max(1, (total + pageSize - 1) // pageSize)

        start = (page - 1) * pageSize
        end = start + pageSize
        page_verses = all_verses[start:end]

        results = []
        for p in page_verses:
            span = str(p["start"]) if p["start"] == p["end"] else f"{p['start']}-{p['end']}"
            results.append(VerseResult(
                reference=f"{p['book']} {p['chapter']}:{span} (LSG)",
                text=p["text"],
                score=1.0,
                semantic_score=None,
                mots_cles=False,
            ))

        return PaginatedResponse(
            query=query,
            mode="reference",
            page=page,
            pageSize=pageSize,
            total=total,
            totalPages=total_pages,
            results=results,
        )

    # ---------- Semantic Mode (existing) ----------
    hits = engine.search(query, size=200)  # we take a large number to be able to paginate
    total = len(hits)
    total_pages = max(1, (total + pageSize - 1) // pageSize)

    start = (page - 1) * pageSize
    end = start + pageSize
    page_hits = hits[start:end]

    results = []
    for hit in page_hits:
        p = meta[hit.idx]
        span = str(p["start"]) if p["start"] == p["end"] else f"{p['start']}-{p['end']}"
        results.append(VerseResult(
            reference=f"{p['book']} {p['chapter']}:{span} (LSG)",
            text=p["text"],
            score=round(hit.score, 3),
            semantic_score=round(hit.semantic, 3),
            mots_cles=bool(hit.keyword > 0),
        ))

    return PaginatedResponse(
        query=query,
        mode="semantic",
        page=page,
        pageSize=pageSize,
        total=total,
        totalPages=total_pages,
        results=results,
    )