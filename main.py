import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from search_engine import SearchConfig, SearchEngine

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MAX_RESULTS = 20

log = logging.getLogger("uvicorn.error")

# On Vercel, a serverless instance may freeze while a background thread loads the index,
# so load it before accepting the first request.
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
    except Exception as exc:  # Report through / and /search instead of failing silently.
        state["error"] = repr(exc)
        log.exception("Failed to load search index")


@asynccontextmanager
async def lifespan(app):
    load_index()
    yield


app = FastAPI(
    title="AI Bible Search API",
    description="Hybrid (semantic + keyword) search in the Louis Segond 1910 French Bible.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


class VerseResult(BaseModel):
    reference: str       # e.g. "Matthieu 22:36 (LSG)"
    text: str
    score: float         # Final score (0-1): relative confidence, not a relevance percentage.
    semantic_score: float  # Raw cosine similarity to the query.
    mots_cles: bool      # Legacy API field: whether a query term appears in the verse.


def not_ready():
    if state["error"]:
        return JSONResponse({"status": "error", "detail": state["error"]}, status_code=500)
    return JSONResponse({"status": "loading", "detail": "The index is loading; please try again in a few seconds."},
                        status_code=503, headers={"Retry-After": "10"})


@app.get("/")
@app.get("/health")
def health():
    # Always return 200 so platform health checks pass while the index is loading.
    if state["error"]:
        return {"status": "error", "detail": state["error"]}
    if state["engine"] is None:
        return {"status": "loading"}
    return {"status": "ok", "passages": len(state["meta"])}


@app.get("/search", response_model=list[VerseResult])
def search(
    query: str | None = Query(None, min_length=2, max_length=300,
                              description='Question or keywords. Use quotes for an exact phrase search.'),
    responseSize: int = Query(8, description=f"Number of verses (1 to {MAX_RESULTS})"),
):
    if not query or not query.strip():
        return JSONResponse({"detail": "The `query` parameter is required."}, status_code=422)
    engine, meta = state["engine"], state["meta"]
    if engine is None:
        return not_ready()
    size = min(max(responseSize, 1), MAX_RESULTS)

    results = []
    for hit in engine.search(query, size=size):
        p = meta[hit.idx]
        span = str(p["start"]) if p["start"] == p["end"] else f"{p['start']}-{p['end']}"
        results.append(VerseResult(
            reference=f"{p['book']} {p['chapter']}:{span} (LSG)",
            text=p["text"],
            score=round(hit.score, 3),
            semantic_score=round(hit.semantic, 3),
            mots_cles=bool(hit.keyword > 0),
        ))
    return results
