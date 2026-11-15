import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.search import load_search_service

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    log.info("Loading search index...")
    application.state.search_service = None
    application.state.startup_error = None
    try:
        service = load_search_service()
    except Exception as exc:
        application.state.startup_error = repr(exc)
        log.exception("Failed to load search index")
    else:
        application.state.search_service = service
        log.info("Search index ready: %d passages", len(service.meta))
    yield


app = FastAPI(
    title="AI Bible Search API",
    description="Hybrid (semantic + keyword) search in the Louis Segond 1910 French Bible.",
    version="2.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
)
app.include_router(router)
