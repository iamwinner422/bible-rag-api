from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.api.schemas import PaginatedResponse
from app.config import MAX_PAGE_SIZE
from app.services.search import SearchService

router = APIRouter()


def _not_ready(request: Request) -> JSONResponse:
    error = request.app.state.startup_error
    if error:
        return JSONResponse(
            {"status": "error", "detail": error},
            status_code=500,
        )
    return JSONResponse(
        {
            "status": "loading",
            "detail": "The index is loading; please try again in a few seconds.",
        },
        status_code=503,
        headers={"Retry-After": "10"},
    )


@router.get("/")
@router.get("/health")
def health(request: Request) -> dict:
    service: SearchService | None = request.app.state.search_service
    if request.app.state.startup_error:
        return {"status": "error", "detail": request.app.state.startup_error}
    if service is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "verses": service.verse_count,
        "passages": len(service.meta),
    }


@router.get("/search", response_model=PaginatedResponse)
def search(
    request: Request,
    query: str = Query(..., min_length=2, max_length=300),
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    pageSize: int = Query(
        20,
        ge=1,
        le=MAX_PAGE_SIZE,
        description="Number of results per page",
    ),
):
    service: SearchService | None = request.app.state.search_service
    if service is None:
        return _not_ready(request)
    return service.search(query, page, pageSize)
