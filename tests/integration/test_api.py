import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_index(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "verses": 31102,
        "passages": 89739,
    }
    assert client.get("/").json() == response.json()


def test_semantic_search_returns_ranked_paginated_results(client):
    response = client.get(
        "/search",
        params={"query": "quel est le plus grand commandement", "pageSize": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "semantic"
    assert body["page"] == 1
    assert body["pageSize"] == 5
    assert len(body["results"]) == 5
    assert body["results"][0]["reference"].startswith("Matthieu 22")
    assert set(body["results"][0]) == {
        "reference",
        "text",
        "score",
        "semantic_score",
        "mots_cles",
    }
    assert all(
        left["score"] >= right["score"]
        for left, right in zip(body["results"], body["results"][1:])
    )


def test_semantic_results_do_not_overlap(client):
    response = client.get("/search", params={"query": "la foi", "pageSize": 8})
    results = response.json()["results"]
    spans = []
    for result in results:
        location, _ = result["reference"].split(" (LSG)")
        book, chapter_and_verse = location.rsplit(" ", 1)
        chapter, verse_span = chapter_and_verse.split(":")
        verse_bounds = verse_span.split("-")
        start = int(verse_bounds[0])
        end = int(verse_bounds[-1])
        spans.append((book, int(chapter), start, end))

    assert not any(
        book_a == book_b
        and chapter_a == chapter_b
        and start_a <= end_b
        and start_b <= end_a
        for index, (book_a, chapter_a, start_a, end_a) in enumerate(spans)
        for book_b, chapter_b, start_b, end_b in spans[index + 1 :]
    )


def test_reference_search_returns_requested_verse(client):
    response = client.get("/search", params={"query": "Jean 3:16"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "reference"
    assert body["total"] == 1
    assert body["results"][0]["reference"] == "Jean 3:16 (LSG)"


def test_semantic_pagination_caps_results_across_pages(client):
    response = client.get(
        "/search",
        params={
            "query": "quel est le plus grand commandement",
            "pageSize": 10,
            "page": 2,
        },
    )
    body = response.json()

    assert body["total"] == 16
    assert body["totalPages"] == 2
    assert len(body["results"]) == 6


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": "Dieu est amour"},
        {"query": "a"},
        {"query": "x" * 301},
        {"query": "la foi", "pageSize": 0},
        {"query": "la foi", "pageSize": 999},
        {"query": "la foi", "page": 0},
    ],
)
def test_invalid_search_parameters_return_422(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_exact_phrase_search_returns_matching_verse(client):
    response = client.get(
        "/search",
        params={"query": '"l\'Éternel est mon berger"'},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["results"][0]["reference"] == "Psaumes 23:1 (LSG)"


@pytest.mark.parametrize(
    "query",
    ['"phrase introuvable xyzzy"'],
)
def test_unmatched_exact_phrase_returns_no_results(client, query):
    response = client.get("/search", params={"query": query})

    assert response.status_code == 200
    assert response.json()["results"] == []
