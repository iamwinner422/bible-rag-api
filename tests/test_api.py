"""End-to-end API tests (real index and model, no network required).

Usage: python3 -I tests/test_api.py      (install `httpx` for TestClient)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'OK ' if cond else 'FAIL'}  {name}{('  -> ' + str(detail)) if not cond else ''}")
    if not cond:
        check.failed += 1


check.failed = 0

with TestClient(app) as client:  # The context manager triggers index loading (lifespan).
    r = client.get("/")
    check("GET / returns ok + passage count", r.status_code == 200 and r.json().get("status") == "ok"
          and r.json().get("passages") == 31102, r.json())
    check("GET /health returns the same response", client.get("/health").json() == r.json())

    r = client.get("/search", params={"query": "quel est le plus grand commandement", "responseSize": 5})
    j = r.json()
    check("/search: 200, list of 5 results", r.status_code == 200 and isinstance(j, list) and len(j) == 5, r.text[:200])
    check("/search: Matthew 22 at the top", j and j[0]["book"] == "Matthieu" and j[0]["chapter"] == 22, j[:1])
    want = {"reference", "book", "chapter", "verse_start", "verse_end", "text", "score", "semantic_score", "mots_cles"}
    check("/search: response schema fields", j and set(j[0]) == want, set(j[0]) ^ want if j else "empty")
    check("/search: scores are descending", all(a["score"] >= b["score"] for a, b in zip(j, j[1:])))
    check("/search: reference uses v1 format", j and j[0]["reference"].endswith("(LSG)") and ":" in j[0]["reference"], j[:1])

    check("q without query -> 422", client.get("/search", params={"q": "Dieu est amour"}).status_code == 422)
    r = client.get("/search", params={"query": "Dieu est amour", "top_k": 3})
    check("top_k is ignored; responseSize defaults to 8", r.status_code == 200 and len(r.json()) == 8,
          r.text[:200])

    r = client.get("/search", params={"query": '"l\'Éternel est mon berger"'})
    j = r.json()
    check("quoted exact phrase -> Psalm 23:1 only",
          len(j) == 1 and j[0]["book"] == "Psaumes" and j[0]["chapter"] == 23 and j[0]["verse_start"] == 1, j)

    r = client.get("/search", params={"query": "jésus pleura"})
    check("very short verse is indexed (John 11:35 at the top)",
          r.json() and r.json()[0]["reference"].startswith("Jean 11:35"), r.json()[:1])

    check("responseSize=0 is clamped to 1", len(client.get("/search", params={"query": "la foi", "responseSize": 0}).json()) == 1)
    check("responseSize=999 is capped at 20", len(client.get("/search", params={"query": "la foi", "responseSize": 999}).json()) <= 20)
    check("missing query -> 422", client.get("/search").status_code == 422)
    check("query too short -> 422", client.get("/search", params={"query": "a"}).status_code == 422)
    check("query too long -> 422", client.get("/search", params={"query": "x" * 301}).status_code == 422)

    r = client.get("/search", params={"query": "comment installer windows 11"})
    check("obviously off-topic query -> empty list (no false results)", r.status_code == 200 and r.json() == [], r.json()[:2])

    r = client.get("/search", params={"query": "qwertyuiop asdfgh"})
    check("gibberish -> empty list", r.status_code == 200 and r.json() == [], r.json()[:2])

print(f"\n{'All tests passed.' if not check.failed else str(check.failed) + ' test(s) failed.'}")
sys.exit(1 if check.failed else 0)
