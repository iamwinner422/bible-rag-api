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
    j = r.json()
    check("GET / returns ok + verse and passage counts",
          r.status_code == 200 and j.get("status") == "ok"
          and j.get("verses") == 31102 and j.get("passages") == 89739, j)
    check("GET /health returns the same response", client.get("/health").json() == j)

    r = client.get("/search", params={"query": "quel est le plus grand commandement", "pageSize": 5})
    j = r.json()
    check("/search: 200, 5 results in the paginated envelope",
          r.status_code == 200 and j["mode"] == "semantic" and len(j["results"]) == 5
          and j["page"] == 1 and j["pageSize"] == 5, r.text[:200])
    check("/search: Matthew 22 at the top", j["results"][0]["reference"].startswith("Matthieu 22"),
          j["results"][:1])
    want = {"reference", "text", "score", "semantic_score", "mots_cles"}
    check("/search: result schema fields", set(j["results"][0]) == want,
          set(j["results"][0]) ^ want)
    check("/search: scores are descending",
          all(a["score"] >= b["score"] for a, b in zip(j["results"], j["results"][1:])))
    check("/search: reference uses (LSG) format",
          j["results"][0]["reference"].endswith("(LSG)") and ":" in j["results"][0]["reference"],
          j["results"][:1])

    r = client.get("/search", params={"query": "la foi", "pageSize": 8})
    j = r.json()
    spans = [tuple(int(x) for x in res["reference"].split(":")[1].split(" ")[0].split("-"))
             for res in j["results"]]
    spans = [(s[0], s[-1]) for s in spans]  # "46:4" -> (4, 4), "46:4-6" -> (4, 6)
    books = [res["reference"].split(":")[0] for res in j["results"]]
    duplicate = any(books[i] == books[k] and spans[i][0] <= spans[k][1] and spans[k][0] <= spans[i][1]
                    for i in range(len(spans)) for k in range(i + 1, len(spans)))
    check("/search: no overlapping passage windows in results", not duplicate, j["results"])

    r = client.get("/search", params={"query": "Jean 3:16"})
    j = r.json()
    check("reference search returns the requested verse",
          j["mode"] == "reference" and j["total"] == 1
          and j["results"][0]["reference"] == "Jean 3:16 (LSG)", j)

    r = client.get("/search", params={"query": "quel est le plus grand commandement", "pageSize": 10, "page": 2})
    j = r.json()
    check("/search: pagination returns the remaining results",
          j["total"] == 16 and j["totalPages"] == 2 and len(j["results"]) == 6, j["totalPages"])

    check("q without query -> 422", client.get("/search", params={"q": "Dieu est amour"}).status_code == 422)
    check("pageSize=0 -> 422", client.get("/search", params={"query": "la foi", "pageSize": 0}).status_code == 422)
    check("pageSize=999 -> 422", client.get("/search", params={"query": "la foi", "pageSize": 999}).status_code == 422)
    check("page=0 -> 422", client.get("/search", params={"query": "la foi", "page": 0}).status_code == 422)
    check("missing query -> 422", client.get("/search").status_code == 422)
    check("query too short -> 422", client.get("/search", params={"query": "a"}).status_code == 422)
    check("query too long -> 422", client.get("/search", params={"query": "x" * 301}).status_code == 422)

    r = client.get("/search", params={"query": "\"l'Éternel est mon berger\""})
    j = r.json()
    check("quoted exact phrase -> Psalm 23:1 only",
          j["total"] == 1 and j["results"][0]["reference"] == "Psaumes 23:1 (LSG)", j)

    r = client.get("/search", params={"query": "jésus pleura"})
    check("very short verse is indexed (John 11:35 at the top)",
          r.json()["results"] and r.json()["results"][0]["reference"].startswith("Jean 11:35"),
          r.json()["results"][:1])

    r = client.get("/search", params={"query": "comment installer windows 11"})
    check("obviously off-topic query -> empty results",
          r.status_code == 200 and r.json()["results"] == [], r.json()["results"][:2])

    r = client.get("/search", params={"query": "qwertyuiop asdfgh"})
    check("gibberish -> empty results", r.json()["results"] == [], r.json()["results"][:2])

print(f"\n{'All tests passed.' if not check.failed else str(check.failed) + ' test(s) failed.'}")
sys.exit(1 if check.failed else 0)
