"""Evaluate a search engine (v1 or merged) on tests/eval_queries.json.

Usage: python3 -I tests/evaluate_verses.py <project-folder> [--verbose]
The folder must contain search_engine.py (with SearchEngine) and outputs/.
Metrics (for each dev / test split): Hit@5, MRR@10, nDCG@10 (binary relevance).
Expected verse lists are intentionally broad but never exhaustive: the scores are for
COMPARING two engines, not measuring absolute accuracy.
"""
import json, math, os, sys, time

import numpy as np

project = os.path.abspath(sys.argv[1])
verbose = "--verbose" in sys.argv
sys.path.insert(0, project)
os.chdir(project)
from search_engine import SearchEngine  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
ev = json.load(open(os.path.join(here, "eval_queries.json"), encoding="utf-8"))
emb = np.load(os.path.join(project, "outputs", "bible_emb.npy")).astype("float32")
meta = json.load(open(os.path.join(project, "outputs", "bible_meta.json"), encoding="utf-8"))
engine = SearchEngine(emb, meta)


def ref(i):
    p = meta[i]
    return f"{p['book']} {p['chapter']}:{p['start']}"


for split in ("dev", "test"):
    hit5, mrr, ndcg, times = [], [], [], []
    for item in ev[split]:
        gold = set(item["gold"])
        t = time.time()
        hits = engine.search(item["q"], size=10)
        got = [ref(r[0]) for r in hits]
        times.append(time.time() - t)
        rel = [1 if g in gold else 0 for g in got]
        hit5.append(1.0 if any(rel[:5]) else 0.0)
        mrr.append(next((1 / (k + 1) for k, v in enumerate(rel) if v), 0.0))
        dcg = sum(v / math.log2(k + 2) for k, v in enumerate(rel))
        idcg = sum(1 / math.log2(k + 2) for k in range(min(len(gold), 10)))
        ndcg.append(dcg / idcg if idcg else 0.0)
        if verbose:
            print(f"[{split}] {item['q']}  hit5={int(hit5[-1])} mrr={mrr[-1]:.2f}")
            for k, g in enumerate(got[:5]):
                print(f"     {'✓' if rel[k] else ' '} {g}  {meta[hits[k][0]]['text'][:70]}")
    print(f"{split:5s}  n={len(hit5):2d}  Hit@5={np.mean(hit5):.2f}  MRR@10={np.mean(mrr):.2f}  "
          f"nDCG@10={np.mean(ndcg):.2f}  avg latency={np.mean(times) * 1000:.0f} ms")
