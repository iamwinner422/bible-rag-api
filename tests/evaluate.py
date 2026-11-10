"""Measure search quality on tests/questions.json (questions with expected verse ranges).

A question is considered "found" at rank k if one of the first k returned verses falls within
one of its expected references. Questions and metrics are from the former bible-ai-api eval/ folder.

Usage:
    python3 -I tests/evaluate.py                    # current project
    python3 -I tests/evaluate.py --project FOLDER   # compare another project (e.g. the former v1)
    python3 -I tests/evaluate.py --details          # include each question's result
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = 8


def parse_ref(ref):
    """Convert 'Actes des Apôtres 9:1-19' to (book, chapter, first verse, last verse)."""
    book, loc = ref.rsplit(" ", 1)
    if ":" not in loc:
        return book, int(loc), 1, 10**6  # Entire chapter.
    chapter, verses = loc.split(":")
    first, _, last = verses.partition("-")
    return book, int(chapter), int(first), int(last or first)


def is_hit(p, refs):
    return any(p["book"] == b and p["chapter"] == c and p["start"] <= hi and p["end"] >= lo
               for b, c, lo, hi in refs)


def evaluate(engine, questions, meta, details=False, **kwargs):
    hits = {1: 0, 3: 0, K: 0}
    rr, start = 0.0, time.time()
    for q in questions:
        refs = [parse_ref(r) for r in q["refs"]]
        results = engine.search(q["q"], size=K, **kwargs)
        rank = next((n for n, r in enumerate(results, 1) if is_hit(meta[r[0]], refs)), None)
        for k in hits:
            hits[k] += rank is not None and rank <= k
        rr += 1 / rank if rank else 0
        if details:
            top = meta[results[0][0]] if results else None
            got = f"{top['book']} {top['chapter']}:{top['start']}" if top else "-"
            print(f"    {'#' + str(rank) if rank else 'MISS':>5}  {q['q'][:58]:<58}  top: {got}")
    n = len(questions)
    return {"hit@1": hits[1] / n, "hit@3": hits[3] / n, f"hit@{K}": hits[K] / n,
            "mrr": rr / n, "ms/query": 1000 * (time.time() - start) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.path.dirname(HERE))
    ap.add_argument("--details", action="store_true")
    args = ap.parse_args()
    project = os.path.abspath(args.project)
    sys.path.insert(0, project)
    os.chdir(project)
    from search_engine import SearchEngine  # noqa: E402

    meta = json.load(open("outputs/bible_meta.json", encoding="utf-8"))
    questions = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))
    known = {(p["book"], p["chapter"]) for p in meta}
    bad = [r for q in questions for r in q["refs"] if parse_ref(r)[:2] not in known]
    if bad:
        sys.exit(f"Unknown references in questions.json: {bad}")

    engine = SearchEngine(np.load("outputs/bible_emb.npy").astype("float32"), meta)
    rows = []
    for mode, kwargs in [("semantic", {"semantic_only": True}), ("hybrid", {})]:
        if args.details:
            print(f"-- {mode}")
        rows.append((mode, evaluate(engine, questions, meta, args.details, **kwargs)))
    print(f"\n{len(questions)} questions  ({os.path.basename(project)})")
    print(f"{'mode':<11} {'hit@1':>6} {'hit@3':>6} {'hit@' + str(K):>6} {'MRR':>6} {'ms/q':>6}")
    for mode, m in rows:
        print(f"{mode:<11} {m['hit@1']:>6.0%} {m['hit@3']:>6.0%} {m[f'hit@{K}']:>6.0%} "
              f"{m['mrr']:>6.2f} {m['ms/query']:>6.0f}")


if __name__ == "__main__":
    main()
