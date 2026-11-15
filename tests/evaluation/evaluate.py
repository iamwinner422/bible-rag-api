"""Measure search quality on the expected verse ranges in questions.json."""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from tests.evaluation.project_loader import load_search_engine

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
K = 8


def parse_ref(ref):
    """Convert a reference into (book, chapter, first verse, last verse)."""
    book, location = ref.rsplit(" ", 1)
    if ":" not in location:
        return book, int(location), 1, 10**6
    chapter, verses = location.split(":")
    first, _, last = verses.partition("-")
    return book, int(chapter), int(first), int(last or first)


def is_hit(passage, refs):
    return any(
        passage["book"] == book
        and passage["chapter"] == chapter
        and passage["start"] <= high
        and passage["end"] >= low
        for book, chapter, low, high in refs
    )


def evaluate(engine, questions, meta, details=False, **kwargs):
    hits = {1: 0, 3: 0, K: 0}
    reciprocal_rank, start = 0.0, time.time()
    for question in questions:
        refs = [parse_ref(reference) for reference in question["refs"]]
        results = engine.search(question["q"], size=K, **kwargs)
        rank = next(
            (
                number
                for number, result in enumerate(results, 1)
                if is_hit(meta[result[0]], refs)
            ),
            None,
        )
        for cutoff in hits:
            hits[cutoff] += rank is not None and rank <= cutoff
        reciprocal_rank += 1 / rank if rank else 0
        if details:
            top = meta[results[0][0]] if results else None
            found = f"{top['book']} {top['chapter']}:{top['start']}" if top else "-"
            print(
                f"    {'#' + str(rank) if rank else 'MISS':>5}  "
                f"{question['q'][:58]:<58}  top: {found}"
            )
    count = len(questions)
    return {
        "hit@1": hits[1] / count,
        "hit@3": hits[3] / count,
        f"hit@{K}": hits[K] / count,
        "mrr": reciprocal_rank / count,
        "ms/query": 1000 * (time.time() - start) / count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    SearchEngine = load_search_engine(project)

    meta_path = project / "outputs" / "bible_meta.json.gz"
    if meta_path.is_file():
        with gzip.open(meta_path, "rt", encoding="utf-8") as file:
            meta = json.load(file)
    else:
        with (project / "outputs" / "bible_meta.json").open(encoding="utf-8") as file:
            meta = json.load(file)

    with (HERE / "questions.json").open(encoding="utf-8") as file:
        questions = json.load(file)
    known = {(passage["book"], passage["chapter"]) for passage in meta}
    invalid = [
        reference
        for question in questions
        for reference in question["refs"]
        if parse_ref(reference)[:2] not in known
    ]
    if invalid:
        parser.error(f"Unknown references in questions.json: {invalid}")

    embeddings_path = project / "outputs" / "bible_emb.npy.gz"
    if embeddings_path.is_file():
        with gzip.open(embeddings_path, "rb") as file:
            embeddings = np.load(file).astype("float32")
    else:
        embeddings = np.load(project / "outputs" / "bible_emb.npy").astype("float32")
    engine = SearchEngine(embeddings, meta)
    rows = []
    for mode, kwargs in [
        ("semantic", {"semantic_only": True}),
        ("hybrid", {}),
    ]:
        if args.details:
            print(f"-- {mode}")
        rows.append(
            (
                mode,
                evaluate(engine, questions, meta, args.details, **kwargs),
            )
        )

    print(f"\n{len(questions)} questions  ({project.name})")
    print(
        f"{'mode':<11} {'hit@1':>6} {'hit@3':>6} {'hit@' + str(K):>6} {'MRR':>6} {'ms/q':>6}"
    )
    for mode, metrics in rows:
        print(
            f"{mode:<11} {metrics['hit@1']:>6.0%} {metrics['hit@3']:>6.0%} "
            f"{metrics[f'hit@{K}']:>6.0%} {metrics['mrr']:>6.2f} "
            f"{metrics['ms/query']:>6.0f}"
        )


if __name__ == "__main__":
    main()
