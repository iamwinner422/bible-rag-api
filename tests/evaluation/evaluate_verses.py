"""Evaluate search quality using Hit@5, MRR@10 and nDCG@10."""

import argparse
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np

from tests.evaluation.project_loader import load_search_engine

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    SearchEngine = load_search_engine(project)

    with (HERE / "eval_queries.json").open(encoding="utf-8") as file:
        evaluation_data = json.load(file)

    embeddings_path = project / "outputs" / "bible_emb.npy.gz"
    if embeddings_path.is_file():
        with gzip.open(embeddings_path, "rb") as file:
            embeddings = np.load(file).astype("float32")
    else:
        embeddings = np.load(project / "outputs" / "bible_emb.npy").astype("float32")
    meta_path = project / "outputs" / "bible_meta.json.gz"
    if meta_path.is_file():
        with gzip.open(meta_path, "rt", encoding="utf-8") as file:
            meta = json.load(file)
    else:
        with (project / "outputs" / "bible_meta.json").open(encoding="utf-8") as file:
            meta = json.load(file)

    engine = SearchEngine(embeddings, meta)

    def reference(index):
        passage = meta[index]
        return f"{passage['book']} {passage['chapter']}:{passage['start']}"

    for split in ("dev", "test"):
        hit_at_five, reciprocal_ranks, ndcgs, durations = [], [], [], []
        for item in evaluation_data[split]:
            gold = set(item["gold"])
            start = time.time()
            results = engine.search(item["q"], size=10)
            got = [reference(result[0]) for result in results]
            durations.append(time.time() - start)
            relevance = [int(result in gold) for result in got]
            hit_at_five.append(float(any(relevance[:5])))
            reciprocal_ranks.append(
                next(
                    (
                        1 / (rank + 1)
                        for rank, is_relevant in enumerate(relevance)
                        if is_relevant
                    ),
                    0.0,
                )
            )
            dcg = sum(
                value / math.log2(rank + 2) for rank, value in enumerate(relevance)
            )
            ideal_dcg = sum(
                1 / math.log2(rank + 2) for rank in range(min(len(gold), 10))
            )
            ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
            if args.verbose:
                print(
                    f"[{split}] {item['q']}  hit5={int(hit_at_five[-1])} "
                    f"mrr={reciprocal_ranks[-1]:.2f}"
                )
                for rank, result in enumerate(results[:5]):
                    marker = "✓" if relevance[rank] else " "
                    passage = meta[result[0]]
                    print(f"     {marker} {got[rank]}  {passage['text'][:70]}")

        print(
            f"{split:5s}  n={len(hit_at_five):2d}  "
            f"Hit@5={np.mean(hit_at_five):.2f}  "
            f"MRR@10={np.mean(reciprocal_ranks):.2f}  "
            f"nDCG@10={np.mean(ndcgs):.2f}  "
            f"avg latency={np.mean(durations) * 1000:.0f} ms"
        )


if __name__ == "__main__":
    main()
