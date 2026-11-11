"""Rebuild the search index (outputs/bible_emb.npy + outputs/bible_meta.json).

No PyTorch or sentence-transformers: reuse the ONNX model in model/, the same one used by the API,
so the generated index is exactly what the API will query.
Allow ~25 minutes to process the entire Bible on a single core.

Usage (from the project root):
    python3 scripts/build_index.py                      # rebuild outputs/
    python3 scripts/build_index.py --limit 300 --out /tmp/test_index   # quick test
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from search_engine import MODEL, load_model  # noqa: E402


def load_books(path):
    """Return [(book name, [chapter = [verse texts]])] from the Bible JSON."""
    with open(path, encoding="utf-8-sig") as file:
        data = json.load(file)
    books = []
    if "Testaments" in data:  # Testaments > Books > Chapters > Verses
        for testament in data["Testaments"]:
            for book in testament["Books"]:
                chapters = [[v["Text"] for v in ch["Verses"]] for ch in book["Chapters"]]
                books.append((book["Text"], chapters))
    elif "verses" in data:  # Flat list: {book_name, chapter, verse, text}.
        index = {}
        for v in data["verses"]:
            index.setdefault(v["book_name"], {}).setdefault(v["chapter"], []).append((v["verse"], v["text"]))
        for name, chapters in index.items():
            books.append((name, [[t for _, t in sorted(chapters[c])] for c in sorted(chapters)]))
    else:
        raise ValueError('Unrecognized JSON format (expected "Testaments" or "verses").')
    return books


def clean(text):
    """Remove the pilcrow (¶) and extra whitespace."""
    return " ".join(text.replace("¶", " ").split())


def build_passages(path, max_window=3):
    """
    Génère :
    - tous les versets isolés
    - des passages de 2 à max_window versets consécutifs
    """
    passages = []
    
    for name, chapters in load_books(path):
        for c, verses in enumerate(chapters, start=1):
            n = len(verses)
            
            # 1. Versets isolés
            for v, text in enumerate(verses, start=1):
                passages.append({
                    "book": name,
                    "chapter": c,
                    "start": v,
                    "end": v,
                    "text": clean(text),
                    "type": "verse"
                })
            
            # 2. Passages de 2 à max_window versets
            for window in range(2, max_window + 1):
                for start in range(n - window + 1):
                    end = start + window
                    texts = [clean(verses[i]) for i in range(start, end)]
                    combined = " ".join(texts)
                    
                    passages.append({
                        "book": name,
                        "chapter": c,
                        "start": start + 1,
                        "end": end,
                        "text": combined,
                        "type": "passage"
                    })
    
    return passages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "bible_fr.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--limit", type=int, default=0, help="Encode only the first N verses (for testing).")
    ap.add_argument("--batch-size", type=int, default=250)
    args = ap.parse_args()

    passages = build_passages(args.data, max_window=3)
    if args.limit:
        passages = passages[:args.limit]
    print(f"Embedding {len(passages)} verses ({MODEL})...")

    model = load_model()
    texts = [p["text"] for p in passages]
    parts, start = [], time.time()
    for i in range(0, len(texts), args.batch_size):
        parts.append(model.embed(texts[i:i + args.batch_size]))
        done = min(i + args.batch_size, len(texts))
        elapsed = time.time() - start
        print(f"  {done}/{len(texts)} ({100 * done // len(texts)}%)  "
              f"elapsed {elapsed / 60:.1f} min, ~{elapsed / done * (len(texts) - done) / 60:.1f} min remaining",
              flush=True)

    vecs = np.vstack(parts)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # Cosine similarity becomes a dot product.

    os.makedirs(args.out, exist_ok=True)
    # float16: half the storage with no noticeable quality loss (the API reloads as float32).
    np.save(os.path.join(args.out, "bible_emb.npy"), vecs.astype("float16"))
    with open(os.path.join(args.out, "bible_meta.json"), "w", encoding="utf-8") as file:
        json.dump(passages, file, ensure_ascii=False)
    print(f"Done: wrote bible_emb.npy and bible_meta.json to {args.out}")


if __name__ == "__main__":
    main()
