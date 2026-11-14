# AI Bible Search API

A REST API for searching the French Louis Segond 1910 (LSG) Bible with natural-language
questions. The search engine combines semantic similarity, BM25 keyword ranking, and exact-phrase
matching. It uses a local ONNX model, so inference runs on CPU without PyTorch or an external
model service.

## Features

- Search 31,102 verses and overlapping 2–3 verse passages (89,739 indexed spans) with French
  questions or keywords.
- Combine semantic and lexical search, with a confidence threshold to reject many irrelevant
  queries.
- Search for an exact phrase by enclosing it in double quotes.
- Run the API locally or deploy it to Vercel.
- Rebuild the verse index from the included Bible data and ONNX model.

## Requirements

- Python 3.12 (the repository's `.python-version` specifies this version).
- `pip`.
- The included `model/` and `outputs/` directories for normal API use.

The model and prebuilt index are included in the repository; you do not need to download model
weights to run the API. Rebuilding the index additionally requires `data/bible_fr.json`.

## Installation

Clone or download the repository, then run these commands from its root directory.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, either adjust the execution policy for your user or use the
environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the API

From the project root, start the ASGI server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

On startup, the API loads the ONNX model and the prebuilt verse index. Once it is ready, open:

- Health status: <http://127.0.0.1:8000/health>
- Interactive API docs: <http://127.0.0.1:8000/docs>

The health endpoint (also served at `/`) returns
`{"status":"ok","verses":31102,"passages":89739}` when the index is ready: `verses` counts
single verses, `passages` counts every indexed span, including the 2–3 verse windows.
If the index cannot be loaded, the startup error is logged and the health endpoint reports it.

## Search the Bible

`GET /search` accepts `query`, an optional page number, and an optional page size:

```text
http://127.0.0.1:8000/search?query=comment%20pardonner%20%C3%A0%20quelqu%27un&page=1&pageSize=5
```

Bible references can be entered as a book and chapter, optionally followed by a verse or
verse range (for example, `Jean 3:16-18`). Chapter ranges (`Jean 3-4`) and comma-separated
verses or ranges (`Jean 3:16,18-20`) are also supported. A trailing sentence punctuation mark
is ignored.

Semantic searches return at most 16 distinct, non-overlapping verses or passages across all pages.
This limit does not apply to explicit Bible-reference searches.

For an exact phrase, surround the query with double quotes (URL-encode them when needed):

```text
http://127.0.0.1:8000/search?query=%22l%27%C3%89ternel%20est%20mon%20berger%22
```

Parameters:

| Parameter | Description |
|---|---|
| `query` | Required search question or keywords, 2–300 characters. |
| `page` | Page number, starting at 1; defaults to 1. |
| `pageSize` | Results per page; defaults to 20 and must be between 1 and 50. |

Example response:

```json
{
  "query": "Jusqu'à votre vieillesse je serai le même",
  "mode": "semantic",
  "page": 1,
  "pageSize": 20,
  "total": 16,
  "totalPages": 1,
  "results": [
    {
      "reference": "Ésaïe 46:4 (LSG)",
      "text": "Jusqu'à votre vieillesse je serai le même, Jusqu'à votre vieillesse je vous soutiendrai; ...",
      "score": 0.928,
      "semantic_score": 0.826,
      "mots_cles": true
    }
  ]
}
```

`score` is the final ranking score, not a probability. `semantic_score` is the raw cosine
similarity to the query. `mots_cles` is a legacy response field indicating whether a query term
appears in the verse. A query with no sufficiently convincing match can return an empty list.

## How search works

1. The question is kept intact: rewriting it into its topic word (for example `peur`) turns a
   thematic search into a keyword search, so `SearchConfig.clean` is off by default.
2. The query and verse text are normalized for keyword search: lowercase, accent removal,
   stop-word removal, and light stemming.
3. The engine retrieves semantic and BM25 candidates, then scores them using semantic similarity,
   keyword relevance, query-term coverage, and a phrase bonus. Questions and personal statements
   (`comment être sauvé`, `je me sens seul`) use semantic-heavy weights, because a paraphrase
   rarely shares the words of the verse; verse quotations keep the lexically weighted defaults.
4. The theme layer (`data/themes.json`, 61 themes and 661 reference verses) promotes curated
   reference verses. A question
   like `que dit la bible sur la peur` shares no word with "ne crains rien", so similarity search
   finds verses that talk *about* fear instead of the verses that answer it; the theme's verses are
   added as candidates and receive the additive bonus `SearchConfig.w_theme` (0.5). Themes are
   detected by their own term lists, and expanding the data file is how new themes are covered —
   no model retraining.
5. Overlapping passage windows are collapsed: once a verse span is returned, every lower-ranked
   hit covering the same verses of that chapter (for example `Ésaïe 46:4-5` after `Ésaïe 46:4`)
   is dropped, so a result page never shows the same verse several times.
6. A confidence threshold can suppress low-scoring results. This is a safeguard, not a guarantee
   that every off-topic query will be rejected.

The embedding model is `intfloat/multilingual-e5-small`, packaged as ONNX in `model_e5/`.
Index passages use the `passage:` prefix and search queries use `query:`. Search configuration
and ranking weights are defined by `SearchConfig` in `search_engine.py`.

## Tests and evaluation

Run commands from the project root after installing `requirements.txt`.

### API integration tests

These tests load the real model and index and exercise the API through FastAPI's test client:

```bash
python -I tests/test_api.py
```

The checks cover health status and counts, result fields and ordering, pagination, the absence of
overlapping results, accepted parameters, exact-phrase search, validation, and low-confidence
queries.

### Search-quality evaluation

Evaluate semantic-only and hybrid search on `tests/questions.json`:

```bash
python -I tests/evaluate.py
python -I tests/evaluate.py --details
```

Evaluate Hit@5, MRR@10, and nDCG@10 on the dev and test splits in
`tests/eval_queries.json`:

```bash
python -I tests/evaluate_verses.py .
python -I tests/evaluate_verses.py . --verbose
```

`tests/eval_queries.json` holds 36 dev and 34 test queries, half of them paraphrased questions that
do not contain the theme's keyword. Their expected references are not exhaustive, so treat the
scores as comparative measures between engine versions, not as absolute accuracy estimates. The
`test` split has also been used during tuning and is not a strictly independent benchmark.

The theme layer was written independently of this dataset, but 216 of the 421 expected references
are listed in it. Read the scores accordingly. On queries whose expected verse is covered by a
theme (30 dev, 30 test), Hit@5 rises from 0.23 to 0.67 (dev) and from 0.50 to 0.83 (test). On
queries outside theme coverage (6 dev, 4 test) Hit@5 is unchanged (0.83 dev, 0.75 test) with MRR
0.03 lower, because a detected theme outranks some correct hits. Overall: Hit@5 0.33 to 0.69 (dev)
and 0.53 to 0.82 (test). Adding themes is therefore the cheapest way to improve a topic, and the
remaining gap is theme coverage rather than ranking quality.

## Rebuild the index

The generated files are `outputs/bible_emb.npy` (verse embeddings) and
`outputs/bible_meta.json` (verse text and references). To rebuild both from the included data:

```bash
python scripts/build_index.py
```

This reuses the same model as the API and may take about 25 minutes on a single CPU core. To run a
small test build in a separate directory:

```bash
python scripts/build_index.py --limit 300 --out outputs-test
```

Other options include `--data PATH`, `--out PATH`, `--limit N`, and `--batch-size N`; run
`python scripts/build_index.py --help` for details. A full rebuild writes to `outputs/`, replacing
the current generated index files.

## Project layout

```text
.
├── main.py                 # FastAPI application and endpoints
├── search_engine.py        # Embeddings, candidate retrieval, and ranking
├── bm25.py                 # French text analysis and BM25 implementation
├── requirements.txt        # Python dependencies
├── model/                  # ONNX model and tokenizer
├── outputs/                # Prebuilt verse embeddings and metadata
├── data/                   # Source Bible JSON used to rebuild the index
├── scripts/build_index.py  # Index-generation script
└── tests/                  # API tests and evaluation datasets/scripts
```

## Deployment

The project includes Vercel configuration. The FastAPI application is exposed as `app` in
`main.py`, Python 3.12 is specified in `.python-version`, and dependencies are listed in
`requirements.txt`. `.vercelignore` excludes `tests/`, `scripts/`, and `data/` from deployment;
the prebuilt `model/` and `outputs/` directories are needed at runtime.

## Known limitations

- The API searches French LSG text; queries and verse names in the index are French.
- Semantic matching can miss narrative paraphrases that do not closely resemble a verse.
- The confidence threshold does not reject every unrelated query.
- Psalm verse numbering in the source appears to follow KJV numbering in some cases and may differ
  from a printed Bible edition.
