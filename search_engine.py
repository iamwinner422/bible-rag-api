"""Hybrid Bible search: semantic + BM25 + exact phrase + relevance feedback.

Runs on NumPy and ONNX Runtime (no PyTorch or FAISS) to fit within Vercel limits.
"""
import os
import re
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from bm25 import BM25, Analyzer, full_phrase_length, phrase_suffix_length, term_coverage

# paraphrase-multilingual-MiniLM-L12-v2 with a reduced vocabulary of French tokens
# (Latin alphabet) to keep the deployed API small.
MODEL = "paraphrase-multilingual-MiniLM-L12-v2 (French vocabulary)"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MAX_TOKENS = 128

# Generic French question phrasing that adds no search value, e.g. "que dit la bible sur la peur".
_BOILERPLATE = re.compile(
    r"^\s*(?:(?:dis|donne|trouve|cherche)[- ]moi\s+)?"
    r"(?:que\s+(?:dit|disent)\s+(?:la\s+bible|l['’]écriture|dieu)\s+(?:sur|à propos de|au sujet de|concernant|de|du|des)"
    r"|qu['’]est[- ]ce\s+que\s+la\s+bible\s+dit\s+(?:sur|à propos de|au sujet de)"
    r"|(?:y a[- ]t[- ]il\s+)?(?:des\s+)?versets?\s+(?:sur|à propos de|au sujet de|concernant)"
    r"|parle[- ]moi\s+(?:de|du|des|d['’]))\s*",
    re.IGNORECASE,
)


def clean_query(query):
    """Remove generic question phrasing; keep the original query if nothing remains."""
    cleaned = _BOILERPLATE.sub("", query).strip(" ?!.")
    return cleaned if len(cleaned) >= 2 else query


class Embedder:
    """Transform text into normalized vectors by averaging token vectors."""

    def __init__(self, model_dir=MODEL_DIR):
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False  # Reduce memory retained between requests.
        # model.onnx is already optimized; skipping optimization makes loading ~20x cheaper.
        # A single thread is fastest on small instances.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])

    def embed(self, texts, batch_size=64):
        """Return a float32 array with one normalized vector per text."""
        parts = []
        for i in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            tokens = self.session.run(None, {
                "input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids),
            })[0].astype(np.float32)
            weights = mask[:, :, None].astype(np.float32)
            vecs = (tokens * weights).sum(1) / np.maximum(weights.sum(1), 1e-9)
            parts.append(vecs / np.linalg.norm(vecs, axis=1, keepdims=True))
        return np.vstack(parts)


def load_model():
    return Embedder()


@dataclass(frozen=True)
class SearchConfig:
    w_sem: float = 2.0        # Semantic score weight.
    w_kw: float = 1.0         # BM25 score weight.
    w_cov: float = 2.0        # Query-term coverage weight.
    w_phrase: float = 1.0     # Bonus when the query ending appears verbatim in the verse.
    depth: int = 60           # Candidates retained per method.
    min_sem: float = 0.30     # Minimum similarity for a semantic candidate.
    stem_len: int = 6         # Light stemming truncation (0 = disabled).
    clean: bool = True        # Remove generic French phrasing such as "que dit la bible sur …".
    prf_docs: int = 0         # Verses used for relevance feedback (0 = disabled; see README).
    prf_beta: float = 0.7     # Centroid weight in the expanded query.
    prf_min_sem: float = 0.40 # A verse must be this close to the query to seed feedback.
    kw_scale: float = 10.0    # A BM25 score of 10 (about one well-placed rare term) counts as 1.
    sem_floor: float = 0.20   # Cosine similarity maps to 0 below this floor...
    sem_ceil: float = 0.80    # ...and to 1 above this ceiling.
    min_top_score: float = 0.25  # Return no results below this best-match score (0 = disabled).


class Hit(NamedTuple):
    idx: int          # Verse position in the index.
    semantic: float   # Cosine similarity to the original query.
    keyword: float    # BM25 score.
    score: float      # Final ranking score (0-1).


class SearchEngine:
    def __init__(self, emb, meta, config=None, model=None, **overrides):
        self.cfg = config or SearchConfig(**overrides)
        self.model = model or load_model()
        self.emb = emb
        self.meta = meta
        self.analyzer = Analyzer(stem_len=self.cfg.stem_len)
        self.keywords = BM25([p["text"] for p in meta], self.analyzer)

    # -- ranking ------------------------------------------------------------------------------
    def _rank(self, sem, kw, query, exact_only, cache):
        """Rank candidates; return [(idx, final score)] from best to worst."""
        c = self.cfg
        candidates = {int(i) for i in np.argsort(-sem)[:c.depth] if sem[i] >= c.min_sem}
        candidates |= {int(i) for i in np.argsort(-kw)[:c.depth] if kw[i] > 0}
        if not candidates:
            return []

        phrase_fn = full_phrase_length if exact_only else phrase_suffix_length
        q_len = max(1, len(self.analyzer.tokens(query)))
        feats = {}
        for i in candidates:
            if i not in cache:
                text = self.meta[i]["text"]
                cache[i] = (phrase_fn(self.analyzer, query, text) / q_len,
                            term_coverage(self.analyzer, query, text))
            feats[i] = cache[i]

        if exact_only:
            candidates = {i for i in candidates if feats[i][0] > 0}
            if not candidates:
                return []
        else:
            # A verse selected only by an isolated word (no semantic match, phrase, or meaningful
            # query coverage) is lexical noise, so discard it.
            candidates = {i for i in candidates
                          if sem[i] >= c.min_sem or feats[i][0] > 0 or feats[i][1] >= 0.5}

        ids = list(candidates)
        s_vals = np.array([sem[i] for i in ids], dtype="float32")
        k_vals = np.array([kw[i] for i in ids], dtype="float32")
        phrase = np.array([feats[i][0] for i in ids], dtype="float32")
        cover = np.array([feats[i][1] for i in ids], dtype="float32")

        # Use ABSOLUTE normalization, not min-max normalization across candidates: a verse's score
        # does not depend on other candidates, so a common word cannot inflate tiny BM25 differences,
        # and final scores remain comparable across queries (useful for the confidence threshold).
        s_n = np.clip((s_vals - c.sem_floor) / (c.sem_ceil - c.sem_floor), 0, 1)
        k_n = np.minimum(k_vals / c.kw_scale, 1.0)
        total = c.w_sem + c.w_kw + c.w_cov + c.w_phrase
        score = (c.w_sem * s_n + c.w_kw * k_n + c.w_cov * cover + c.w_phrase * phrase) / total
        order = sorted(range(len(ids)), key=lambda j: (-float(score[j]), ids[j]))
        return [(ids[j], float(score[j])) for j in order]

    # -- search -------------------------------------------------------------------------------
    def search(self, query, size=8, semantic_only=False):
        """Return a list of Hit objects, from best to worst."""
        c = self.cfg
        query = query.strip()
        # A quoted query returns only verses containing the exact phrase.
        exact_only = len(query) >= 2 and query[0] == query[-1] == '"'
        if exact_only:
            query = query[1:-1].strip()

        sem_text = query if exact_only or not c.clean else clean_query(query)
        qv = self.model.embed([sem_text])[0]
        sem = self.emb @ qv
        sem_raw = self.emb @ self.model.embed([query])[0] if sem_text != query else sem

        if semantic_only:
            return [Hit(int(i), float(sem_raw[i]), 0.0, float(sem[i])) for i in np.argsort(-sem)[:size]]

        kw = self.keywords.scores(query)
        cache = {}
        ranked = self._rank(sem, kw, query, exact_only, cache)

        # Relevance feedback: expand the query using the centroid of the best verses
        # (in embedding space, without another model call), then rank again.
        if c.prf_docs and not exact_only and ranked:
            seeds = [i for i, _ in ranked if sem[i] >= c.prf_min_sem][:c.prf_docs]
            if len(seeds) >= 2:
                centroid = self.emb[seeds].mean(0)
                qv2 = qv + c.prf_beta * centroid / np.linalg.norm(centroid)
                sem = self.emb @ (qv2 / np.linalg.norm(qv2))
                ranked = self._rank(sem, kw, query, exact_only, cache)

        if not ranked or ranked[0][1] < c.min_top_score:
            return []  # No convincing answer (off-topic query, gibberish, etc.).
        return [Hit(i, float(sem_raw[i]), float(kw[i]), s) for i, s in ranked[:size]]
