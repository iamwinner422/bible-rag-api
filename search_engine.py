"""Hybrid Bible search: semantic + BM25 + exact phrase + relevance feedback.

Runs on NumPy and ONNX Runtime (no PyTorch or FAISS) to fit within Vercel limits.
"""
import os
import re
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from bm25 import BM25, Analyzer, full_phrase_length, phrase_suffix_length, term_coverage
from themes import ThemeLayer
from embedder_e5 import E5Embedder

# intfloat/multilingual-e5-small, exported as ONNX.
MODEL = "intfloat/multilingual-e5-small"
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

_TEACHING_STRUCTURE = re.compile(
    r"""
    (?:dans\s+quel\s+passage|où|quand|en\s+quel\s+endroit|à\s+quel\s+endroit)
    .{0,60}?                                                
    (?:enseigne|enseignait|enseignent|enseigné|
       parle|parlait|parlent|parlé|
       dit|disait|disent|dit|
       explique|expliquait|expliqué|
       montre|montrait|montré|
       proclame|proclamait|proclamé|
       donne|donnait|donné)
    (?:\s+(?:sur|de|du|des|à\s+propos\s+de|au\s+sujet\s+de|concernant))?
    \s*(?P<topic>.+)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


# First word of a question or of a personal statement ("j'ai peur de...", "comment...").
# Such queries describe a situation rather than quote a verse, so lexical coverage is a poor
# signal and semantic similarity should dominate.
_QUESTION_WORDS = {
    "quel", "quelle", "quels", "quelles", "comment", "pourquoi", "quand", "ou", "où",
    "qui", "combien", "faut", "dois", "doit", "puis", "peut", "est", "y",
    "je", "j", "me", "m", "mon", "ma", "mes", "moi", "nous", "on", "tu", "te", "toi",
}


def question_like(query: str) -> bool:
    """True when the query is a question or a personal statement, not a verse quotation."""
    q = query.strip()
    if not q:
        return False
    if q.endswith("?"):
        return True

    q_norm = q.lower().replace("’", "'")
    q_norm = q_norm.lstrip(" \t\n\r\"'“”‘’([{")

    if q_norm.startswith("que "):
        rest = q_norm[4:].lstrip()
        if not re.match(r"^(?:dit|dis|disent|signifie|est-ce|veux|veut|pense|penses|dois|doit|faut|peux|sais|as|a|ai|est|sont)\b", rest):
            return False

    if q_norm.startswith(("comment", "pourquoi", "quand", "ou", "qui", "combien",
                          "quel", "quelle", "quels", "quelles",
                          "qu'est-ce", "qu'est", "qu'a", "qu'y a-t-il")):
        return True

    if q_norm.startswith(("je", "j", "me", "m", "mon", "ma", "mes", "moi", "nous", "on",
                          "tu", "te", "toi", "il", "elle", "ils", "elles")):
        return True

    first_match = re.match(r"^([^\s,;:!?.]+)", q_norm)
    if not first_match:
        return False

    first = first_match.group(1)
    first_base = first.split("'")[0]
    if first_base in {"que", "qu"}:
        rest = q_norm[len(first):].lstrip(" \t\n\r\"'“”‘’([{")
        if re.match(r"^(?:dit|dis|disent|signifie|est-ce|veux|veut|pense|penses|dois|doit|faut|peux|sais|as|a|ai|est|sont)\b", rest):
            return True
        return False

    return first_base in _QUESTION_WORDS


def rewrite_query(query: str) -> str:
    """
    Reformule les questions d'enseignement de façon générale
    (fonctionne pour Jésus, les prophètes, les apôtres, Moïse, etc.)
    """
    q = query.strip()

    m = _TEACHING_STRUCTURE.search(q)
    if m:
        topic = m.group("topic").strip(" ?!.")
        # On force le modèle à chercher l'enseignement + le sujet
        return f"enseigne {topic} enseignement {topic} {topic}"

    # Nettoyage classique
    cleaned = _BOILERPLATE.sub("", q).strip(" ?!.")
    return cleaned if len(cleaned) >= 2 else q



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
    return E5Embedder("model_e5")


@dataclass(frozen=True)
class SearchConfig:
    w_sem: float = 2.0        # Semantic score weight.
    w_kw: float = 1.0         # BM25 score weight.
    w_cov: float = 2.0        # Query-term coverage weight.
    w_phrase: float = 1.0     # Bonus when the query ending appears verbatim in the verse.
    depth: int = 60           # Candidates retained per method.
    min_sem: float = 0.30     # Minimum similarity for a semantic candidate.
    stem_len: int = 6         # Light stemming truncation (0 = disabled).
    clean: bool = False       # Remove generic phrasing such as "que dit la bible sur …".
                              # Off by default: rewriting a question into its topic word turns a
                              # thematic search into a keyword search (see README).
    adaptive: bool = True     # Switch to question weights for questions and personal statements.
    question_w_sem: float = 4.0     # Semantic weight for question-like queries.
    question_w_kw: float = 0.5      # BM25 weight (a paraphrase rarely shares the verse's words).
    question_w_cov: float = 1.0     # Query-term coverage weight.
    question_w_phrase: float = 0.5  # Verbatim phrase weight.
    question_min_sem: float = 0.25  # A paraphrase often sits around 0.25 cosine similarity.
    prf_docs: int = 0         # Verses used for relevance feedback (0 = disabled; see README).
    prf_beta: float = 0.7     # Centroid weight in the expanded query.
    prf_min_sem: float = 0.40 # A verse must be this close to the query to seed feedback.
    kw_scale: float = 10.0    # A BM25 score of 10 (about one well-placed rare term) counts as 1.
    sem_floor: float = 0.20   # Cosine similarity maps to 0 below this floor...
    sem_ceil: float = 1.00    # ...and to 1 above this ceiling (kept at 1.0 so a verbatim quote,
                              # whose cosine is ~0.83, is not clamped to a saturated 1.0 score).
    min_top_score: float = 0.19  # Return no results below this best-match score (0 = disabled).
    w_theme: float = 0.5      # Additive bonus for the theme layer's reference verses (0 = disabled).

    def for_query(self, query, exact_only):
        """Return the config to rank this query with."""
        if not self.adaptive or exact_only or not question_like(query):
            return self
        return replace(self, w_sem=self.question_w_sem, w_kw=self.question_w_kw,
                       w_cov=self.question_w_cov, w_phrase=self.question_w_phrase,
                       min_sem=self.question_min_sem)


class Hit(NamedTuple):
    idx: int          # Verse position in the index.
    semantic: float   # Cosine similarity to the original query.
    keyword: float    # BM25 score.
    score: float      # Final ranking score (0-1).


class SearchEngine:
    def __init__(self, emb, meta, config=None, model=None, themes=None, **overrides):
        self.cfg = config or SearchConfig(**overrides)
        self.model = model or load_model()
        self.emb = emb
        self.meta = meta
        self.analyzer = Analyzer(stem_len=self.cfg.stem_len)
        self.keywords = BM25([p["text"] for p in meta], self.analyzer)
        self.themes = ThemeLayer(meta) if themes is None else themes

    # -- ranking ------------------------------------------------------------------------------
    def _rank(self, sem, kw, query, exact_only, cache, cfg, boost=None):
        """Rank candidates; return [(idx, final score)] from best to worst."""
        c = cfg
        candidates = {int(i) for i in np.argsort(-sem)[:c.depth] if sem[i] >= c.min_sem}
        candidates |= {int(i) for i in np.argsort(-kw)[:c.depth] if kw[i] > 0}
        if boost is not None:
            # A theme's reference verses are candidates even when neither method ranked them.
            candidates |= {int(i) for i in np.flatnonzero(boost > 0)}
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
                          if sem[i] >= c.min_sem or feats[i][0] > 0 or feats[i][1] >= 0.5
                          or (boost is not None and boost[i] > 0)}

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
        if boost is not None and c.w_theme > 0:
            # Additive bonus: a themed verse whose own cosine is mediocre (a single verse reads
            # differently from the wider passage) must still be able to outrank a merely topical one.
            score = np.minimum(score + c.w_theme * np.array([boost[i] for i in ids], dtype="float32"), 1.0)
        order = sorted(range(len(ids)), key=lambda j: (-float(score[j]), ids[j]))
        return [(ids[j], float(score[j])) for j in order]

    def _drop_overlapping(self, ranked):
        """Keep only the best hit per verse span.

        The index stores single verses plus 2-3 verse windows, so the same verse can reach the
        results several times (46:4, 46:4-5, 46:3-5, ...). Once a span is returned, every other
        hit overlapping it in the same chapter is redundant and is skipped.
        """
        kept, occupied = [], {}
        for idx, score in ranked:
            p = self.meta[idx]
            spans = occupied.setdefault((p["book"], p["chapter"]), [])
            start, end = p["start"], p["end"]
            if any(start <= s_end and s_start <= end for s_start, s_end in spans):
                continue
            spans.append((start, end))
            kept.append((idx, score))
        return kept

    # -- search -------------------------------------------------------------------------------
    def search(self, query, size=8, semantic_only=False):
        """Return a list of Hit objects, from best to worst."""
        c = self.cfg
        query = query.strip()
        # A quoted query returns only verses containing the exact phrase.
        exact_only = len(query) >= 2 and query[0] == query[-1] == '"'
        if exact_only:
            query = query[1:-1].strip()

        cfg = c.for_query(query, exact_only)
        sem_text = query if exact_only or not c.clean else rewrite_query(query)
        qv = self.model.embed([sem_text], is_query=True)[0]
        sem = self.emb @ qv
        sem_raw = (
            self.emb @ self.model.embed([query], is_query=True)[0]
            if sem_text != query
            else sem
        )

        if semantic_only:
            return [Hit(int(i), float(sem_raw[i]), 0.0, float(sem[i])) for i in np.argsort(-sem)[:size]]

        kw = self.keywords.scores(query)
        cache = {}
        boost = None if exact_only else self.themes.match(query)[0]
        ranked = self._rank(sem, kw, query, exact_only, cache, cfg, boost)

        # Relevance feedback: expand the query using the centroid of the best verses
        # (in embedding space, without another model call), then rank again.
        if c.prf_docs and not exact_only and ranked:
            seeds = [i for i, _ in ranked if sem[i] >= c.prf_min_sem][:c.prf_docs]
            if len(seeds) >= 2:
                centroid = self.emb[seeds].mean(0)
                qv2 = qv + c.prf_beta * centroid / np.linalg.norm(centroid)
                sem = self.emb @ (qv2 / np.linalg.norm(qv2))
                ranked = self._rank(sem, kw, query, exact_only, cache, cfg, boost)

        if not ranked or ranked[0][1] < c.min_top_score:
            return []  # No convincing answer (off-topic query, gibberish, etc.).
        ranked = self._drop_overlapping(ranked)
        return [Hit(i, float(sem_raw[i]), float(kw[i]), s) for i, s in ranked[:size]]
