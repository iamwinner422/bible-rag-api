"""French text analysis and BM25 (NumPy only; no heavy dependencies)."""
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

STOP_WORDS = frozenset(
    "a au aux avec ce ces dans de des d du elle en et eux il je j l la le les leur lui "
    "ma mais me m meme mes moi mon n ne nos notre nous on ou par pas pour qu que qui sa "
    "se s ses son sur ta te t tes toi ton tu un une vos votre vous y c ci la "
    # Forms of "être" and "avoir" (accents are removed during normalization).
    "suis es est sommes etes sont etait etaient ete etre "
    "ai as avons avez ont avait avaient eu avoir "
    # Question words and phrasing that do not carry the search topic.
    "comment pourquoi quel quelle quels quelles quoi quand dit disent bible verset versets "
    "quelqu quelque ceci cela celui celle ceux ainsi aussi tout tous toute toutes si car donc "
    "ni cette cet".split()
)


class Analyzer:
    """Normalize (lowercase, remove accents, œ→oe), tokenize, and apply light stemming.

    Stemming is intentionally simple and identical for indexing and queries:
    remove plural endings (s/x), then truncate to ``stem_len`` characters (0 = no truncation).
    This brings forms such as "pardonner", "pardonnez", and "pardon" closer together.
    """

    def __init__(self, stem_len=0, plural=True, stop_words=STOP_WORDS):
        self.stem_len = stem_len
        self.plural = plural
        self.stop = {self._stem(w) for w in stop_words}

    @staticmethod
    def normalize(text):
        t = text.lower().replace("œ", "oe").replace("æ", "ae")
        t = unicodedata.normalize("NFD", t)
        return "".join(c for c in t if unicodedata.category(c) != "Mn")

    def _stem(self, word):
        if self.plural and len(word) > 3 and word[-1] in "sx":
            word = word[:-1]
        if self.stem_len and len(word) > self.stem_len:
            word = word[:self.stem_len]
        return word

    def tokens(self, text):
        """Return all words (including stop words), in order, for phrase matching."""
        return [self._stem(w) for w in re.findall(r"[a-z0-9]+", self.normalize(text))]

    def terms(self, text):
        """Return only meaningful words (excluding stop words), in order."""
        return [w for w in self.tokens(text) if w not in self.stop]


def _contains(text_tokens, phrase_tokens):
    width = len(phrase_tokens)
    return width > 0 and any(
        text_tokens[i:i + width] == phrase_tokens
        for i in range(len(text_tokens) - width + 1)
    )


def phrase_suffix_length(analyzer, query, text):
    """Return the longest query suffix (at least 2 meaningful words) found verbatim in the text."""
    q, t = analyzer.tokens(query), analyzer.tokens(text)
    for start in range(max(0, len(q) - 1)):
        suffix = q[start:]
        if sum(w not in analyzer.stop for w in suffix) < 2:
            continue
        if _contains(t, suffix):
            return len(suffix)
    return 0


def full_phrase_length(analyzer, query, text):
    """Return the query length if it appears verbatim in the text; otherwise return 0."""
    q = analyzer.tokens(query)
    return len(q) if _contains(analyzer.tokens(text), q) else 0


def term_coverage(analyzer, query, text):
    """Return the share (0–1) of meaningful query words present in the text."""
    terms = set(analyzer.terms(query))
    if not terms:
        return 0.0
    return len(terms.intersection(analyzer.tokens(text))) / len(terms)


class BM25:
    """BM25: favor passages containing RARE query words."""

    def __init__(self, texts, analyzer, k1=1.5, b=0.75):
        self.analyzer, self.k1, self.b = analyzer, k1, b
        docs = [analyzer.tokens(t) for t in texts]
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.len.mean())
        post = defaultdict(list)  # term -> [(passage index, occurrence count)]
        for i, d in enumerate(docs):
            for w, c in Counter(d).items():
                post[w].append((i, c))
        self.ids = {w: np.array([i for i, _ in p], dtype="int32") for w, p in post.items()}
        self.tfs = {w: np.array([c for _, c in p], dtype="float32") for w, p in post.items()}
        self.idf = {w: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for w, p in post.items()}

    def scores(self, query):
        s = np.zeros(self.n, dtype="float32")
        for w in set(self.analyzer.terms(query)):
            if w not in self.ids:
                continue
            ids, tf = self.ids[w], self.tfs[w]
            norm = tf + self.k1 * (1 - self.b + self.b * self.len[ids] / self.avg)
            s[ids] += self.idf[w] * tf * (self.k1 + 1) / norm
        return s
