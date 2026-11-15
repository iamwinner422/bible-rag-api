"""Theme layer: map a thematic question onto the verses that answer it.

A paraphrase such as "que dit la bible sur la peur" shares no words with "ne crains rien", and a
generic sentence encoder only finds verses that *talk about* fear, not the canonical ones. The theme
layer closes that gap with curated data instead of another model: query terms select a theme, and
the theme's reference verses are promoted in the ranking.
"""
import json
import os
import re
import unicodedata

import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "themes.json")

# One matched term is already strong evidence: "peur" must be able to outrank a verse that merely
# mentions fear, so even a single match is near full strength.
_STRENGTH = (0.85, 0.95, 1.0)  # one, two, three or more matched terms


def normalize(text):
    """Lowercase, unaccented, punctuation-free form used for term matching."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("’", "'").replace("'", " ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _ref(entry):
    return f"{entry['book']} {entry['chapter']}:{entry['start']}"


class ThemeLayer:
    """Curated themes with their reference verses, and matching against a query."""

    def __init__(self, meta, path=None):
        self.masks = {}      # theme id -> float32 boost array aligned with the index
        self.labels = {}     # theme id -> human label
        self.terms = {}      # theme id -> normalized detection terms
        path = path or DEFAULT_PATH
        if not os.path.exists(path):
            self._empty = True
            return
        self._empty = False

        # A reference can also start a multi-verse window, so single verses are mapped first.
        position = {}
        for i, entry in enumerate(meta):
            if entry["start"] == entry["end"]:
                position.setdefault(_ref(entry), i)
        for i, entry in enumerate(meta):
            position.setdefault(_ref(entry), i)

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for theme in data["themes"]:
            indexes = [position[r] for r in theme["verses"] if r in position]
            if not indexes:
                continue
            mask = np.zeros(len(meta), dtype="float32")
            mask[indexes] = 1.0
            self.masks[theme["id"]] = mask
            self.labels[theme["id"]] = theme["label"]
            self.terms[theme["id"]] = [t for t in (normalize(x) for x in theme["terms"]) if t]

    def __len__(self):
        return len(self.masks)

    def match(self, query):
        """Return (boost array or None, [(theme id, strength)]) for a query."""
        if self._empty:
            return None, []
        haystack = f" {normalize(query)} "
        found = []
        for theme_id, terms in self.terms.items():
            hits = sum(1 for t in terms if f" {t} " in haystack)
            if hits:
                found.append((theme_id, _STRENGTH[min(hits, len(_STRENGTH)) - 1]))
        if not found:
            return None, []
        found.sort(key=lambda x: -x[1])
        top = found[:2]
        boost = np.zeros_like(next(iter(self.masks.values())))
        for theme_id, strength in top:
            np.maximum(boost, self.masks[theme_id] * strength, out=boost)
        return boost, top
