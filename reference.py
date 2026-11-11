import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# Normalize book names to a canonical form, e.g. "genese" -> "Genèse", "1 cor" -> "1 Corinthiens"
BOOK_ALIASES = {
    "genese": "Genèse", "gen": "Genèse", "gn": "Genèse",
    "exode": "Exode", "ex": "Exode",
    "levitique": "Lévitique", "lev": "Lévitique", "lv": "Lévitique",
    "nombres": "Nombres", "nom": "Nombres", "nb": "Nombres",
    "deuteronome": "Deutéronome", "deut": "Deutéronome", "dt": "Deutéronome",
    "josue": "Josué", "jos": "Josué",
    "juges": "Juges", "jug": "Juges", "jg": "Juges",
    "ruth": "Ruth", "rt": "Ruth",
    "1 samuel": "1 Samuel", "1samuel": "1 Samuel", "1 sam": "1 Samuel", "1s": "1 Samuel",
    "2 samuel": "2 Samuel", "2samuel": "2 Samuel", "2 sam": "2 Samuel", "2s": "2 Samuel",
    "1 rois": "1 Rois", "1rois": "1 Rois", "1r": "1 Rois",
    "2 rois": "2 Rois", "2rois": "2 Rois", "2r": "2 Rois",
    "1 chroniques": "1 Chroniques", "1chroniques": "1 Chroniques", "1 ch": "1 Chroniques", "1chr": "1 Chroniques",
    "2 chroniques": "2 Chroniques", "2chroniques": "2 Chroniques", "2 ch": "2 Chroniques", "2chr": "2 Chroniques",
    "esdras": "Esdras", "esd": "Esdras",
    "nehemie": "Néhémie", "neh": "Néhémie", "ne": "Néhémie",
    "esther": "Esther", "est": "Esther",
    "job": "Job",
    "psaumes": "Psaumes", "psaume": "Psaumes", "ps": "Psaumes", "psa": "Psaumes",
    "proverbes": "Proverbes", "prov": "Proverbes", "pr": "Proverbes",
    "ecclesiaste": "Ecclésiaste", "ecc": "Ecclésiaste", "ec": "Ecclésiaste",
    "cantique": "Cantique des cantiques", "cantique des cantiques": "Cantique des cantiques", "ct": "Cantique des cantiques",
    "esaie": "Ésaïe", "es": "Ésaïe", "isaie": "Ésaïe",
    "jeremie": "Jérémie", "jer": "Jérémie", "jr": "Jérémie",
    "lamentations": "Lamentations", "lam": "Lamentations",
    "ezechiel": "Ézéchiel", "eze": "Ézéchiel", "ez": "Ézéchiel",
    "daniel": "Daniel", "dan": "Daniel", "dn": "Daniel",
    "osee": "Osée", "os": "Osée",
    "joel": "Joël", "jl": "Joël",
    "amos": "Amos", "am": "Amos",
    "abdias": "Abdias", "abd": "Abdias",
    "jonas": "Jonas", "jon": "Jonas",
    "michee": "Michée", "mic": "Michée", "mi": "Michée",
    "nahum": "Nahum", "na": "Nahum",
    "habakuk": "Habakuk", "hab": "Habakuk",
    "sophonie": "Sophonie", "soph": "Sophonie", "so": "Sophonie",
    "aggee": "Aggée", "ag": "Aggée",
    "zacharie": "Zacharie", "zach": "Zacharie", "za": "Zacharie",
    "malachie": "Malachie", "mal": "Malachie", "ml": "Malachie",


    "matthieu": "Matthieu", "matt": "Matthieu", "mt": "Matthieu", "mat": "Matthieu",
    "marc": "Marc", "mc": "Marc", "mr": "Marc",
    "luc": "Luc", "lc": "Luc", "lk": "Luc",
    "jean": "Jean", "jn": "Jean",
    "actes": "Actes des Apôtres", "actes des apotres": "Actes des Apôtres", "ac": "Actes des Apôtres", "act": "Actes des Apôtres",
    "romains": "Romains", "rom": "Romains", "rm": "Romains",
    "1 corinthiens": "1 Corinthiens", "1corinthiens": "1 Corinthiens", "1 cor": "1 Corinthiens", "1co": "1 Corinthiens",
    "2 corinthiens": "2 Corinthiens", "2corinthiens": "2 Corinthiens", "2 cor": "2 Corinthiens", "2co": "2 Corinthiens",
    "galates": "Galates", "gal": "Galates", "ga": "Galates",
    "ephesiens": "Éphésiens", "eph": "Éphésiens", "ep": "Éphésiens",
    "philippiens": "Philippiens", "ph": "Philippiens", "php": "Philippiens",
    "colossiens": "Colossiens", "col": "Colossiens",
    "1 thessaloniciens": "1 Thessaloniciens", "1thessaloniciens": "1 Thessaloniciens", "1 th": "1 Thessaloniciens", "1th": "1 Thessaloniciens",
    "2 thessaloniciens": "2 Thessaloniciens", "2thessaloniciens": "2 Thessaloniciens", "2 th": "2 Thessaloniciens", "2th": "2 Thessaloniciens",
    "1 timothee": "1 Timothée", "1timothee": "1 Timothée", "1 tm": "1 Timothée", "1ti": "1 Timothée",
    "2 timothee": "2 Timothée", "2timothee": "2 Timothée", "2 tm": "2 Timothée", "2ti": "2 Timothée",
    "tite": "Tite", "tt": "Tite",
    "philemon": "Philémon", "phm": "Philémon",
    "hebreux": "Hébreux", "heb": "Hébreux", "he": "Hébreux",
    "jacques": "Jacques", "jac": "Jacques", "jc": "Jacques",
    "1 pierre": "1 Pierre", "1pierre": "1 Pierre", "1 p": "1 Pierre", "1pe": "1 Pierre",
    "2 pierre": "2 Pierre", "2pierre": "2 Pierre", "2 p": "2 Pierre", "2pe": "2 Pierre",
    "1 jean": "1 Jean", "1jean": "1 Jean", "1 jn": "1 Jean", "1jn": "1 Jean",
    "2 jean": "2 Jean", "2jean": "2 Jean", "2 jn": "2 Jean", "2jn": "2 Jean",
    "3 jean": "3 Jean", "3jean": "3 Jean", "3 jn": "3 Jean", "3jn": "3 Jean",
    "jude": "Jude",
    "apocalypse": "Apocalypse", "apoc": "Apocalypse", "ap": "Apocalypse", "apo": "Apocalypse",
}

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text)

@dataclass
class BibleReference:
    book: str
    chapter: int
    verse_start: Optional[int] = None
    verse_end: Optional[int] = None
    verse_ranges: Optional[list[tuple[int, int]]] = None
    chapter_end: Optional[int] = None

def parse_reference(query: str) -> Optional[BibleReference]:
    q = query.strip()
    if len(q) < 3:
        return None

    # Pattern principal
    pattern = re.compile(
        r"""
        ^\s*
        (?P<book>[\w\s]+?)                  # book name
        \s+
        (?P<chapter>\d{1,3})                 # chapter number
        (?:\s*[-–—]\s*(?P<chapter_end>\d{1,3}))?
        (?:
            \s*[:.\s]\s*                     # separator :  .  or space
            (?P<verses>\d{1,3}(?:\s*[-–—]\s*\d{1,3})?(?:\s*,\s*\d{1,3}(?:\s*[-–—]\s*\d{1,3})?)*)
        )?
        \s*[.,;!?]?$
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    m = pattern.match(q)
    if not m:
        return None

    book_raw = _normalize(m.group("book"))
    book = BOOK_ALIASES.get(book_raw)
    if not book:
        return None

    if m.group("chapter_end") and m.group("verses"):
        return None

    chapter = int(m.group("chapter"))
    chapter_end = int(m.group("chapter_end")) if m.group("chapter_end") else None
    if chapter_end is not None and chapter_end < chapter:
        return None

    verse_ranges = []
    verses = m.group("verses")
    if verses:
        for item in re.split(r"\s*,\s*", verses):
            bounds = re.split(r"\s*[-–—]\s*", item)
            start = int(bounds[0])
            end = int(bounds[1]) if len(bounds) == 2 else start
            verse_ranges.append((start, end))

    verse_start = verse_ranges[0][0] if verse_ranges else None
    verse_end = verse_ranges[0][1] if verse_ranges else None

    return BibleReference(
        book=book,
        chapter=chapter,
        chapter_end=chapter_end,
        verse_start=verse_start,
        verse_end=verse_end,
        verse_ranges=verse_ranges if len(verse_ranges) > 1 else None,
    )