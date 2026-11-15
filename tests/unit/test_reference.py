from app.references import get_reference_verses, parse_reference


def test_parses_single_verse_and_range():
    assert parse_reference("Jean 3:16").verse_start == 16
    assert parse_reference("Jean 3:16-18").verse_end == 18


def test_parses_comma_separated_verses_and_ranges():
    reference = parse_reference("Jean 3:16, 18-20")

    assert reference is not None
    assert reference.verse_start == 16
    assert reference.verse_end == 16
    assert reference.verse_ranges == [(16, 16), (18, 20)]


def test_parses_chapter_ranges():
    reference = parse_reference("Jean 3-4")

    assert reference is not None
    assert reference.chapter == 3
    assert reference.chapter_end == 4
    assert parse_reference("Jean 4-3") is None
    assert parse_reference("Jean 3-4:16") is None


def test_ignores_trailing_sentence_punctuation():
    reference = parse_reference("Jean 3:16-18.")

    assert reference is not None
    assert reference.verse_start == 16
    assert reference.verse_end == 18


def test_reference_lookup_selects_only_requested_verses():
    reference = parse_reference("Jean 3:16,18-20")
    assert reference is not None
    metadata = [
        {"book": "Jean", "chapter": 3, "start": verse, "end": verse}
        for verse in (15, 16, 17, 18, 19, 20, 21)
    ]

    assert [verse["start"] for verse in get_reference_verses(reference, metadata)] == [
        16,
        18,
        19,
        20,
    ]


def test_chapter_range_lookup_includes_all_verses_in_range():
    reference = parse_reference("Jean 3-4.")
    assert reference is not None
    metadata = [
        {"book": "Jean", "chapter": chapter, "start": verse, "end": verse}
        for chapter, verse in ((4, 1), (3, 2), (2, 1), (3, 1), (5, 1))
    ]

    assert [
        (verse["chapter"], verse["start"])
        for verse in get_reference_verses(reference, metadata)
    ] == [(3, 1), (3, 2), (4, 1)]
