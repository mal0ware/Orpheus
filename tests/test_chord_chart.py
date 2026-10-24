"""The owner's chord-chart notation (spec 2026-10-24 §2), as written in their song sheets.

``|`` bar line; a lone chord fills its bar; `` / `` (space on either side) splits a bar evenly;
``C/E`` (no spaces) is a slash chord; ``(xN)`` repeats the line; ``E---`` holds and fills the
section's remaining bars; ``(stop)`` and other free text become notes, never guesses.
"""

from __future__ import annotations

import pytest

from orpheus_mcp.notation.chart import parse_chord_chart


def _plain(result) -> list[tuple[str, int, float, float]]:
    return [(c.symbol, c.bar, c.beat, c.beats) for c in result.chords]


def test_one_chord_per_bar_with_a_slash_chord_and_repeat():
    r = parse_chord_chart("Am | Am | Fmaj7 | C/E (x2)")
    assert r.bars == 8 and r.notes == []
    assert _plain(r)[:4] == [("Am", 1, 1.0, 4.0), ("Am", 2, 1.0, 4.0), ("Fmaj7", 3, 1.0, 4.0),
                             ("C/E", 4, 1.0, 4.0)]
    assert _plain(r)[4] == ("Am", 5, 1.0, 4.0)


def test_spaced_slash_splits_the_bar():
    r = parse_chord_chart("Am / Fmaj7 | C / E (x4)")
    assert r.bars == 8
    assert _plain(r)[:4] == [("Am", 1, 1.0, 2.0), ("Fmaj7", 1, 3.0, 2.0), ("C", 2, 1.0, 2.0),
                             ("E", 2, 3.0, 2.0)]
    assert len(r.chords) == 16


def test_one_sided_space_still_splits():
    r = parse_chord_chart("F /C | G / Am")
    assert _plain(r) == [("F", 1, 1.0, 2.0), ("C", 1, 3.0, 2.0), ("G", 2, 1.0, 2.0),
                         ("Am", 2, 3.0, 2.0)]


def test_three_way_split():
    r = parse_chord_chart("C / G / Am")
    assert [c.beats for c in r.chords] == pytest.approx([4 / 3] * 3)


def test_stop_becomes_a_note_on_its_bar_and_extra_bars_are_silent():
    r = parse_chord_chart("Fmaj7 | C | E | E (stop)", bars=5)
    assert _plain(r)[-1] == ("E", 4, 1.0, 4.0)
    assert r.bars == 5
    assert r.notes == ["bar 4: stop", "bar 5: no chord"]


def test_held_chord_fills_the_remaining_bars():
    r = parse_chord_chart("F | C | G | Am | F | C | E---", bars=8)
    assert _plain(r)[-1] == ("E", 7, 1.0, 8.0)
    assert r.bars == 8 and r.notes == []


def test_free_text_is_kept_never_guessed():
    r = parse_chord_chart("F /C | G / Am (x4) but the last bar is just Am")
    assert r.bars == 8
    assert r.notes == ["but the last bar is just Am"]


def test_unreadable_tokens_and_ellipsis_become_notes():
    r = parse_chord_chart("Am / Fmaj7 | C / E | ... Am / Fmaj7 | EEAm")
    assert ("Am", 3, 1.0, 2.0) in _plain(r)
    assert any("'...'" in n for n in r.notes)
    assert any("EEAm" in n for n in r.notes)


def test_bar_count_mismatch_is_reported():
    r = parse_chord_chart("Am | F | C | G", bars=3)
    assert any("chart has 4 bars" in n for n in r.notes)


def test_three_four_bars_are_three_beats():
    r = parse_chord_chart("Am / E | Am", meter=(3, 4))
    assert _plain(r) == [("Am", 1, 1.0, 1.5), ("E", 1, 2.5, 1.5), ("Am", 2, 1.0, 3.0)]


@pytest.mark.parametrize(
    "symbol",
    ["C", "Am", "F#m7", "Bbmaj7", "Ebmaj9", "Gsus4", "C7sus4", "Am(maj7)", "Bm7b5", "Cdim7",
     "Caug", "Cadd9", "C6", "Dm6", "G/B", "F#m7/C#", "C5", "E7#9"],
)
def test_chord_symbols_are_recognised(symbol):
    assert _plain(parse_chord_chart(symbol)) == [(symbol, 1, 1.0, 4.0)]


def test_bar_lines_at_the_edges_are_ignored():
    assert parse_chord_chart("| Am | F |").bars == 2


def test_empty_chart_is_an_error():
    with pytest.raises(ValueError, match="no chords"):
        parse_chord_chart("   ")
