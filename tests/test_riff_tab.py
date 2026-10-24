"""The owner's riff tab (spec 2026-10-24 §2): 4-string bass in standard tuning (E1 A1 D2 G2),
string letter + fret per slot, ``-`` holds the previous note, a bar starting with ``-`` is silent
until its first note, text before the first ``|`` is a label, ``(xN)`` repeats a line. A bar's
slots divide it evenly (8 = eighth notes in 4/4; 1 = one hit for the whole bar)."""

from __future__ import annotations

import pytest

from orpheus_mcp.notation.tab import parse_riff_tab


def _plain(result) -> list[tuple[int, float, float]]:
    return [(n.pitch, n.start_beat, n.duration_beats) for n in result.notes]


def test_eighth_note_pulse_with_holds():
    r = parse_riff_tab("Am   | A0 -  A0 -  A0 -  A0 -  |")
    assert r.bars == 1 and r.labels == ["Am"]
    assert _plain(r) == [(33, 0.0, 1.0), (33, 1.0, 1.0), (33, 2.0, 1.0), (33, 3.0, 1.0)]


def test_strings_and_frets_map_to_bass_pitches():
    r = parse_riff_tab("x | E0 E1 A3 D0 G0 G2 A0 E3 |")
    assert [n.pitch for n in r.notes] == [28, 29, 36, 38, 43, 45, 33, 31]
    assert all(n.duration_beats == 0.5 for n in r.notes)


def test_long_holds_and_lines_stack_into_bars():
    r = parse_riff_tab("Am/F | A0 -  -  -  E1 -  -  -  |\nC/E  | A3 -  -  -  E0 -  -  -  |   (x4)")
    assert r.bars == 5
    assert _plain(r)[:2] == [(33, 0.0, 2.0), (29, 2.0, 2.0)]
    assert _plain(r)[2:4] == [(36, 4.0, 2.0), (28, 6.0, 2.0)]
    assert len(r.notes) == 2 + 4 * 2


def test_a_bar_starting_with_a_hold_is_silent_until_its_first_note():
    r = parse_riff_tab("E | E0 E0 E0 E0 E0 E0 E0 E0 |\nstop | -  -  -  -  -  -  -  -  |")
    assert r.bars == 2
    assert all(n.start_beat < 4.0 for n in r.notes)  # nothing sounds in the stop bar
    r2 = parse_riff_tab("x | -  -  A0 -  -  -  -  -  |")
    assert _plain(r2) == [(33, 1.0, 3.0)]


def test_holds_never_cross_the_bar_line():
    r = parse_riff_tab("x | A0 - - - - - - - | - - - - E0 - - - |")
    assert _plain(r) == [(33, 0.0, 4.0), (28, 6.0, 2.0)]


def test_one_hit_per_bar():
    r = parse_riff_tab("(Three a.m.)     Dm | D0 |\n(Half asleep)    Am | A0 |")
    assert _plain(r) == [(38, 0.0, 4.0), (33, 4.0, 4.0)]
    assert r.labels == ["(Three a.m.) Dm", "(Half asleep) Am"]


def test_guitar_tuning_option():
    r = parse_riff_tab("x | e0 B1 G0 |", tuning="guitar")
    assert [n.pitch for n in r.notes] == [64, 60, 55]


def test_three_four_meter():
    r = parse_riff_tab("x | A0 - E0 - G0 - |", meter=(3, 4))
    assert _plain(r) == [(33, 0.0, 1.0), (28, 1.0, 1.0), (43, 2.0, 1.0)]


def test_unknown_tokens_are_reported_not_guessed():
    r = parse_riff_tab("x | A0 X9 - - |")
    assert any("X9" in w for w in r.warnings)
    assert _plain(r) == [(33, 0.0, 1.0)]  # the unknown slot silences: no guessed note


def test_empty_tab_is_an_error():
    with pytest.raises(ValueError, match="no bars"):
        parse_riff_tab("just words")
