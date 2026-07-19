"""Pure melody-notation parsing — no REAPER."""
from __future__ import annotations

import pytest

from orpheus_mcp.theory.melody import parse_melody


def test_simple_line_pitches_and_timing():
    notes = parse_melody("A4:q C5:q E5:h")
    assert [n["pitch"] for n in notes] == [69, 72, 76]
    assert [n["start_beat"] for n in notes] == [0.0, 1.0, 2.0]
    assert [n["duration_beats"] for n in notes] == [1.0, 1.0, 2.0]


def test_rest_advances_time_without_a_note():
    notes = parse_melody("A4:q r:q A4:q")
    assert [n["start_beat"] for n in notes] == [0.0, 2.0]
    assert len(notes) == 2


def test_default_octave_is_4():
    assert parse_melody("C:q")[0]["pitch"] == 60


def test_accidentals():
    assert parse_melody("F#4:q")[0]["pitch"] == 66
    assert parse_melody("Bb3:q")[0]["pitch"] == 58


def test_in_key_snapping():
    # F#4 (66) is out of C major; snapped to nearest in-scale pitch.
    notes = parse_melody("F#4:q", key="C", mode="major")
    assert notes[0]["pitch"] in (65, 67)  # F or G


def test_rejects_bad_token():
    for bad in ("H4:q", "A4:z", "A4", "A4:"):
        with pytest.raises(ValueError):
            parse_melody(bad)
