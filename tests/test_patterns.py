"""Pure drum-grid + bassline generation — no REAPER."""
from __future__ import annotations

import pytest

from orpheus_mcp.theory.patterns import GM_DRUMS, bassline_notes, parse_drum_grid


def test_kick_on_every_quarter():
    pattern = "kick: x...x...x...x..."
    notes = parse_drum_grid(pattern, steps_per_bar=16)
    assert len(notes) == 4
    assert all(n["pitch"] == GM_DRUMS["kick"] for n in notes)
    assert [n["start_beat"] for n in notes] == [0.0, 1.0, 2.0, 3.0]


def test_multi_voice_grid():
    pattern = "kick:  x...x...x...x...\nsnare: ....x.......x...\nhat:   x.x.x.x.x.x.x.x."
    notes = parse_drum_grid(pattern)
    pitches = {n["pitch"] for n in notes}
    assert pitches == {GM_DRUMS["kick"], GM_DRUMS["snare"], GM_DRUMS["hat"]}
    snares = [n for n in notes if n["pitch"] == GM_DRUMS["snare"]]
    assert [n["start_beat"] for n in snares] == [1.0, 3.0]


def test_unknown_voice_raises():
    with pytest.raises(ValueError):
        parse_drum_grid("cowbell: x...")


def test_bassline_root_style_one_note_per_chord():
    chords = [[60, 64, 67], [65, 69, 72]]  # C, F
    notes = bassline_notes(chords, style="root", bars_per_chord=1)
    assert [n["pitch"] for n in notes] == [60, 65]
    assert [n["start_beat"] for n in notes] == [0.0, 4.0]
    assert notes[0]["duration_beats"] == 4.0


def test_bassline_root_fifth():
    chords = [[60, 64, 67]]
    notes = bassline_notes(chords, style="root_fifth", bars_per_chord=1)
    assert [n["pitch"] for n in notes] == [60, 67]
    assert [n["start_beat"] for n in notes] == [0.0, 2.0]
