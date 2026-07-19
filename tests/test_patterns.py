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


def test_space_is_a_rest_not_removed():
    # Space is a valid rest char that occupies a step, exactly like '.'.
    notes = parse_drum_grid("kick: x   x   x   x   ", steps_per_bar=16)
    assert [n["start_beat"] for n in notes] == [0.0, 1.0, 2.0, 3.0]
    assert len(notes) == 4


def test_named_patterns_parse_and_are_nonempty():
    from orpheus_mcp.theory.patterns import DRUM_PATTERNS

    assert {"backbeat", "halftime", "fourfloor"}.issubset(DRUM_PATTERNS)
    for name, grid in DRUM_PATTERNS.items():
        hits = parse_drum_grid(grid)
        assert hits, f"{name} produced no hits"


def test_backbeat_has_snare_on_2_and_4():
    from orpheus_mcp.theory.patterns import DRUM_PATTERNS

    snares = [h for h in parse_drum_grid(DRUM_PATTERNS["backbeat"])
              if h["pitch"] == 38]
    assert [s["start_beat"] for s in snares] == [1.0, 3.0]
