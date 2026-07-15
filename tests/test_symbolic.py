"""M2 symbolic analysis: key, Roman-numeral harmony, and the groove analyzer.

Pure functions over Note lists in BEATS (the repo's canonical unit) — fixtures are
constructed in code, no .mid files, no REAPER, no network (music21's Krumhansl-Schmuckler
key profile is a local computation).
"""

from __future__ import annotations

from orpheus_mcp.analysis.symbolic import (
    analyze_groove,
    analyze_harmony,
    analyze_key,
    looks_percussive,
)
from orpheus_mcp.models import Mode, Note


def _chords(progression: list[list[int]], beats_per_chord: float = 4.0) -> list[Note]:
    """Block chords at consecutive offsets — the simplest clearly-tonal fixture."""
    notes = []
    for i, pitches in enumerate(progression):
        for p in pitches:
            notes.append(
                Note(
                    pitch=p,
                    start_beat=i * beats_per_chord,
                    duration_beats=beats_per_chord,
                    velocity=96,
                )
            )
    return notes


C_MAJOR_I_IV_V_I = _chords(
    [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]]
)
# i-iv-V-i in A minor with the harmonic-minor G# dominant — unambiguously A minor.
A_MINOR_CADENTIAL = _chords(
    [[57, 60, 64], [62, 65, 69], [64, 68, 71], [57, 60, 64]]
)


# --------------------------------------------------------------------------- #
# analyze_key
# --------------------------------------------------------------------------- #


def test_analyze_key_c_major():
    result = analyze_key(C_MAJOR_I_IV_V_I)
    assert result.key_root == "C"
    assert result.mode == Mode.MAJOR
    assert result.key_confidence is not None and result.key_confidence > 0.8
    assert result.alternative_keys  # never present a single key as ground truth


def test_analyze_key_a_minor():
    result = analyze_key(A_MINOR_CADENTIAL)
    assert result.key_root == "A"
    assert result.mode == Mode.MINOR


def test_analyze_key_empty_notes_hedges_instead_of_guessing():
    result = analyze_key([])
    assert result.key_root is None
    assert result.key_confidence is None
    assert "no notes" in result.note.lower()


# --------------------------------------------------------------------------- #
# analyze_harmony
# --------------------------------------------------------------------------- #


def test_analyze_harmony_roman_numerals_and_cadence():
    result = analyze_harmony(C_MAJOR_I_IV_V_I)
    assert result.key_root == "C"
    assert result.roman_numerals == ["I", "IV", "V", "I"]
    assert any("authentic" in c for c in result.cadences)  # V -> I


def test_analyze_harmony_monophonic_material_hedges():
    melody = [
        Note(pitch=p, start_beat=float(i), duration_beats=1.0)
        for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])
    ]
    result = analyze_harmony(melody)
    # A single line has no chords: harmony inference would be invention, so say so.
    assert "monophonic" in result.note.lower() or "chord" in result.note.lower()


def test_analyze_harmony_empty_notes():
    result = analyze_harmony([])
    assert result.roman_numerals == []
    assert "no notes" in result.note.lower()


# --------------------------------------------------------------------------- #
# analyze_groove
# --------------------------------------------------------------------------- #


def _onsets(onsets: list[float], velocity: int = 100) -> list[Note]:
    return [
        Note(pitch=42, start_beat=o, duration_beats=0.1, velocity=velocity) for o in onsets
    ]


def test_groove_straight_quantized_eighths():
    notes = _onsets([i * 0.5 for i in range(16)])  # 8 beats of straight 8ths
    g = analyze_groove(notes)
    assert g.swing_pct is not None and abs(g.swing_pct) < 0.05
    assert g.tightness is not None and g.tightness > 0.99
    assert g.density_notes_per_beat is not None and abs(g.density_notes_per_beat - 2.0) < 0.3
    assert "quantized" in g.feel


def test_groove_triplet_swing_is_full_swing():
    onsets = []
    for beat in range(8):
        onsets += [float(beat), beat + 2.0 / 3.0]  # offbeat delayed to the triplet
    g = analyze_groove(_onsets(onsets))
    assert g.swing_pct is not None and g.swing_pct > 0.9
    assert "swing" in g.feel or "swung" in g.feel


def test_groove_halfway_swing_is_partial():
    onsets = []
    for beat in range(8):
        onsets += [float(beat), beat + 7.0 / 12.0]  # halfway between 0.5 and 2/3
    g = analyze_groove(_onsets(onsets))
    assert g.swing_pct is not None and 0.3 < g.swing_pct < 0.7


def test_groove_jitter_lowers_tightness():
    jittered = [i * 0.5 + (0.04 if i % 2 else -0.03) for i in range(2, 18)]
    g = analyze_groove(_onsets(jittered))
    assert g.tightness is not None and g.tightness < 0.9


def test_groove_velocity_dynamics():
    notes = [
        Note(pitch=36, start_beat=float(i), duration_beats=0.1, velocity=v)
        for i, v in enumerate([100, 60, 100, 60])
    ]
    g = analyze_groove(notes)
    assert g.velocity_mean == 80.0
    assert g.velocity_stddev is not None and g.velocity_stddev > 15.0


def test_groove_empty_notes_returns_nothing_not_zero():
    g = analyze_groove([])
    assert g.swing_pct is None
    assert g.tightness is None
    assert g.density_notes_per_beat is None


# --------------------------------------------------------------------------- #
# percussion-name filter (feeds the drum-track filtering in the MCP layer)
# --------------------------------------------------------------------------- #


def test_looks_percussive():
    for name in ("Drums", "kick 2", "808 Snare", "Hi-Hats", "PERC loop"):
        assert looks_percussive(name), name
    for name in ("Piano", "Bass", "Lead Vox", ""):
        assert not looks_percussive(name), name
