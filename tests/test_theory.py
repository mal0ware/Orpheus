"""Theory layer is pure data — these run green today, no REAPER, no optional deps."""

from __future__ import annotations

import pytest

from orpheus_mcp.theory.genre_profiles import GENRE_PROFILES, get_profile
from orpheus_mcp.theory.music_theory_data import (
    diatonic_triad_pitches,
    note_to_pc,
    progression_triads,
    roman_to_degree,
    scale_notes,
    snap_to_scale,
)


def test_note_to_pc():
    assert note_to_pc("C") == 0
    assert note_to_pc("A") == 9
    assert note_to_pc("Bb") == 10


def test_c_major_scale_is_middle_c_octave():
    assert scale_notes("C", "major", 4) == [60, 62, 64, 65, 67, 69, 71]


def test_a_minor_scale():
    # A4 = 69; natural minor steps
    assert scale_notes("A", "minor", 4) == [69, 71, 72, 74, 76, 77, 79]


def test_diatonic_triads_in_c_major():
    assert diatonic_triad_pitches("C", "major", 1) == [60, 64, 67]   # I  = C major
    assert diatonic_triad_pitches("C", "major", 2) == [62, 65, 69]   # ii = D minor
    assert diatonic_triad_pitches("C", "major", 7) == [71, 74, 77]   # vii = B diminished


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        scale_notes("C", "klingon")


def test_genre_profile_lookup_is_forgiving():
    assert get_profile("Hip-Hop") is GENRE_PROFILES["hiphop"]
    with pytest.raises(ValueError):
        get_profile("polka-core")


# --------------------------------------------------------------------------- #
# M2 helpers: Roman-numeral parsing, progression voicing, key snapping
# --------------------------------------------------------------------------- #


def test_roman_to_degree_ignores_extensions():
    assert roman_to_degree("I") == 1
    assert roman_to_degree("ii") == 2
    assert roman_to_degree("V7") == 5
    assert roman_to_degree("VII") == 7
    with pytest.raises(ValueError):
        roman_to_degree("VIII")
    with pytest.raises(ValueError):
        roman_to_degree("7")


def test_progression_triads_diatonic_major():
    chords = progression_triads("C", "major", "I-IV-V-I")
    assert [r for r, _ in chords] == ["I", "IV", "V", "I"]
    assert chords[0][1] == [60, 64, 67]   # C major
    assert chords[1][1] == [65, 69, 72]   # F major
    assert chords[2][1] == [67, 71, 74]   # G major


def test_progression_triads_uppercase_v_in_minor_is_major():
    # Classical practice: "V" in a minor key means the harmonic-minor major dominant,
    # even though the natural-minor diatonic triad on 5 is minor.
    chords = progression_triads("A", "minor", "i-iv-V-i")
    v_roman, v_pitches = chords[2]
    assert v_roman == "V"
    root = v_pitches[0]
    assert [p - root for p in v_pitches] == [0, 4, 7]  # major quality


def test_progression_triads_lowercase_v_in_minor_stays_minor():
    chords = progression_triads("A", "minor", "i-v")
    _, v_pitches = chords[1]
    root = v_pitches[0]
    assert [p - root for p in v_pitches] == [0, 3, 7]  # minor quality


def test_snap_to_scale_tie_resolves_down():
    # F# (66) in C major is equidistant from F (65) and G (67) — prefer down.
    assert snap_to_scale([66], "C", "major") == [65]


def test_snap_to_scale_keeps_diatonic_pitches():
    assert snap_to_scale([60, 62, 64], "C", "major") == [60, 62, 64]


def test_snap_to_scale_unknown_mode_raises():
    with pytest.raises(ValueError):
        snap_to_scale([60], "C", "klingon")
