"""Theory layer is pure data — these run green today, no REAPER, no optional deps."""

from __future__ import annotations

import pytest

from orpheus_mcp.theory.genre_profiles import GENRE_PROFILES, get_profile
from orpheus_mcp.theory.music_theory_data import (
    diatonic_triad_pitches,
    note_to_pc,
    scale_notes,
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
