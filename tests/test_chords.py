"""Pure chord parsing/resolution — no REAPER, no optional deps."""
from __future__ import annotations

import pytest

from orpheus_mcp.theory.chords import parse_chord_symbol, resolve_progression


def test_major_triad():
    assert parse_chord_symbol("C", octave=4) == [60, 64, 67]


def test_minor_triad():
    assert parse_chord_symbol("Am", octave=4) == [69, 72, 76]


def test_dominant_seventh_adds_flat7():
    assert parse_chord_symbol("C7", octave=4) == [60, 64, 67, 70]


def test_minor_seventh():
    assert parse_chord_symbol("Cm7", octave=4) == [60, 63, 67, 70]


def test_major_seventh():
    assert parse_chord_symbol("Cmaj7", octave=4) == [60, 64, 67, 71]


def test_flat_root():
    assert parse_chord_symbol("Bb", octave=4) == [70, 74, 77]


def test_rejects_garbage():
    for bad in ("H", "", "C#b", "Xmaj"):
        with pytest.raises(ValueError):
            parse_chord_symbol(bad)


def test_resolve_symbol_progression_by_comma():
    prog = resolve_progression("Cm7, Fm7, Bb7")
    assert prog[0] == [60, 63, 67, 70]
    assert len(prog) == 3


def test_resolve_roman_progression_needs_key():
    prog = resolve_progression("i-iv-V-i", key="A", mode="minor")
    assert len(prog) == 4
    # i in A minor = A minor triad at octave 4 -> A4=69
    assert prog[0] == [69, 72, 76]


def test_resolve_roman_without_key_raises():
    with pytest.raises(ValueError):
        resolve_progression("i-iv-V-i")
