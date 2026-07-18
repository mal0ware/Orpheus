"""Chord-symbol parsing, dual-notation progression resolution, and voice-leading.

Pure and REAPER-free. Builds on music_theory_data's pitch-class + triad tables and the
existing Roman-numeral resolver (progression_triads) so the two notations agree.
"""
from __future__ import annotations

import re

from orpheus_mcp.theory.music_theory_data import (
    NOTE_TO_PC,
    TRIAD_INTERVALS,
    progression_triads,
    roman_to_degree,
)

# quality token -> triad key in TRIAD_INTERVALS
_QUALITIES: dict[str, str] = {
    "": "maj", "maj": "maj", "M": "maj",
    "m": "min", "min": "min", "-": "min",
    "dim": "dim", "aug": "aug", "+": "aug",
}
# seventh token -> semitones above root
_SEVENTHS: dict[str, int] = {"7": 10, "maj7": 11}

_ROOT_RE = re.compile(r"^([A-Ga-g][#b]?)(.*)$")


def parse_chord_symbol(symbol: str, octave: int = 4) -> list[int]:
    """'Cm7' -> MIDI pitches for the voiced chord at `octave` (C4 = 60).

    Grammar: root(A-G)(#|b)? quality(m|min|-|dim|aug|+|maj|M|"") seventh(7|maj7)?.
    Dim/aug take no seventh in v1; a trailing '7' is a (dominant) flat-7 except after
    'maj'. Unknown input raises ValueError.
    """
    token = symbol.strip()
    m = _ROOT_RE.match(token)
    if not m:
        raise ValueError(f"not a chord symbol: {symbol!r}")
    root_name, rest = m.group(1), m.group(2)
    try:
        root_pc = NOTE_TO_PC[root_name[:1].upper() + root_name[1:].lower()]
    except KeyError as exc:
        raise ValueError(f"bad chord root: {symbol!r}") from exc

    seventh: int | None = None
    if rest.endswith("maj7"):
        seventh, rest = _SEVENTHS["maj7"], rest[:-4]
    elif rest.endswith("7"):
        seventh, rest = _SEVENTHS["7"], rest[:-1]

    if rest not in _QUALITIES:
        raise ValueError(f"bad chord quality in {symbol!r}: {rest!r}")
    quality = _QUALITIES[rest]

    if seventh is not None and quality in ("dim", "aug"):
        raise ValueError(f"dim/aug chords take no seventh in v1: {symbol!r}")

    base = root_pc + 12 * (octave + 1)  # MIDI: C-1 = 0, so C4 = 60
    pitches = [base + i for i in TRIAD_INTERVALS[quality]]
    if seventh is not None:
        pitches.append(base + seventh)
    return pitches


def _looks_roman(spec: str) -> bool:
    tokens = [t.strip() for t in spec.split("-") if t.strip()]
    if not tokens:
        return False
    try:
        for t in tokens:
            roman_to_degree(t)
        return True
    except ValueError:
        return False


def resolve_progression(
    spec: str, key: str | None = None, mode: str = "minor", octave: int = 4
) -> list[list[int]]:
    """Resolve either notation to a list of chord pitch-lists.

    - Comma-separated OR non-Roman '-'-separated  -> absolute chord symbols.
    - Roman numerals ('i-iv-V-i')                 -> requires `key`; uses progression_triads.
    """
    if "," in spec:
        return [parse_chord_symbol(tok, octave) for tok in spec.split(",") if tok.strip()]
    if _looks_roman(spec):
        if key is None:
            raise ValueError("Roman-numeral progressions require a `key`.")
        return [pitches for _numeral, pitches in progression_triads(key, mode, spec, octave)]
    # dash-separated symbols, e.g. "Cm7-Fm7-Bb7"
    return [parse_chord_symbol(tok, octave) for tok in spec.split("-") if tok.strip()]
