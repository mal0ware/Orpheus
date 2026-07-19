"""Parse a model-friendly melody notation into Note-shaped dicts (beats). Pure."""
from __future__ import annotations

import re

from orpheus_mcp.theory.music_theory_data import NOTE_TO_PC, snap_to_scale

DURATIONS: dict[str, float] = {"w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25}

_TOKEN = re.compile(r"^([A-Ga-g][#b]?)(-?\d+)?:([whqes])$")
_REST = re.compile(r"^[rR]:([whqes])$")


def parse_melody(
    notation: str, key: str | None = None, mode: str | None = None, velocity: int = 90
) -> list[dict]:
    """'A4:q C5:q E5:h' -> sequential note dicts. `r:dur` is a rest. Tokens are
    whitespace-separated `Name[octave]:dur` (octave defaults to 4). If key+mode given,
    non-rest pitches are snapped into the scale so the line stays in key."""
    notes: list[dict] = []
    beat = 0.0
    tokens = notation.split()
    if not tokens:
        raise ValueError("empty melody")
    for tok in tokens:
        rest = _REST.match(tok)
        if rest:
            beat += DURATIONS[rest.group(1)]
            continue
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"bad melody token: {tok!r}")
        name, octave, dur = m.group(1), m.group(2), m.group(3)
        try:
            pc = NOTE_TO_PC[name[:1].upper() + name[1:].lower()]
        except KeyError as exc:
            raise ValueError(f"bad note name in {tok!r}") from exc
        octn = int(octave) if octave is not None else 4
        pitch = pc + 12 * (octn + 1)  # MIDI: C4 = 60
        if key is not None and mode is not None:
            pitch = snap_to_scale([pitch], key, mode)[0]
        notes.append({"pitch": pitch, "start_beat": round(beat, 6),
                      "duration_beats": DURATIONS[dur], "velocity": velocity})
        beat += DURATIONS[dur]
    return notes
