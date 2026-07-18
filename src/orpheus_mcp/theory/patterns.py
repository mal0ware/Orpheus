"""Drum-grid parsing + bassline generation. Pure; emits Note-shaped dicts in BEATS."""
from __future__ import annotations

# General-MIDI percussion notes the stock/kit instruments respond to.
GM_DRUMS: dict[str, int] = {"kick": 36, "snare": 38, "hat": 42}

_BEATS_PER_BAR = 4.0  # v1 is 4/4 only


def parse_drum_grid(pattern: str, steps_per_bar: int = 16) -> list[dict]:
    """Multi-line step grid -> note dicts. 'x' = hit, '.'/' ' = rest. One row per voice,
    'voice: xxxx'. Row label must be in GM_DRUMS. Step length = 4 beats / steps_per_bar."""
    step_beats = _BEATS_PER_BAR / steps_per_bar
    notes: list[dict] = []
    for raw in pattern.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"drum row needs 'voice: steps': {raw!r}")
        voice, cells = (s.strip() for s in line.split(":", 1))
        if voice not in GM_DRUMS:
            raise ValueError(f"unknown drum voice {voice!r}; known: {sorted(GM_DRUMS)}")
        vel = 100 if voice != "hat" else 80
        for i, ch in enumerate(cells):
            if ch == "x":
                notes.append(
                    {
                        "pitch": GM_DRUMS[voice],
                        "start_beat": round(i * step_beats, 6),
                        "duration_beats": round(step_beats, 6),
                        "velocity": vel,
                    }
                )
            elif ch not in (".", " "):
                raise ValueError(f"drum cell must be 'x' or '.', got {ch!r}")
    return notes


def bassline_notes(
    chords: list[list[int]], style: str = "root", bars_per_chord: int = 1
) -> list[dict]:
    """Turn resolved chords into a bass line. Root = lowest chord tone dropped to bass
    register is the caller's job (pass already-registered chords); here root = chords[i][0]."""
    span = _BEATS_PER_BAR * bars_per_chord
    notes: list[dict] = []
    for i, chord in enumerate(chords):
        root = chord[0]
        start = i * span
        if style == "root":
            notes.append({"pitch": root, "start_beat": start,
                          "duration_beats": span, "velocity": 100})
        elif style == "root_fifth":
            notes.append({"pitch": root, "start_beat": start,
                          "duration_beats": span / 2, "velocity": 100})
            notes.append({"pitch": root + 7, "start_beat": start + span / 2,
                          "duration_beats": span / 2, "velocity": 96})
        elif style == "octave":
            notes.append({"pitch": root, "start_beat": start,
                          "duration_beats": span / 2, "velocity": 100})
            notes.append({"pitch": root + 12, "start_beat": start + span / 2,
                          "duration_beats": span / 2, "velocity": 96})
        else:
            raise ValueError(f"unknown bass style {style!r}")
    return notes
