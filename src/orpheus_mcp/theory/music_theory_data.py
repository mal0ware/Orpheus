"""Reimplemented music-theory primitives (not copied — original Python).

Real, working data + helpers: scale interval sets, diatonic triad qualities, and
note↔MIDI conversion. This is the cheapest, highest-leverage thing in Orpheus — it keeps
the LLM in key on day one.
"""

from __future__ import annotations

# Pitch-class index for each note name (sharps canonical; flats aliased below).
NOTE_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Semitone offsets from the tonic for each mode/scale.
SCALES: dict[str, tuple[int, ...]] = {
    "major":      (0, 2, 4, 5, 7, 9, 11),
    "minor":      (0, 2, 3, 5, 7, 8, 10),   # natural minor / aeolian
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian":     (0, 2, 3, 5, 7, 9, 10),
    "phrygian":   (0, 1, 3, 5, 7, 8, 10),
    "lydian":     (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian":    (0, 1, 3, 5, 6, 8, 10),
    "pentatonic_minor": (0, 3, 5, 7, 10),
}

# Diatonic triad qualities per scale degree (I..vii).
DIATONIC_TRIADS: dict[str, tuple[str, ...]] = {
    "major": ("maj", "min", "min", "maj", "maj", "min", "dim"),
    "minor": ("min", "dim", "maj", "min", "min", "maj", "maj"),
}

# Triad interval recipes (semitones above the root).
TRIAD_INTERVALS: dict[str, tuple[int, int, int]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
}


def note_to_pc(name: str) -> int:
    """'C', 'F#', 'Bb' → pitch class 0–11."""
    try:
        return NOTE_TO_PC[name.strip().capitalize().replace("b", "b").replace("#", "#")]
    except KeyError as exc:
        raise ValueError(f"Unknown note name: {name!r}") from exc


def scale_notes(key: str, mode: str = "major", octave: int = 4) -> list[int]:
    """MIDI note numbers for one octave of a scale (middle C = C4 = 60)."""
    if mode not in SCALES:
        raise ValueError(f"Unknown mode {mode!r}; known: {sorted(SCALES)}")
    root = note_to_pc(key) + 12 * (octave + 1)  # MIDI: C-1 = 0, so C4 = 60
    return [root + step for step in SCALES[mode]]


def diatonic_triad_pitches(key: str, mode: str, degree: int, octave: int = 4) -> list[int]:
    """MIDI pitches for the diatonic triad on scale `degree` (1-indexed I..vii)."""
    if mode not in DIATONIC_TRIADS:
        raise ValueError(f"Diatonic triads defined for {sorted(DIATONIC_TRIADS)}, not {mode!r}")
    if not 1 <= degree <= 7:
        raise ValueError("degree must be 1..7")
    scale = scale_notes(key, mode, octave)
    root = scale[degree - 1]
    quality = DIATONIC_TRIADS[mode][degree - 1]
    return [root + i for i in TRIAD_INTERVALS[quality]]


# Longest-first so 'vii' is not read as 'v' + 'ii'. Quality suffixes ('7', '°') are
# ignored on purpose — only the degree is load-bearing; quality comes from the diatonic
# table plus the numeral's case (see progression_triads).
_ROMAN_CORES: tuple[tuple[str, int], ...] = (
    ("vii", 7), ("iii", 3), ("vi", 6), ("iv", 4), ("ii", 2), ("v", 5), ("i", 1),
)


def roman_to_degree(numeral: str) -> int:
    """'V7' → 5, 'ii' → 2. Extensions/quality marks after the core are ignored."""
    token = numeral.strip()
    for core, degree in _ROMAN_CORES:
        if token.lower().startswith(core):
            rest = token[len(core):]
            # Reject 'VIII' etc.: whatever follows the core must not be more numeral.
            if rest[:1].lower() in ("i", "v"):
                continue
            return degree
    raise ValueError(f"Not a Roman-numeral degree: {numeral!r}")


def progression_triads(
    key: str, mode: str, progression: str, octave: int = 4
) -> list[tuple[str, list[int]]]:
    """'i-iv-V-i' → [(numeral, MIDI triad), ...].

    Quality starts from the diatonic table, but the numeral's CASE wins when it
    contradicts the table's third: 'V' in minor means the harmonic-minor major dominant
    (the table's natural-minor v is minor), which is how the genre-profile progressions
    are written.
    """
    chords: list[tuple[str, list[int]]] = []
    for numeral in progression.split("-"):
        numeral = numeral.strip()
        degree = roman_to_degree(numeral)
        pitches = diatonic_triad_pitches(key, mode, degree, octave)
        core_is_upper = numeral.lstrip()[0].isupper()
        third = pitches[1] - pitches[0]
        if core_is_upper and third == 3:
            pitches[1] += 1   # raise the third: minor → major (e.g. V in minor)
        elif not core_is_upper and third == 4:
            pitches[1] -= 1   # lower the third: major → minor
        chords.append((numeral, pitches))
    return chords


def snap_to_scale(pitches: list[int], key: str, mode: str = "major") -> list[int]:
    """Snap each MIDI pitch to the nearest in-scale pitch; ties resolve DOWN.

    Downward tie-breaking is deliberate: nudging a wrong note down keeps voicings and
    bass lines from creeping upward when a whole passage gets constrained.
    """
    if mode not in SCALES:
        raise ValueError(f"Unknown mode {mode!r}; known: {sorted(SCALES)}")
    allowed = {(note_to_pc(key) + step) % 12 for step in SCALES[mode]}
    out: list[int] = []
    for pitch in pitches:
        for delta in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6):
            candidate = pitch + delta
            if 0 <= candidate <= 127 and candidate % 12 in allowed:
                out.append(candidate)
                break
        else:  # pragma: no cover - every scale has a pitch within 6 semitones
            out.append(pitch)
    return out
