"""Symbolic (MIDI/score) analysis via music21 — key, harmony, and the custom groove analyzer.

Pure functions over Note lists in BEATS (the repo's canonical unit), unit-testable with
in-code fixtures and zero REAPER. music21's Krumhansl-Schmuckler profile correlation is a
local computation — no corpus download, no network.
"""

from __future__ import annotations

import re
import statistics

from orpheus_mcp.models import GrooveAnalysis, HarmonyAnalysis, Mode, Note

# How many alternative key readings to surface — enough to show detection is
# probabilistic, few enough to stay readable.
_MAX_ALTERNATIVES = 3

# Track names that mean "percussion" — K-S key detection is garbage on beats, so the
# MCP layer uses this to exclude drum tracks before pooling notes for harmony analysis.
_PERCUSSIVE_RE = re.compile(
    r"(?i)\b(drum|drums|kick|snare|hat|hats|hi-?hats?|perc|percussion|clap|cymbal|tom|808)s?\b"
)


def looks_percussive(track_name: str) -> bool:
    """True if a track name says 'percussion' (drum-track filter for key detection)."""
    return bool(_PERCUSSIVE_RE.search(track_name))


def _notes_to_stream(notes: list[Note]):  # -> music21.stream.Stream
    """Build a music21 stream at beat offsets (music21 quarterLength == our beat)."""
    from music21 import note as m21note
    from music21 import stream

    s = stream.Stream()
    for n in notes:
        m = m21note.Note(n.pitch)
        m.quarterLength = n.duration_beats
        m.volume.velocity = n.velocity
        s.insert(n.start_beat, m)
    return s


def _key_to_fields(k) -> tuple[str, Mode | None, float, list[str]]:
    """music21 Key → (root, Mode, clamped confidence, alternative key names).

    Confidence is the K-S profile correlation clamped to [0, 1] (the model field is a
    0-1 score; the raw coefficient lives in [-1, 1]).
    """
    mode = Mode(k.mode) if k.mode in Mode.__members__.values() else None
    confidence = min(1.0, max(0.0, float(k.correlationCoefficient)))
    alternatives = [
        f"{alt.tonic.name} {alt.mode}"
        for alt in (k.alternateInterpretations or [])[:_MAX_ALTERNATIVES]
    ]
    return str(k.tonic.name), mode, confidence, alternatives


def analyze_key(notes: list[Note]) -> HarmonyAnalysis:
    """Krumhansl-Schmuckler key detection with confidence + alternatives.

    IMPORTANT: filter percussion/drum tracks BEFORE calling (see looks_percussive) — K-S
    detection is garbage on beats. Detection is probabilistic (~75% on tonal material,
    worse on short loops), so we always return confidence and alternatives and never
    present a single key as ground truth.
    """
    if not notes:
        return HarmonyAnalysis(note="No notes to analyze — key detection skipped.")
    key = _notes_to_stream(notes).analyze("key")
    root, mode, confidence, alternatives = _key_to_fields(key)
    note = ""
    if confidence < 0.7:
        note = (
            f"Low key confidence ({confidence:.2f}) — treat '{root} {key.mode}' as a guess "
            "and check the alternatives."
        )
    return HarmonyAnalysis(
        key_root=root,
        mode=mode,
        key_confidence=confidence,
        alternative_keys=alternatives,
        note=note,
    )


def analyze_harmony(notes: list[Note]) -> HarmonyAnalysis:
    """Chordify → Roman-numeral functional harmony → cadence detection.

    Hedges when the material is monophonic / non-chordal (a lot of real beats are):
    inferring functional harmony there is a hard MIR problem, not a clean readout, so the
    `note` field says so instead of inventing chords.
    """
    if not notes:
        return HarmonyAnalysis(note="No notes to analyze — harmony analysis skipped.")

    from music21 import roman

    base = analyze_key(notes)
    stream = _notes_to_stream(notes)
    key = stream.analyze("key")

    chords = list(stream.chordify().recurse().getElementsByClass("Chord"))
    polyphonic = [c for c in chords if len(c.pitches) >= 2]
    if not polyphonic:
        base.note = (
            "Material is monophonic — no simultaneous notes, so no chords to read. "
            "Roman-numeral harmony would be invention here; skipped."
        )
        return base

    numerals: list[str] = []
    offsets: list[float] = []
    for c in polyphonic:
        figure = roman.romanNumeralFromChord(c, key).figure
        # Collapse repeats of the SAME harmony (held/re-struck chords), keep returns.
        if not numerals or numerals[-1] != figure:
            numerals.append(figure)
            offsets.append(float(c.offset))

    base.roman_numerals = numerals
    base.cadences = _detect_cadences(numerals, offsets)
    if len(polyphonic) < len(chords) // 2:
        base.note = (
            "Mostly single-line material — the Roman numerals below cover only the "
            "chordal moments; treat them as sparse evidence, not a progression."
        )
    return base


# Cadence patterns over ROOT numerals (figures stripped of inversions/extensions).
_CADENCES: dict[tuple[str, str], str] = {
    ("V", "I"): "authentic (V-I)",
    ("V", "i"): "authentic (V-i)",
    ("IV", "I"): "plagal (IV-I)",
    ("iv", "i"): "plagal (iv-i)",
    ("V", "vi"): "deceptive (V-vi)",
    ("V", "VI"): "deceptive (V-VI)",
}


def _root_numeral(figure: str) -> str:
    """'V65' → 'V', 'ii°' -> 'ii' — keep only the leading Roman core."""
    m = re.match(r"(?i)[b#]?[iv]+", figure)
    return m.group(0) if m else figure


def _detect_cadences(numerals: list[str], offsets: list[float]) -> list[str]:
    found = []
    for i in range(len(numerals) - 1):
        pair = (_root_numeral(numerals[i]), _root_numeral(numerals[i + 1]))
        label = _CADENCES.get(pair)
        if label:
            found.append(f"{label} at beat {offsets[i + 1]:g}")
    return found


# --------------------------------------------------------------------------- #
# Groove
# --------------------------------------------------------------------------- #

# Straight 16ths plus triplet 8ths — the union grid `tightness` is measured against, so
# a swung-but-consistent performance still counts as tight.
_GRID_POINTS = (0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0)
# Half of a 16th: an onset this far from every grid point scores tightness 0.
_MAX_DEVIATION = 0.125
# Offbeat-8th window: onsets whose in-beat position lands here are "the and" of the
# beat, whose delay past 0.5 is what swing IS.
_SWING_WINDOW = (0.45, 0.72)
_STRAIGHT_OFFBEAT = 0.5
_TRIPLET_OFFBEAT = 2.0 / 3.0


def analyze_groove(notes: list[Note]) -> GrooveAnalysis:
    """Swing, tightness, velocity dynamics, and density from note onsets in beats.

    No existing DAW-MCP server provides this — a genuine Orpheus differentiator.
    Swing is measured on offbeat 8ths only (position of "the and" between 0.5 = straight
    and 2/3 = full triplet swing); tightness is mean deviation from the nearest point of
    a straight-16th + triplet-8th union grid, so swing does not read as sloppiness.
    """
    if not notes:
        return GrooveAnalysis()

    onsets = sorted(n.start_beat for n in notes)
    velocities = [float(n.velocity) for n in notes]

    # -- swing ---------------------------------------------------------------
    offbeats = [o % 1.0 for o in onsets if _SWING_WINDOW[0] <= o % 1.0 <= _SWING_WINDOW[1]]
    swing_pct: float | None = None
    if offbeats:
        mean_pos = statistics.fmean(offbeats)
        raw = (mean_pos - _STRAIGHT_OFFBEAT) / (_TRIPLET_OFFBEAT - _STRAIGHT_OFFBEAT)
        swing_pct = min(1.0, max(0.0, raw))

    # -- tightness -----------------------------------------------------------
    deviations = [
        min(abs((o % 1.0) - g) for g in _GRID_POINTS) for o in onsets
    ]
    tightness = max(0.0, 1.0 - statistics.fmean(deviations) / _MAX_DEVIATION)

    # -- density + velocity ----------------------------------------------------
    span = max(onsets[-1] - onsets[0], 1.0)
    density = len(onsets) / span
    velocity_mean = statistics.fmean(velocities)
    velocity_stddev = statistics.pstdev(velocities)

    return GrooveAnalysis(
        swing_pct=swing_pct,
        tightness=tightness,
        velocity_mean=velocity_mean,
        velocity_stddev=velocity_stddev,
        density_notes_per_beat=density,
        feel=_describe_feel(swing_pct, tightness, velocity_stddev),
    )


def _describe_feel(swing_pct: float | None, tightness: float, velocity_stddev: float) -> str:
    """Turn the numbers into the words a musician would use — the 'explain' in v0.1."""
    if tightness > 0.95:
        grid = "hard-quantized"
    elif tightness > 0.8:
        grid = "tight"
    elif tightness > 0.5:
        grid = "loose, human timing"
    else:
        grid = "very loose timing"

    if swing_pct is None:
        swing = "no offbeats to judge swing from"
    elif swing_pct > 0.6:
        swing = "heavy triplet swing"
    elif swing_pct > 0.25:
        swing = "swung"
    else:
        swing = "straight"

    dynamics = "machine-flat dynamics" if velocity_stddev < 2.0 else "dynamic velocities"
    return f"{grid}, {swing}, {dynamics}"
