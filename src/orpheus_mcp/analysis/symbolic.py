"""Symbolic (MIDI/score) analysis via music21 — key, harmony, and the custom groove analyzer.

These are pure functions over note data, unit-testable with .mid fixtures and zero REAPER.
"""

from __future__ import annotations

from orpheus_mcp.models import GrooveAnalysis, HarmonyAnalysis, Note


def analyze_key(notes: list[Note]) -> HarmonyAnalysis:
    """Krumhansl-Schmuckler key detection with confidence + alternatives.

    IMPORTANT: filter percussion/drum tracks BEFORE calling — K-S detection is garbage on
    beats. Detection is probabilistic (~75% on tonal material, worse on short loops), so we
    always return confidence and alternatives and never present a single key as ground truth.

    Implementation (M2): build a music21 stream from `notes`, call ``stream.analyze('key')``,
    read ``key.tonalCertainty()`` and ``key.alternateInterpretations``.
    """
    raise NotImplementedError("M2 — see docs/roadmap.md")


def analyze_harmony(notes: list[Note]) -> HarmonyAnalysis:
    """Chordify + Roman-numeral functional harmony + cadence detection.

    Implementation (M2): ``stream.chordify()`` then
    ``roman.romanNumeralFromChord(chord, key)`` per chord. Hedge when the material is
    monophonic / non-chordal (a lot of real beats are), because inferring functional
    harmony there is a hard MIR problem, not a clean readout.
    """
    raise NotImplementedError("M2 — see docs/roadmap.md")


def analyze_groove(onsets_ppq: list[int], ppq_per_beat: int = 960) -> GrooveAnalysis:
    """Swing% and tightness from raw MIDI onset deviation off the nearest 8th/16th grid.

    No existing DAW-MCP server provides this — it's a genuine Orpheus differentiator.
    Pure arithmetic over onset positions; trivially unit-testable.
    """
    raise NotImplementedError("M2 — see docs/roadmap.md")
