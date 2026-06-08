"""The UNDERSTAND brain (north-star half 1). Read-only; builds CompositionSpec(current)
plus the LLM-Readable Music Report. Logic lives in orpheus_mcp.analysis (pure functions)."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.models import AudioCharacter, GrooveAnalysis, HarmonyAnalysis, MusicReport

_RO = {"readOnlyHint": True}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_RO)
    def analyze_harmony() -> HarmonyAnalysis:
        """Detect key/scale/chords/Roman numerals/cadences.

        Exports the project's MIDI, filters drum tracks (Krumhansl detection is garbage on
        beats), and runs music21. Always returns a confidence + alternatives; on drum-only
        or non-chordal material it hedges rather than inventing harmony.
        """
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def analyze_groove() -> GrooveAnalysis:
        """Compute swing%% and tightness from raw MIDI PPQ onset deviation off the grid.
        No existing DAW-MCP server provides this — a genuine Orpheus differentiator."""
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def analyze_audio_character() -> AudioCharacter:
        """Render stems/master to WAV and compute the post-FX sonic fingerprint:
        3-band energy, spectral centroid, LUFS, true peak, crest factor, stereo width.
        (Sharper with the optional [analysis] extra / librosa.)"""
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def build_project_spec() -> MusicReport:
        """Fuse symbolic + audio analysis into one CompositionSpec(current) + a plain-English
        report. The contract handed to recommend_changes."""
        raise NotImplementedError("M2 — see docs/roadmap.md")
