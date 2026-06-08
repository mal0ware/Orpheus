"""In-key scaffolding so the LLM doesn't drift out of key. A read-only knowledge oracle
backed by music21 + reimplemented theory tables + ported genre profiles."""

from __future__ import annotations

from fastmcp import FastMCP

_RO = {"readOnlyHint": True}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_RO)
    def get_scale_notes(key: str, mode: str = "major") -> dict:
        """MIDI notes for a key/mode so generated notes stay diatonic."""
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def suggest_chord_progression(key: str, mode: str = "major", genre: str | None = None) -> dict:
        """A genre-typical diatonic progression as Roman numerals + concrete MIDI."""
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def constrain_to_key(notes: list[int], key: str, mode: str = "major") -> dict:
        """Snap a proposed note set to the nearest in-key pitches before writing."""
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def get_genre_profile(genre: str) -> dict:
        """A style's typical progressions / scales / BPM range / instruments / drum rhythms.
        The RECOMMEND-side lookup."""
        raise NotImplementedError("M2 — see docs/roadmap.md")
