"""Generate-from-scratch composers (v1-stretch). Thin orchestrators over the same public
primitives the agent can call directly — composers never get private superpowers."""

from __future__ import annotations

from fastmcp import FastMCP

_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_chord_progression(track: str, chords: str, key: str | None = None) -> dict:
        """Text chord names ('Cm7, Fm7, Bb7') → voiced, in-key, tempo-correct MIDI."""
        raise NotImplementedError("M4 — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_drum_pattern(track: str, pattern: str) -> dict:
        """Step-grid string → GM drum MIDI (kick 36 / snare 38 / hat 42, ch9), humanized."""
        raise NotImplementedError("M4 — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def humanize_pass(track: str) -> dict:
        """~12ms timing + 6-velocity jitter on generated MIDI — a cheap, big quality win."""
        raise NotImplementedError("M4 — see docs/roadmap.md")
