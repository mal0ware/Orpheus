"""The load-bearing primitive: PPQ/tempo/take-correct MIDI writing.

Exposes BEATS to the model; ALL tick/PPQ/tempo-map conversion happens inside the bridge.
This is the single most correctness-sensitive code in Orpheus — every change here must
keep tests/test_midi_roundtrip.py green (a note written at beat B must read back at beat B).
"""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.models import Note

_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def insert_midi_notes(track: str, notes: list[Note], at_bar: int = 1) -> dict:
        """Batch-write notes into a take, PPQ/tempo-correct, in ONE bridge round-trip.

        Notes are in beats (pitch, start_beat, duration_beats, velocity). Batched because
        per-note round-trips would hit the ~10 ops/sec file-IPC ceiling. Capped per call
        (see BridgeClient.MAX_NOTES_PER_CALL) so REAPER's audio thread never stalls.
        """
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_midi_item(track: str, start_bar: int, length_bars: int) -> dict:
        """Create a MIDI item/take over a bar range; returns a stable take identity."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def quantize_notes(track: str, grid: str = "1/16") -> dict:
        """Quantize a take's notes to a grid (cleanup before/after analysis)."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def transpose_notes(track: str, semitones: int) -> dict:
        """Transpose a take's notes — the APPLY verb for key/mode retargeting."""
        raise NotImplementedError("M1 — see docs/roadmap.md")
