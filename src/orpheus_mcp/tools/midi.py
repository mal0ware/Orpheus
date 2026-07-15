"""The load-bearing primitive: PPQ/tempo/take-correct MIDI writing.

Exposes BEATS to the model; ALL tick/PPQ/tempo-map conversion happens inside the bridge.
This is the single most correctness-sensitive code in Orpheus — every change here must
keep tests/test_midi_roundtrip.py green (a note written at beat B must read back at beat B).
"""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.bridge.client import MAX_NOTES_PER_CALL
from orpheus_mcp.models import Note

_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def insert_midi_notes(track: str, notes: list[Note], at_bar: int = 1) -> dict:
        """Batch-write notes into a take, PPQ/tempo-correct, in ONE bridge round-trip.

        Notes are in beats (pitch, start_beat, duration_beats, velocity). Batched because
        per-note round-trips would hit the ~10 ops/sec file-IPC ceiling. Capped per call
        (see BridgeClient.MAX_NOTES_PER_CALL) so REAPER's audio thread never stalls.

        ``at_bar`` (1-based) anchors beat 0 of ``notes``. The beats→PPQ conversion is
        performed in the bridge against the project tempo/QN map, never here.
        """
        if len(notes) > MAX_NOTES_PER_CALL:
            raise ValueError(
                f"insert_midi_notes accepts at most {MAX_NOTES_PER_CALL} notes per call "
                f"(got {len(notes)}); split into multiple calls."
            )
        payload = [
            {
                "pitch": n.pitch,
                "start_beat": n.start_beat,
                "duration_beats": n.duration_beats,
                "velocity": n.velocity,
            }
            for n in notes
        ]
        result = BridgeClient().call(
            "insert_midi_notes", track=track, notes=payload, at_bar=at_bar
        )
        return {
            "track": result.get("track"),
            "inserted": result.get("inserted", len(notes)),
            "at_bar": result.get("at_bar", at_bar),
        }

    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_midi_item(track: str, start_bar: int = 1, length_bars: int = 1) -> dict:
        """Create a MIDI item/take over a bar range; returns a stable take identity."""
        result = BridgeClient().call(
            "create_midi_item", track=track, start_bar=start_bar, length_bars=length_bars
        )
        return {
            "track": result.get("track"),
            "item_index": result.get("item_index"),
            "start_bar": result.get("start_bar", start_bar),
            "length_bars": result.get("length_bars", length_bars),
        }

    if include_stubs:

        @mcp.tool(annotations=_DESTRUCTIVE)
        def quantize_notes(track: str, grid: str = "1/16") -> dict:
            """[NOT IMPLEMENTED] Quantize a take's notes to a grid (cleanup
            before/after analysis)."""
            raise NotImplementedError("M1 (quantize) — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def transpose_notes(track: str, semitones: int) -> dict:
        """Transpose a take's notes — the APPLY verb for key/mode retargeting.

        Notes that would fall outside MIDI 0–127 are left untouched; the response
        reports how many notes were actually moved.
        """
        result = BridgeClient().call("transpose_notes", track=track, semitones=semitones)
        return {
            "track": result.get("track"),
            "transposed": result.get("transposed"),
            "semitones": result.get("semitones", semitones),
        }
