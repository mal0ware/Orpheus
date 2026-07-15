"""Read-only project grounding: see current state before acting (reads before writes)."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.models import Note, TrackSpec

_RO = {"readOnlyHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_RO)
    def get_project_info() -> dict:
        """Tempo, time signature, length, track count, and transport state."""
        info = BridgeClient().call("get_project_info")
        ts = info.get("time_signature") or [4, 4]
        return {
            "tempo_bpm": info.get("tempo"),
            "time_signature": [int(ts[0]), int(ts[1])],
            "length_seconds": info.get("length"),
            "num_tracks": info.get("num_tracks"),
            "playing": bool(info.get("playing")),
        }

    @mcp.tool(annotations=_RO)
    def list_tracks() -> list[TrackSpec]:
        """The typed track tree: name, vol/pan, mute/solo, FX-chain names, item count."""
        rows = BridgeClient().call("list_tracks") or []
        return [
            TrackSpec(
                guid=row.get("guid"),
                name=row.get("name", ""),
                volume_db=row.get("volume_db", 0.0),
                pan=row.get("pan", 0.0),
                mute=bool(row.get("mute")),
                solo=bool(row.get("solo")),
            )
            for row in rows
        ]

    @mcp.tool(annotations=_RO)
    def get_track_midi(track: str, at_bar: int = 1) -> dict:
        """Read a take's MIDI notes back in BEATS (pitch, start/duration, velocity).

        The inverse of insert_midi_notes' PPQ math, anchored so beat 0 == ``at_bar``;
        a note written at beat B reads back at beat B. All tick math stays in the bridge.
        """
        result = BridgeClient().call("get_track_midi", track=track, at_bar=at_bar)
        notes = [
            Note(
                pitch=n["pitch"],
                start_beat=n["start_beat"],
                duration_beats=n["duration_beats"],
                velocity=n.get("velocity", 96),
            )
            for n in result.get("notes", [])
        ]
        return {"track": result.get("track"), "notes": notes}

    if not include_stubs:
        return

    @mcp.tool(annotations=_RO)
    def get_fx_params(track: str) -> dict:
        """[NOT IMPLEMENTED] Decode an FX chain's parameters BY NAME (e.g. compressor
        ratio), not as opaque 0-1."""
        raise NotImplementedError("M1 (FX) — see docs/roadmap.md")
