"""Read-only project grounding: see current state before acting (reads before writes)."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.models import TrackSpec

_RO = {"readOnlyHint": True}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_RO)
    def get_project_info() -> dict:
        """Tempo, time signature, length, track count, and transport state."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def list_tracks() -> list[TrackSpec]:
        """The typed track tree: name, vol/pan, mute/solo, role, FX-chain names, item count."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def get_track_midi(track: str) -> dict:
        """Read a take's MIDI notes (pitch, start/duration in BEATS, velocity) + CC. Feeds analysis."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def get_fx_params(track: str) -> dict:
        """Decode an FX chain's parameters BY NAME (e.g. compressor ratio), not as opaque 0–1."""
        raise NotImplementedError("M1 — see docs/roadmap.md")
