"""Tempo / meter / playback — table-stakes construction verbs. Tempo is the easiest,
highest-confidence style dimension."""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations={"destructiveHint": True})
    def set_tempo(bpm: Annotated[float, Field(gt=0, le=400)]) -> dict:
        """Set the project tempo in BPM (downstream: seconds_per_beat = 60 / bpm)."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations={"destructiveHint": True})
    def set_time_signature(numerator: Annotated[int, Field(ge=1, le=32)], denominator: int = 4) -> dict:
        """Set the project time signature."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations={"destructiveHint": True})
    def play_stop_record(command: str) -> dict:
        """Transport control: 'play' | 'stop' | 'record', via Main_OnCommand action IDs."""
        raise NotImplementedError("M1 — see docs/roadmap.md")
