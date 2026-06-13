"""Tempo / meter / playback — table-stakes construction verbs. Tempo is the easiest,
highest-confidence style dimension."""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from orpheus_mcp.bridge import BridgeClient

_TRANSPORT_COMMANDS = ("play", "stop", "record")


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations={"destructiveHint": True})
    def set_tempo(bpm: Annotated[float, Field(gt=0, le=400)]) -> dict:
        """Set the project tempo in BPM (downstream: seconds_per_beat = 60 / bpm)."""
        result = BridgeClient().call("set_tempo", bpm=bpm)
        return {"tempo_bpm": result.get("tempo", bpm)}

    @mcp.tool(annotations={"destructiveHint": True})
    def set_time_signature(
        numerator: Annotated[int, Field(ge=1, le=32)], denominator: int = 4
    ) -> dict:
        """Set the project time signature."""
        result = BridgeClient().call(
            "set_time_signature", numerator=numerator, denominator=denominator
        )
        ts = result.get("time_signature") or [numerator, denominator]
        return {"time_signature": [int(ts[0]), int(ts[1])]}

    @mcp.tool(annotations={"destructiveHint": True})
    def play_stop_record(command: str) -> dict:
        """Transport control: 'play' | 'stop' | 'record', via Main_OnCommand action IDs."""
        cmd = command.strip().lower()
        if cmd not in _TRANSPORT_COMMANDS:
            raise ValueError(
                f"command must be one of {_TRANSPORT_COMMANDS}, got {command!r}"
            )
        result = BridgeClient().call("play_stop_record", command=cmd)
        return {"command": cmd, "play_state": result.get("play_state")}
