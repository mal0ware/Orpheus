# src/orpheus_mcp/tools/arrange.py
"""Song arrangement: section markers + section/song builders (Slice 2)."""
from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient

_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def add_marker(name: str, bar: int = 1) -> dict:
        """Place a named marker at the start of `bar` (1-based) on the REAPER timeline."""
        result = BridgeClient().call("add_marker", name=name, bar=bar)
        return {"name": result.get("name"), "bar": result.get("bar"),
                "index": result.get("index")}
