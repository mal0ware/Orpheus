# src/orpheus_mcp/tools/instruments.py
"""Instrument discovery + loading tools (Slice 1). `list_installed_fx` is read-only;
`add_instrument` loads a synth or a stock drum kit. Both are thin bridge wrappers."""
from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient

_RO = {"readOnlyHint": True}
_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_RO)
    def list_installed_fx() -> dict:
        """List the plugins/instruments installed in REAPER (names). Read-only."""
        result = BridgeClient().call("list_installed_fx")
        return {"fx": result.get("fx", [])}
