"""Connection health — the agent's first grounding call."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations={"readOnlyHint": True})
    def get_connection_status() -> dict:
        """Check whether REAPER is running and the Orpheus bridge loop is alive.

        Call this first. Returns connection state and round-trip latency, or a clear
        "not listening" message (rather than a mystery hang) if the in-REAPER script
        isn't running.
        """
        bridge = BridgeClient()
        if not bridge.is_alive():
            return {
                "connected": False,
                "hint": "Run orpheus_bridge.lua inside REAPER (Actions → Run ReaScript).",
                "bridge_dir": str(bridge.bridge_dir),
            }
        # TODO(M0): round-trip a get_connection_status call for true latency.
        return {"connected": True, "bridge_dir": str(bridge.bridge_dir)}
