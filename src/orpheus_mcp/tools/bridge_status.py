"""Connection health — the agent's first grounding call."""

from __future__ import annotations

import time

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient, BridgeError
from orpheus_mcp.bridge.client import read_heartbeat


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:  # noqa: ARG001 - uniform signature
    @mcp.tool(annotations={"readOnlyHint": True})
    def get_connection_status() -> dict:
        """Check whether REAPER is running and the Orpheus bridge loop is alive.

        Call this first. Round-trips a real ping through the bridge and returns REAPER's
        version + latency (and, with heartbeat v2, which project is open). If REAPER is open
        but the bridge isn't, it starts the bridge first; otherwise it says why it couldn't
        (``autostart``) rather than hanging.
        """
        bridge = BridgeClient()
        if not bridge.ensure_alive():  # tries starting the bridge once if REAPER is open
            status_off: dict = {
                "connected": False,
                "hint": "Open REAPER: the bridge starts with it after `orpheus-mcp "
                        "install-bridge`. In a REAPER that's already open, `orpheus-mcp "
                        "start-bridge` starts it.",
                "bridge_dir": str(bridge.bridge_dir),
            }
            if bridge.last_autostart:
                status_off["autostart"] = bridge.last_autostart
            return status_off
        try:
            t0 = time.monotonic()
            info = bridge.call("get_connection_status")
            status: dict = {
                "connected": True,
                "reaper_version": info.get("reaper_version"),
                "bridge_dir": str(bridge.bridge_dir),
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            }
            hb = read_heartbeat(bridge.bridge_dir) or {}
            if hb.get("v", 0) >= 2:  # heartbeat v2: which project is open
                status["project"] = {
                    "name": hb.get("project_name", ""),
                    "path": hb.get("project_path", ""),
                    "playing": bool(hb.get("playing")),
                    "state_change_count": hb.get("state_change_count"),
                }
            return status
        except BridgeError as exc:
            return {"connected": False, "error": str(exc), "bridge_dir": str(bridge.bridge_dir)}
