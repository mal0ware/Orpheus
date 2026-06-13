"""Track / routing / FX construction primitives — the APPLY verbs (atomic, LLM-orchestrated)."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient

_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_DESTRUCTIVE)
    def create_track(name: str, index: int | None = None) -> dict:
        """Insert a named track and return its stable GUID (one composite bridge call).

        ``index`` is 1-based; omit it to append at the end. The returned GUID is the
        stable identity to pass to MIDI/transport tools (never a live pointer).
        """
        params: dict = {"name": name}
        if index is not None:
            params["index"] = index
        result = BridgeClient().call("create_track", **params)
        return {
            "guid": result.get("guid"),
            "index": result.get("index"),
            "name": result.get("name", name),
        }

    @mcp.tool(annotations=_DESTRUCTIVE)
    def set_track_volume_pan(track: str, volume: str | None = None, pan: str | None = None) -> dict:
        """Set fader/pan. Accepts fuzzy values ('-6dB', '+3', '50%', 'L50', 'center')."""
        raise NotImplementedError("M1 (FX/mix) — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def add_fx_by_name(track: str, fx_name: str) -> dict:
        """Add an FX by fuzzy name, VALIDATED against the installed-plugin inventory.

        On a no-match returns a graceful "not found — here's how to install it" message
        rather than silently loading the wrong plugin (the bug all three live servers have).
        Orpheus never auto-installs plugins.
        """
        raise NotImplementedError("M1 (FX) — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def set_fx_param(track: str, fx: str, param: str, value: float) -> dict:
        """Set an FX parameter with name→index resolution (so 'ratio' works, not just index 7)."""
        raise NotImplementedError("M1 (FX) — see docs/roadmap.md")
