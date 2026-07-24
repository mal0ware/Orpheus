"""Track / routing / FX construction primitives — the APPLY verbs (atomic, LLM-orchestrated)."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient

_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
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

    if not include_stubs:
        return

    @mcp.tool(annotations=_DESTRUCTIVE)
    def set_track_volume_pan(track: str, volume: str | None = None, pan: str | None = None) -> dict:
        """[NOT IMPLEMENTED] Set fader/pan. Accepts fuzzy values ('-6dB', '+3', '50%',
        'L50', 'center')."""
        raise NotImplementedError("M1 (FX/mix) — see docs/roadmap.md")

    # add_fx_by_name / set_fx_param graduated to real implementations in
    # orpheus_mcp.tools.mix (Task 11.1, §8 step 3) — no longer stubbed here.
