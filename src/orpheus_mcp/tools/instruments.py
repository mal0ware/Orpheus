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

    @mcp.tool(annotations=_DESTRUCTIVE)
    def add_instrument(track: str, kind: str = "named", name: str | None = None) -> dict:
        """Load an instrument so a track is audible. kind='named' adds `name` (idempotent);
        kind='drumkit' loads a stock 3-voice kit (kick/snare/hat) from bundled samples."""
        if kind == "drumkit":
            import tempfile
            from pathlib import Path

            from orpheus_mcp.drumkit import ensure_drum_samples

            samples = ensure_drum_samples(Path(tempfile.gettempdir()) / "orpheus_drumkit")
            result = BridgeClient().call(
                "add_instrument", track=track, kind="drumkit", samples=samples
            )
        else:
            if not name:
                raise ValueError("kind='named' requires an instrument name")
            result = BridgeClient().call(
                "add_instrument", track=track, kind="named", name=name
            )
        return {
            "track": result.get("track"),
            "loaded": result.get("loaded"),
            "already_present": result.get("already_present", False),
        }
