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
            from orpheus_mcp.drumkit import load_drumkit

            result = load_drumkit(BridgeClient(), track)
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

    @mcp.tool(annotations=_DESTRUCTIVE)
    def install_sound_pack(confirm: bool = False) -> dict:
        """OPTIONAL, opt-in: download ONE BSD-licensed sfizz .vst3 + a CC0 patch into a
        user-writable VST folder for a nicer default sound. Requires confirm=True. No admin,
        no installer, no licensed software. Falls back to stock if declined."""
        if not confirm:
            return {
                "installed": False,
                "note": "Set confirm=True to install the open-source sfizz + CC0 pack. "
                        "Stock instruments are used until then.",
            }
        import os
        from pathlib import Path

        from orpheus_mcp.soundpack import install_sound_pack as _install

        vst_dir = Path(os.path.expanduser("~")) / ".orpheus" / "vst"
        result = _install(vst_dir)
        # Point REAPER at the folder + rescan (best-effort; stock still works if this fails).
        try:
            BridgeClient().call("add_vst_path_and_rescan", path=str(vst_dir))
        except Exception:  # noqa: BLE001
            result["rescan"] = "manual — add the folder to REAPER's VST paths + rescan"
        return result
