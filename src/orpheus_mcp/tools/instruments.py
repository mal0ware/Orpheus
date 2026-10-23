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
        """OPTIONAL, opt-in: download a small default sound pack. Three CC0 (public domain)
        FreePats instruments: Spanish classical guitar, picked electric bass, and an upright
        piano (loud layer). About 24.5 MB from FreePats' GitHub at pinned commits, every file
        sha256-checked, into ~/.orpheus/soundpack. They play through REAPER's own sampler
        (RS5k): no plugin, no installer, no rescan. Requires confirm=True. After installing,
        composers use these instead of stock ReaSynth when no better installed instrument
        exists, and load_sound_pack_instrument puts one on any track."""
        from orpheus_mcp import soundpack

        if not confirm:
            manifest = soundpack.load_manifest()
            size = sum(f["size"] for i in manifest["instruments"] for f in i["files"])
            names = ", ".join(i["name"] for i in manifest["instruments"])
            return {
                "installed": False,
                "note": f"Set confirm=True to download the CC0 FreePats pack "
                        f"({size / 1e6:.1f} MB: {names}). Stock instruments are used until then.",
            }
        return soundpack.install_sound_pack()

    @mcp.tool(annotations=_DESTRUCTIVE)
    def load_sound_pack_instrument(track: str, instrument: str) -> dict:
        """Put an installed sound-pack instrument ('piano', 'bass', 'guitar') on `track` as
        REAPER sampler (RS5k) zones, one per recorded note range. Run install_sound_pack first."""
        from orpheus_mcp.instruments import load_instrument

        return load_instrument(
            BridgeClient(), track, {"kind": "pack", "name": instrument, "source": "pack"}
        )
