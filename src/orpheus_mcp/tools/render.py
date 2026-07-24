"""Headless render/bounce — for auditioning and for feeding the audio analyzer."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from orpheus_mcp.bridge import BridgeClient

# The render output directory: a `renders/` subdir of the same cache root the analysis
# CacheFS uses (~/.orpheus_cache, overridable via ORPHEUS_CACHE_DIR — the value of
# CFG.CACHE_DIR_ENV). Resolved locally (not via analysis.cachefs) so this tool stays free
# of the heavy analysis deps and is never skipped when they are absent.
#
# LRU note (§9): CacheFS.evict_lru iterates EVERY dir under the cache root, so `renders/`
# is already visible to size_bytes() and evictable — but as ONE whole directory keyed by
# its own mtime, not per-render, and only pinnable via the parent's in-process pin set.
# Renders are project-side artifacts, so coarse eviction is acceptable; if per-render LRU
# is ever wanted, renders would need registering as pcm-hash-style entries. See report.
_CACHE_DIR_ENV = "ORPHEUS_CACHE_DIR"


def _renders_dir() -> Path:
    override = os.environ.get(_CACHE_DIR_ENV)
    root = Path(override) if override else Path.home() / ".orpheus_cache"
    d = root / "renders"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _basename_for(track: str, start_s: float, end_s: float) -> str:
    """A filesystem-safe, region-stable basename (no extension). Track refs may be GUIDs
    ("{...}") or names with spaces, so everything non-alphanumeric collapses to "_"."""
    safe = re.sub(r"[^0-9A-Za-z]+", "_", str(track)).strip("_") or "track"
    return f"region_{safe}_{round(start_s * 1000)}-{round(end_s * 1000)}"


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations={"destructiveHint": True})
    def render_track_region(
        track: str,
        start_s: Annotated[float, Field(ge=0)],
        end_s: Annotated[float, Field(gt=0)],
    ) -> dict:
        """Bounce ONE track's [start_s, end_s) region to a 32-bit-float WAV and return
        ``{path, start_s, end_s, sr}`` (§15.2, the bridge verb).

        The bridge does this behind a non-reentrant mutex with a FULL snapshot/restore of
        global render settings, every track's solo state, and the master-FX bypass — so a
        render never corrupts the user's project (§15.2). The target is soloed in place
        (SIP) so its sends survive; the master mix is the render source.

        Bounds are SECONDS. Beats→seconds conversion is the CALLER's duty and MUST use the
        PROJECT tempo map (e.g. ``TimeMap2_beatsToTime`` / the project QN map) — NEVER the
        reference song's grid (§15.2 step 5). This tool never accepts beats.

        Refuses while the transport is playing/recording, and raises if the render came out
        as digital silence (the tell-tale of a botched solo/routing setup). Output lands in
        ``<cache>/renders/`` (LRU-visible; see module notes).
        """
        if end_s <= start_s:
            raise ValueError(f"end_s ({end_s}) must be greater than start_s ({start_s})")
        out_dir = _renders_dir()
        basename = _basename_for(track, start_s, end_s)
        result = BridgeClient().call(
            "render_track_region",
            track=track,
            start_s=start_s,
            end_s=end_s,
            out_dir=str(out_dir),
            basename=basename,
        )
        return {
            "path": result.get("path"),
            "start_s": result.get("start_s", start_s),
            "end_s": result.get("end_s", end_s),
            "sr": result.get("sr"),
        }

    if not include_stubs:
        return  # every tool below is a stub - see the registry honesty rule

    @mcp.tool(annotations={"destructiveHint": True})
    def render_project(out_path: str, fmt: str = "wav") -> dict:
        """[NOT IMPLEMENTED]
        Headless render to WAV/MP3 (RENDER_FILE + format/bounds + Main_OnCommand(41824))."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations={"destructiveHint": True})
    def render_stems(out_dir: str) -> dict:
        """[NOT IMPLEMENTED]
        Render per-track stems by sequential solo — feeds analyze_audio_character."""
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations={"readOnlyHint": True})
    def render_and_audit() -> dict:
        """[NOT IMPLEMENTED] Render, then measure LUFS/spectrum in one call so the agent self-checks
        ('loud enough? too bright?') and iterates without a human ear."""
        raise NotImplementedError("M2 — see docs/roadmap.md")
