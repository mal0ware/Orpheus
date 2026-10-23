"""Track / routing / FX construction primitives — the APPLY verbs (atomic, LLM-orchestrated)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

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

    @mcp.tool(annotations=_DESTRUCTIVE)
    def import_audio_files(
        files: list[str],
        at_s: Annotated[float, Field(ge=0)] = 0.0,
        name_prefix: str | None = None,
    ) -> dict:
        """Import LOCAL AUDIO FILES the user names as new tracks — the programmatic
        equivalent of drag-and-drop. Returns ``{imported, failed}``.

        Each file becomes ONE new track appended at the end (named ``name_prefix`` +
        the file's basename without extension) carrying one media item placed at
        ``at_s`` seconds, sized to the source's real length. The whole import is wrapped
        in a single undo block: one ``Ctrl+Z`` removes every track and item it created.

        COPYRIGHT BOUNDARY (§1 "owned audio only", §14.3): this tool inserts EXACTLY the
        local files the USER names — their own audio, e.g. Orpheus-separated stems of
        music they own, sitting in a folder for A/B against their mix. Orpheus never
        fetches, scrapes, discovers, or auto-embeds reference audio into a project; it
        only ever places paths the user explicitly handed it. This tool is opt-in by
        construction — nothing else in Orpheus calls it.

        ``files`` must be non-empty and every path ABSOLUTE (REAPER resolves relative
        paths against its own working directory, not the caller's). A file that cannot be
        opened as audio lands in ``failed`` with its error while every other file still
        imports — never a half-silent partial.

        Imported items are pinned to TIME base at native playrate, so a later ``set_tempo``
        never stretches them.
        """
        if not files:
            raise ValueError("files must not be empty — name at least one audio file to import")
        not_absolute = [f for f in files if not Path(f).is_absolute()]
        if not_absolute:
            raise ValueError(f"file paths must be absolute: {not_absolute}")

        params: dict = {"files": [str(f) for f in files], "at_s": float(at_s)}
        if name_prefix is not None:
            params["name_prefix"] = name_prefix
        result = BridgeClient().call("import_audio_files", **params)
        # `or []`: the bridge's minimal JSON encoder emits an EMPTY Lua table as `{}`
        # (an object), so an all-succeeded import can deliver `failed` as {} not [].
        return {
            "imported": list(result.get("imported") or []),
            "failed": list(result.get("failed") or []),
        }

    @mcp.tool(annotations=_DESTRUCTIVE)
    def clone_track_items(
        src_track: str,
        new_track_name: str,
        offset_s: float = 0.0,
        item_volume_db: float = 0.0,
    ) -> dict:
        """Clone every AUDIO item on ``src_track`` onto ONE new appended track — the
        artificial-double-tracking primitive (clone a recorded take, then detune/nudge/
        pan the clone). Returns ``{track_guid, track_name, cloned, skipped_midi, failed}``.

        Each cloned item re-opens the SAME source file the original take plays (nothing
        is copied on disk, nothing fetched — the same owned-audio boundary as
        ``import_audio_files``), lands at the original position plus ``offset_s`` seconds,
        and carries a take volume of ``item_volume_db`` dB. MIDI takes are counted under
        ``skipped_midi``, never cloned; an item whose source has no filename (in-project/
        section source) lands in ``failed`` with its reason while every other item still
        clones. The whole clone is ONE undo block, and every created item is pinned to
        TIME base at native playrate, so a later ``set_tempo`` never stretches it.
        """
        if not new_track_name:
            raise ValueError("new_track_name must not be empty")
        result = BridgeClient().call(
            "clone_track_items",
            src_track=src_track,
            new_track_name=new_track_name,
            offset_s=float(offset_s),
            item_volume_db=float(item_volume_db),
        )
        # `or []`: the bridge's minimal JSON encoder emits an EMPTY Lua table as `{}`.
        return {
            "track_guid": result.get("track_guid"),
            "track_name": result.get("track_name", new_track_name),
            "cloned": int(result.get("cloned") or 0),
            "skipped_midi": int(result.get("skipped_midi") or 0),
            "failed": list(result.get("failed") or []),
        }

    @mcp.tool(annotations=_DESTRUCTIVE)
    def set_track_params(
        track: str,
        volume_db: float | None = None,
        pan: float | None = None,
    ) -> dict:
        """Set a track's fader ``volume_db`` (dB) and/or ``pan`` (-1.0 left .. 1.0 right;
        out-of-range values are clamped bridge-side), then read BOTH back — the result
        ``{track_guid, volume_db, pan}`` reports what the track now actually holds, never
        an echo of the request. Omit either argument to leave it untouched (a call with
        neither is a pure read-back). A fader at -inf reads back as the -150.0 dB floor.
        """
        params: dict = {"track": track}
        if volume_db is not None:
            params["volume_db"] = float(volume_db)
        if pan is not None:
            params["pan"] = float(pan)
        result = BridgeClient().call("set_track_params", **params)
        return {
            "track_guid": result.get("track_guid"),
            "volume_db": result.get("volume_db"),
            "pan": result.get("pan"),
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
