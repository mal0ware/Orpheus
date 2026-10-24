"""Read-only project grounding: see current state before acting (reads before writes)."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.models import Note, ProjectChanges, ProjectTimeline, TrackSpec

_RO = {"readOnlyHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:  # noqa: ARG001 - uniform signature
    @mcp.tool(annotations=_RO)
    def get_project_info() -> dict:
        """Tempo, time signature, length, track count, and transport state."""
        info = BridgeClient().call("get_project_info")
        ts = info.get("time_signature") or [4, 4]
        return {
            "tempo_bpm": info.get("tempo"),
            "time_signature": [int(ts[0]), int(ts[1])],
            "length_seconds": info.get("length"),
            "num_tracks": info.get("num_tracks"),
            "playing": bool(info.get("playing")),
        }

    @mcp.tool(annotations=_RO)
    def list_tracks() -> list[TrackSpec]:
        """The typed track tree: name, vol/pan, mute/solo, FX-chain names, item count."""
        rows = BridgeClient().call("list_tracks") or []
        return [
            TrackSpec(
                guid=row.get("guid"),
                name=row.get("name", ""),
                volume_db=row.get("volume_db", 0.0),
                pan=row.get("pan", 0.0),
                mute=bool(row.get("mute")),
                solo=bool(row.get("solo")),
            )
            for row in rows
        ]

    @mcp.tool(annotations=_RO)
    def get_track_midi(track: str, at_bar: int = 1) -> dict:
        """Read a take's MIDI notes back in BEATS (pitch, start/duration, velocity).

        The inverse of insert_midi_notes' PPQ math, anchored so beat 0 == ``at_bar``;
        a note written at beat B reads back at beat B. All tick math stays in the bridge.
        """
        result = BridgeClient().call("get_track_midi", track=track, at_bar=at_bar)
        notes = [
            Note(
                pitch=n["pitch"],
                start_beat=n["start_beat"],
                duration_beats=n["duration_beats"],
                velocity=n.get("velocity", 96),
            )
            for n in result.get("notes", [])
        ]
        return {"track": result.get("track"), "notes": notes}

    @mcp.tool(annotations=_RO)
    def get_project_timeline() -> ProjectTimeline:
        """Where we are in the live REAPER session: the open project's name and path, every
        tempo and time-signature change, all markers and regions with bar and beat (as REAPER's
        ruler shows them), and the transport (playing, cursor, play position).

        Call this before placing anything by bar. ``project.state_change_count`` rises on every
        edit in REAPER; when it has moved since your last look, call ``project_changes``.
        """
        raw = BridgeClient().call("get_project_timeline") or {}
        return ProjectTimeline.model_validate({
            "project": raw.get("project") or {},
            "transport": raw.get("transport") or {},
            "tempo_map": raw.get("tempo_map") or [],
            "markers": raw.get("markers") or [],  # an empty Lua table arrives as {}
            "regions": raw.get("regions") or [],
            "truncated": bool(raw.get("truncated")),
        })

    @mcp.tool(annotations=_RO)
    def project_changes() -> ProjectChanges:
        """What changed in the open REAPER project since the last check, in plain words: tracks
        added, removed, renamed, moved, muted, re-levelled; items added, moved, resized, or with
        edited notes; markers, regions, tempo and meter changes. Positions come in bars.

        The first check of a project records a baseline and reports nothing. Each check then
        becomes the new baseline (kept on disk per project, so it survives restarts). Edits
        Orpheus made itself are included. Read-only for the project.
        """
        from orpheus_mcp.analysis.projectdiff import check_changes

        return check_changes(BridgeClient().call("get_project_snapshot") or {})

    # get_fx_params graduated to a real implementation in orpheus_mcp.tools.mix
    # (Task 11.1, §8 step 3) — no longer stubbed here.
