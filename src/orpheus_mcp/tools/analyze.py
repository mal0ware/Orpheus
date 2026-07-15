"""The UNDERSTAND brain (north-star half 1). Read-only; builds CompositionSpec(current)
plus the LLM-Readable Music Report. Logic lives in orpheus_mcp.analysis (pure functions);
this layer only pulls project data over the bridge (same wire patterns as the M1 tools)
and hands it to them."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.analysis import symbolic
from orpheus_mcp.analysis.audio import analyze_audio_character as _analyze_wav
from orpheus_mcp.analysis.fingerprint import explain_features
from orpheus_mcp.bridge import BridgeClient
from orpheus_mcp.models import (
    AudioCharacter,
    CompositionSpec,
    GrooveAnalysis,
    HarmonyAnalysis,
    MusicReport,
    Note,
    TrackSpec,
)

_RO = {"readOnlyHint": True}


def _read_tracks(bridge: BridgeClient) -> list[TrackSpec]:
    """Pull the track tree + per-track MIDI (in beats) over the bridge.

    One get_track_midi round-trip per track WITH items — tracks without items are
    skipped to respect the ~10 ops/sec file-IPC ceiling.
    """
    rows = bridge.call("list_tracks") or []
    tracks: list[TrackSpec] = []
    for row in rows:
        name = row.get("name", "")
        notes: list[Note] = []
        if row.get("num_items"):
            result = bridge.call("get_track_midi", track=row.get("guid") or str(row["index"]))
            notes = [
                Note(
                    pitch=n["pitch"],
                    start_beat=n["start_beat"],
                    duration_beats=n["duration_beats"],
                    velocity=n.get("velocity", 96),
                )
                for n in result.get("notes", [])
            ]
        tracks.append(
            TrackSpec(
                guid=row.get("guid"),
                name=name,
                role=symbolic.infer_role(name),
                volume_db=row.get("volume_db", 0.0),
                pan=row.get("pan", 0.0),
                mute=bool(row.get("mute")),
                solo=bool(row.get("solo")),
                notes=notes,
            )
        )
    return tracks


def _harmony_for(tracks: list[TrackSpec]) -> HarmonyAnalysis:
    """Key/harmony over the pooled NON-percussion notes (K-S is garbage on drums)."""
    tonal = [n for t in tracks if not symbolic.looks_percussive(t.name) for n in t.notes]
    if tonal:
        return symbolic.analyze_harmony(tonal)
    if any(t.notes for t in tracks):
        return HarmonyAnalysis(
            note="Only percussion-named tracks carry MIDI — key/harmony detection "
            "skipped rather than run on drums (where it is meaningless)."
        )
    return HarmonyAnalysis(note="No MIDI notes in the project — nothing to analyze.")


def _groove_for(tracks: list[TrackSpec]) -> GrooveAnalysis:
    """Groove from the drum tracks when they exist (that's where the pocket lives),
    otherwise from everything."""
    percussive = [n for t in tracks if symbolic.looks_percussive(t.name) for n in t.notes]
    pooled = percussive or [n for t in tracks for n in t.notes]
    return symbolic.analyze_groove(pooled)


def build_current_spec(wav_path: str | None = None) -> CompositionSpec:
    """CompositionSpec(current) over the bridge; audio measured only if a rendered WAV
    is provided (the render tools are separate — unmeasured audio stays None, not zeros).

    Module-level (not a tool) so style tools reuse the exact same snapshot logic.
    """
    bridge = BridgeClient()
    info = bridge.call("get_project_info")
    tracks = _read_tracks(bridge)
    ts = info.get("time_signature") or [4, 4]
    return CompositionSpec(
        tempo_bpm=float(info.get("tempo", 120.0)),
        time_signature=(int(ts[0]), int(ts[1])),
        harmony=_harmony_for(tracks),
        groove=_groove_for(tracks),
        audio=_analyze_wav(wav_path) if wav_path else AudioCharacter(),
        tracks=tracks,
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_RO)
    def analyze_harmony() -> HarmonyAnalysis:
        """Detect key/scale/chords/Roman numerals/cadences from the project's MIDI.

        Reads every track's MIDI over the bridge, filters drum tracks (Krumhansl
        detection is garbage on beats), and runs music21. Always returns a confidence +
        alternatives; on drum-only or non-chordal material it hedges rather than
        inventing harmony.
        """
        return _harmony_for(_read_tracks(BridgeClient()))

    @mcp.tool(annotations=_RO)
    def analyze_groove() -> GrooveAnalysis:
        """Swing, tightness, velocity dynamics, and density from note onsets in beats.
        Prefers drum tracks (that's where the pocket lives). No existing DAW-MCP server
        provides this — a genuine Orpheus differentiator."""
        return _groove_for(_read_tracks(BridgeClient()))

    @mcp.tool(annotations=_RO)
    def analyze_audio_character(wav_path: str) -> AudioCharacter:
        """Measure a rendered WAV's post-FX sonic fingerprint: integrated LUFS, 3-band
        energy, spectral centroid, sample peak, crest factor, stereo width.

        Takes a file path — render the master/stem first (render tools are separate).
        """
        return _analyze_wav(wav_path)

    @mcp.tool(annotations=_RO)
    def build_project_spec(wav_path: str | None = None) -> MusicReport:
        """Fuse symbolic (+ optionally audio) analysis into one CompositionSpec(current)
        plus a plain-English report. Pass a rendered WAV to include the audio dimensions;
        without one they stay unmeasured (None), never guessed."""
        spec = build_current_spec(wav_path)
        harmony = spec.harmony
        key = (
            f"{harmony.key_root} {harmony.mode.value}"
            if harmony.key_root and harmony.mode
            else "key undetected"
        )
        feel = spec.groove.feel or "no groove data"
        return MusicReport(
            spec=spec,
            summary=f"{spec.tempo_bpm:g} BPM, {key}; {feel}.",
            observations=explain_features(spec),
        )
