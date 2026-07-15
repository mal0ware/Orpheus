"""M2 tool wrappers: the ANALYZE brain end-to-end through an in-memory MCP client,
against the FakeReaperBridge (the executable spec for the Lua side). No REAPER.

The bridge-facing tools must follow the same wire patterns the M1 tools use
(list_tracks / get_track_midi through BridgeClient); the pure-DSP tool
(analyze_audio_character) takes a rendered WAV path — render tools stay out of scope.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from fake_reaper import (
    FakeNote,
    FakeReaperBridge,
    FakeReaperProject,
    FakeTake,
    FakeTrack,
    make_handlers,
)
from orpheus_mcp.registry import register_tools

PPQ = 960


def _add_track(project: FakeReaperProject, name: str, notes_beats: list[tuple]) -> FakeTrack:
    """Seed a fake track with (pitch, start_beat, duration_beats, velocity) tuples."""
    track = FakeTrack(guid=project._next_guid(), name=name)
    take = FakeTake()
    for pitch, start, duration, velocity in notes_beats:
        take.notes.append(
            FakeNote(
                pitch=pitch,
                start_ppq=round(start * PPQ),
                end_ppq=round((start + duration) * PPQ),
                velocity=velocity,
            )
        )
    track.takes.append(take)
    project.tracks.append(track)
    return track


def _c_major_chords() -> list[tuple]:
    notes = []
    for i, chord in enumerate([[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]]):
        notes += [(p, i * 4.0, 4.0, 96) for p in chord]
    return notes


def _swung_hats() -> list[tuple]:
    notes = []
    for beat in range(8):
        notes += [(42, float(beat), 0.1, 100), (42, beat + 2.0 / 3.0, 0.1, 60)]
    return notes


@pytest.fixture
def project():
    return FakeReaperProject(tempo=75.0)  # inside classical's 60-90 fingerprint range


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="OrpheusM2Test")
    register_tools(mcp, profile="explain")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


# --------------------------------------------------------------------------- #
# analyze_harmony
# --------------------------------------------------------------------------- #


async def test_analyze_harmony_reads_key_and_numerals_from_project(project, mcp_client):
    _add_track(project, "Piano", _c_major_chords())
    # Chromatic junk on a drum-named track must be filtered OUT of key detection.
    _add_track(project, "Drums", [(37, 0.0, 0.1, 100), (38, 0.5, 0.1, 100), (39, 1.0, 0.1, 100)])
    async with mcp_client as c:
        result = (await c.call_tool("analyze_harmony", {})).data
    assert result.key_root == "C"
    assert result.mode == "major"
    assert result.roman_numerals == ["I", "IV", "V", "I"]
    assert result.key_confidence is not None


async def test_analyze_harmony_drums_only_hedges(project, mcp_client):
    _add_track(project, "Drums", [(36, 0.0, 0.1, 100), (38, 1.0, 0.1, 100)])
    async with mcp_client as c:
        result = (await c.call_tool("analyze_harmony", {})).data
    assert result.key_root is None
    assert "percussion" in result.note.lower() or "drum" in result.note.lower()


async def test_analyze_harmony_empty_project_hedges(mcp_client):
    async with mcp_client as c:
        result = (await c.call_tool("analyze_harmony", {})).data
    assert result.key_root is None
    assert result.note  # says why, never silently guesses


# --------------------------------------------------------------------------- #
# analyze_groove
# --------------------------------------------------------------------------- #


async def test_analyze_groove_measures_swing_from_drum_track(project, mcp_client):
    _add_track(project, "Hats", _swung_hats())
    async with mcp_client as c:
        result = (await c.call_tool("analyze_groove", {})).data
    assert result.swing_pct is not None and result.swing_pct > 0.9
    assert "swing" in result.feel or "swung" in result.feel


async def test_analyze_groove_empty_project_returns_empty_analysis(mcp_client):
    async with mcp_client as c:
        result = (await c.call_tool("analyze_groove", {})).data
    assert result.swing_pct is None
    assert result.tightness is None


# --------------------------------------------------------------------------- #
# analyze_audio_character (pure DSP on a rendered file path)
# --------------------------------------------------------------------------- #


async def test_analyze_audio_character_measures_wav(mcp_client, tmp_path):
    sr = 48000
    t = np.arange(sr * 2) / sr
    wav = tmp_path / "render.wav"
    sf.write(str(wav), 0.5 * np.sin(2 * np.pi * 1000 * t), sr)
    async with mcp_client as c:
        result = (await c.call_tool("analyze_audio_character", {"wav_path": str(wav)})).data
    assert result.lufs_integrated is not None
    assert result.spectral_centroid_hz == pytest.approx(1000.0, abs=50.0)


async def test_analyze_audio_character_missing_file_errors(mcp_client, tmp_path):
    async with mcp_client as c:
        with pytest.raises(ToolError, match="No such audio file"):
            await c.call_tool(
                "analyze_audio_character", {"wav_path": str(tmp_path / "missing.wav")}
            )


# --------------------------------------------------------------------------- #
# build_project_spec
# --------------------------------------------------------------------------- #


async def test_build_project_spec_fuses_everything(project, mcp_client):
    _add_track(project, "Piano", _c_major_chords())
    _add_track(project, "Hats", _swung_hats())
    async with mcp_client as c:
        report = (await c.call_tool("build_project_spec", {})).data
    assert report.spec.tempo_bpm == 75.0
    assert report.spec.harmony.key_root == "C"
    assert report.spec.groove.swing_pct is not None
    assert len(report.spec.tracks) == 2
    assert report.summary
    assert report.observations  # explain_features output rides along
    # Audio is NOT measured here (no render) — the spec must say nothing, not zeros.
    assert report.spec.audio.lufs_integrated is None


# --------------------------------------------------------------------------- #
# style tools
# --------------------------------------------------------------------------- #


async def test_list_style_fingerprints(mcp_client):
    async with mcp_client as c:
        result = (await c.call_tool("list_style_fingerprints", {})).data
    assert "classical" in result


async def test_explain_style_produces_measured_deltas(project, mcp_client):
    _add_track(project, "Piano", _c_major_chords())
    async with mcp_client as c:
        result = (await c.call_tool("explain_style", {"style": "classical"})).data
    assert result.style == "classical"
    assert result.deltas
    tempo_delta = next(d for d in result.deltas if d.feature == "tempo")
    assert tempo_delta.matches  # 75 BPM inside 60-90
    assert "75" in tempo_delta.explanation
    # No audio was rendered — the audio dimensions must be caveats, not claims.
    assert any("not measured" in c_ for c_ in result.caveats)


async def test_explain_style_with_rendered_wav_adds_audio_evidence(
    project, mcp_client, tmp_path
):
    _add_track(project, "Piano", _c_major_chords())
    sr = 48000
    t = np.arange(sr * 2) / sr
    wav = tmp_path / "master.wav"
    sf.write(str(wav), 0.1 * np.sin(2 * np.pi * 800 * t), sr)
    async with mcp_client as c:
        result = (
            await c.call_tool("explain_style", {"style": "classical", "wav_path": str(wav)})
        ).data
    assert any(d.feature == "loudness" for d in result.deltas)


async def test_explain_style_unknown_style_errors(mcp_client):
    async with mcp_client as c:
        with pytest.raises(ToolError, match="No fingerprint"):
            await c.call_tool("explain_style", {"style": "polka-core"})
