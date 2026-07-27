"""``import_audio_files``: the opt-in bridge verb that imports LOCAL audio the user owns
as new tracks — the programmatic equivalent of drag-and-drop.

Two layers, both without REAPER (the live smoke is the controller's, not CI's):
  1. *Bridge contract* — the real BridgeClient against FakeReaperBridge, inspecting the
     fake project to prove one track per file (appended, named from the basename), the
     item's position/length, the partial-failure result, and the ONE undo block that makes
     the whole import a single Ctrl+Z.
  2. *Tool wrapper* — the actual FastMCP ``import_audio_files`` tool end-to-end, including
     its argument validation (non-empty ``files``, absolute paths).

The fake reads the REAL duration of the WAVs these tests render, so a length assertion here
is a genuine source-length assertion, not a hardcoded stand-in.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient, BridgeError
from orpheus_mcp.registry import register_tools

SR = 48000


def _write_wav(path, seconds: float) -> str:
    """A real WAV of an exact duration — the fake measures it back off disk."""
    n = int(round(SR * seconds))
    t = np.arange(n) / SR
    sf.write(str(path), 0.2 * np.sin(2.0 * np.pi * 220.0 * t), SR)
    return str(path)


@pytest.fixture
def stems(tmp_path):
    """Three "separated stems" of distinct, exactly-known durations."""
    d = tmp_path / "stems"
    d.mkdir()
    return {
        "vocals": _write_wav(d / "vocals.wav", 1.0),
        "drums": _write_wav(d / "drums.wav", 0.5),
        "bass": _write_wav(d / "bass.wav", 2.0),
    }


@pytest.fixture
def project():
    return FakeReaperProject()


@pytest.fixture
def client(project, tmp_path):
    with FakeReaperBridge(tmp_path / "bridge", make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path / "bridge")


# --------------------------------------------------------------------------- #
# Bridge contract
# --------------------------------------------------------------------------- #


def test_each_file_becomes_its_own_appended_track(client, project, stems):
    res = client.call(
        "import_audio_files", files=[stems["vocals"], stems["drums"]], at_s=0.0
    )
    assert len(res["imported"]) == 2
    assert res["failed"] == []
    # One NEW track per file, appended in the order given, named from the basename.
    assert [tr.name for tr in project.tracks] == ["vocals", "drums"]
    assert [e["track_guid"] for e in res["imported"]] == [tr.guid for tr in project.tracks]


def test_items_land_at_at_s_with_the_real_source_length(client, project, stems):
    client.call("import_audio_files", files=[stems["vocals"], stems["bass"]], at_s=6.25)
    vocals, bass = project.tracks
    assert len(vocals.items) == 1
    assert vocals.items[0].position == 6.25
    assert vocals.items[0].length == pytest.approx(1.0)
    assert bass.items[0].position == 6.25
    assert bass.items[0].length == pytest.approx(2.0)  # a DIFFERENT, real duration
    # The take carries the same name as the track, and points at the file we named.
    assert vocals.items[0].take_name == "vocals"
    assert vocals.items[0].source_file == stems["vocals"]


def test_imported_items_are_pinned_to_time_base_at_native_playrate(client, project, stems):
    # Live incident (2026-07-25): imported items inherited a beats timebase, so a later
    # set_tempo stretched the audio (half-speed). Imported reference audio must never
    # tempo-warp — assert both the time-base pin and the native playrate.
    client.call("import_audio_files", files=[stems["vocals"], stems["drums"]], at_s=0.0)
    for tr in project.tracks:
        item = tr.items[0]
        assert item.beat_attach_mode == 0, "imported item must be pinned to TIME base (0)"
        assert item.playrate == 1.0, "imported item must play back at its native rate"


def test_at_s_defaults_to_zero(client, project, stems):
    client.call("import_audio_files", files=[stems["drums"]])
    assert project.tracks[0].items[0].position == 0.0


def test_name_prefix_is_prepended(client, project, stems):
    client.call("import_audio_files", files=[stems["vocals"]], name_prefix="REF ")
    assert project.tracks[0].name == "REF vocals"
    assert project.tracks[0].items[0].take_name == "REF vocals"


def test_missing_file_lands_in_failed_and_the_rest_still_import(client, project, stems, tmp_path):
    missing = str(tmp_path / "stems" / "does-not-exist.wav")
    res = client.call(
        "import_audio_files", files=[stems["vocals"], missing, stems["drums"]]
    )
    # The bad file is REPORTED, never silently skipped ...
    assert [e["file"] for e in res["failed"]] == [missing]
    assert "could not open audio file" in res["failed"][0]["error"]
    # ... and every other file still landed, in order, with no stray track for the bad one.
    assert [e["file"] for e in res["imported"]] == [stems["vocals"], stems["drums"]]
    assert [tr.name for tr in project.tracks] == ["vocals", "drums"]


def test_non_audio_file_is_a_per_file_failure(client, project, tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not audio")
    res = client.call("import_audio_files", files=[str(junk)])
    assert res["imported"] == []
    assert res["failed"][0]["file"] == str(junk)
    assert project.tracks == []


def test_whole_import_is_one_undo_block(client, project, stems):
    client.call("import_audio_files", files=[stems["vocals"], stems["drums"], stems["bass"]])
    # Exactly ONE balanced begin/end pair wraps all three files: one Ctrl+Z undoes it all.
    assert project.undo_events == [
        ("begin", "Orpheus: import 3 audio files"),
        ("end", "Orpheus: import 3 audio files"),
    ]


def test_undo_block_is_closed_even_when_a_file_fails(client, project, stems, tmp_path):
    client.call(
        "import_audio_files", files=[stems["vocals"], str(tmp_path / "nope.wav")]
    )
    # A mid-loop failure must NOT leave the block open — still exactly one begin + one end.
    assert [kind for kind, _ in project.undo_events] == ["begin", "end"]


def test_bridge_rejects_an_empty_file_list(client, project):
    with pytest.raises(BridgeError, match="no files given"):
        client.call("import_audio_files", files=[])
    assert project.undo_events == []  # the guard fires before any undo block is opened


def test_imported_tracks_report_their_item(client, project, stems):
    client.call("import_audio_files", files=[stems["vocals"]])
    listed = client.call("list_tracks")
    assert listed[0]["name"] == "vocals"
    assert listed[0]["num_items"] == 1


# --------------------------------------------------------------------------- #
# Tool wrapper — end-to-end through an in-memory MCP client
# --------------------------------------------------------------------------- #


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    bridge_dir = tmp_path / "bridge"
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", bridge_dir)
    mcp = FastMCP(name="OrpheusImportTest")
    register_tools(mcp, profile="default")
    with FakeReaperBridge(bridge_dir, make_handlers(project)):
        yield Client(mcp)


async def test_tool_imports_multiple_files(mcp_client, project, stems):
    async with mcp_client as c:
        res = await c.call_tool(
            "import_audio_files",
            {"files": [stems["vocals"], stems["drums"]], "at_s": 2.0, "name_prefix": "A/B "},
        )
    data = res.data
    assert [e["file"] for e in data["imported"]] == [stems["vocals"], stems["drums"]]
    assert data["failed"] == []
    assert [tr.name for tr in project.tracks] == ["A/B vocals", "A/B drums"]
    assert all(tr.items[0].position == 2.0 for tr in project.tracks)
    assert data["imported"][0]["length_s"] == pytest.approx(1.0)


async def test_tool_reports_partial_failure(mcp_client, project, stems, tmp_path):
    missing = str(tmp_path / "gone.wav")
    async with mcp_client as c:
        res = await c.call_tool("import_audio_files", {"files": [missing, stems["bass"]]})
    data = res.data
    assert [e["file"] for e in data["failed"]] == [missing]
    assert [e["file"] for e in data["imported"]] == [stems["bass"]]
    assert [tr.name for tr in project.tracks] == ["bass"]


async def test_tool_rejects_an_empty_file_list(mcp_client, project):
    async with mcp_client as c:
        with pytest.raises(Exception, match="must not be empty"):
            await c.call_tool("import_audio_files", {"files": []})
    assert project.tracks == []


async def test_tool_rejects_relative_paths(mcp_client, project):
    async with mcp_client as c:
        with pytest.raises(Exception, match="absolute"):
            await c.call_tool("import_audio_files", {"files": ["stems/vocals.wav"]})
    assert project.tracks == []


async def test_tool_rejects_negative_position(mcp_client, stems):
    async with mcp_client as c:
        # Pydantic's ge=0 rejects it at the schema boundary, before the tool body runs.
        with pytest.raises(Exception, match="at_s|greater than or equal"):
            await c.call_tool(
                "import_audio_files", {"files": [stems["vocals"]], "at_s": -1.0}
            )
