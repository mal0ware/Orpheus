"""``clone_track_items`` + ``set_track_params``: the artificial-double-tracking primitives.

``clone_track_items`` clones every AUDIO item on a source track onto ONE new appended
track (the "clone a recorded take, then detune/nudge/pan the clone" verb);
``set_track_params`` sets/reads a track's fader volume (dB) and pan (-1..1, clamped).

Two layers, both without REAPER (same shape as tests/test_import_audio.py):
  1. *Bridge contract* — the real BridgeClient against FakeReaperBridge, inspecting the
     fake project to prove the clone's placement/offset, the f4394e9 tempo-immunity pins,
     the take volume, the MIDI-skip count, the per-item failure capture, and the ONE undo
     block that makes the whole clone a single Ctrl+Z.
  2. *Tool wrapper* — the actual FastMCP tools end-to-end, including argument validation.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from fastmcp import Client, FastMCP

from fake_reaper import (
    FakeMediaItem,
    FakeReaperBridge,
    FakeReaperProject,
    FakeTake,
    FakeTrack,
    make_handlers,
)
from orpheus_mcp.bridge.client import BridgeClient, BridgeError
from orpheus_mcp.registry import register_tools

SR = 48000


def _write_wav(path, seconds: float) -> str:
    n = int(round(SR * seconds))
    t = np.arange(n) / SR
    sf.write(str(path), 0.2 * np.sin(2.0 * np.pi * 220.0 * t), SR)
    return str(path)


@pytest.fixture
def take_wav(tmp_path):
    """One "recorded take" WAV of an exactly-known duration."""
    return _write_wav(tmp_path / "my_take.wav", 1.5)


@pytest.fixture
def project():
    return FakeReaperProject()


@pytest.fixture
def client(project, tmp_path):
    with FakeReaperBridge(tmp_path / "bridge", make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path / "bridge")


def _audio_track(project, name: str, source_file: str, *, position: float = 0.0,
                 length: float = 1.5, start_offs: float = 0.0) -> FakeTrack:
    """A track carrying one audio item — built directly (not via import) so a test can
    control every field, including a start offset and a bogus source path."""
    tr = FakeTrack(guid=project._next_guid(), name=name)
    tr.items.append(
        FakeMediaItem(
            source_file=source_file, position=position, length=length, take_name=name,
            start_offs=start_offs,
        )
    )
    project.tracks.append(tr)
    return tr


# --------------------------------------------------------------------------- #
# Bridge contract — clone_track_items
# --------------------------------------------------------------------------- #


def test_clone_lands_offset_pinned_and_volume_applied(client, project, take_wav):
    _audio_track(project, "take", take_wav, position=4.0, start_offs=0.75)
    res = client.call(
        "clone_track_items", src_track="take", new_track_name="take (double)",
        offset_s=0.02, item_volume_db=-6.0,
    )
    assert res["cloned"] == 1
    assert res["skipped_midi"] == 0
    assert res["failed"] == []
    assert res["track_name"] == "take (double)"

    # ONE new track, appended at the end, named as asked; the result carries its GUID.
    assert [tr.name for tr in project.tracks] == ["take", "take (double)"]
    dest = project.tracks[-1]
    assert res["track_guid"] == dest.guid

    item = dest.items[0]
    assert item.source_file == take_wav  # re-opens the SAME file — no copy on disk
    assert item.position == pytest.approx(4.02)  # original position + offset_s
    assert item.length == pytest.approx(1.5)
    assert item.start_offs == pytest.approx(0.75)  # D_STARTOFFS copied
    # f4394e9 tempo-immunity pins: a clone must never warp under a later set_tempo.
    assert item.beat_attach_mode == 0
    assert item.playrate == 1.0
    assert item.take_volume == pytest.approx(10.0 ** (-6.0 / 20.0))


def test_clone_offset_and_volume_default_to_zero(client, project, take_wav):
    _audio_track(project, "take", take_wav, position=2.0)
    client.call("clone_track_items", src_track="take", new_track_name="dbl")
    item = project.tracks[-1].items[0]
    assert item.position == pytest.approx(2.0)
    assert item.take_volume == pytest.approx(1.0)  # 0 dB


def test_clone_of_empty_track_errors_before_any_mutation(client, project):
    project.tracks.append(FakeTrack(guid=project._next_guid(), name="empty"))
    with pytest.raises(BridgeError, match="no items to clone"):
        client.call("clone_track_items", src_track="empty", new_track_name="dbl")
    assert [tr.name for tr in project.tracks] == ["empty"]  # no dest track created
    assert project.undo_events == []  # the guard fires before any undo block is opened


def test_midi_takes_are_skipped_and_counted(client, project, take_wav):
    tr = _audio_track(project, "mixed", take_wav)
    tr.takes.append(FakeTake())  # a MIDI take on the same track
    res = client.call("clone_track_items", src_track="mixed", new_track_name="dbl")
    assert res["cloned"] == 1
    assert res["skipped_midi"] == 1
    assert res["failed"] == []
    assert len(project.tracks[-1].items) == 1  # only the audio item was cloned


def test_missing_source_filename_is_recorded_in_failed(client, project, take_wav):
    tr = _audio_track(project, "take", take_wav)
    tr.items.append(FakeMediaItem(source_file="", position=0.0, length=1.0))
    res = client.call("clone_track_items", src_track="take", new_track_name="dbl")
    # The in-project source is REPORTED, never silently skipped ...
    assert len(res["failed"]) == 1
    assert "no filename" in res["failed"][0]["error"]
    # ... and the good item still cloned.
    assert res["cloned"] == 1
    assert len(project.tracks[-1].items) == 1


def test_unopenable_source_file_is_recorded_in_failed(client, project, tmp_path):
    _audio_track(project, "take", str(tmp_path / "gone.wav"))
    res = client.call("clone_track_items", src_track="take", new_track_name="dbl")
    assert res["cloned"] == 0
    assert "could not re-open source file" in res["failed"][0]["error"]


def test_whole_clone_is_one_undo_block(client, project, take_wav):
    tr = _audio_track(project, "take", take_wav)
    tr.items.append(FakeMediaItem(source_file=take_wav, position=3.0, length=1.5))
    client.call("clone_track_items", src_track="take", new_track_name="dbl")
    # Exactly ONE balanced begin/end pair wraps both items: one Ctrl+Z undoes it all.
    assert project.undo_events == [
        ("begin", "Orpheus: clone 2 items to dbl"),
        ("end", "Orpheus: clone 2 items to dbl"),
    ]


def test_undo_block_is_closed_even_when_an_item_fails(client, project, take_wav):
    tr = _audio_track(project, "take", take_wav)
    tr.items.append(FakeMediaItem(source_file="", position=0.0, length=1.0))
    client.call("clone_track_items", src_track="take", new_track_name="dbl")
    assert [kind for kind, _ in project.undo_events] == ["begin", "end"]


def test_clone_requires_a_new_track_name(client, project, take_wav):
    _audio_track(project, "take", take_wav)
    with pytest.raises(BridgeError, match="new_track_name is required"):
        client.call("clone_track_items", src_track="take", new_track_name="")
    assert project.undo_events == []


# --------------------------------------------------------------------------- #
# Bridge contract — set_track_params
# --------------------------------------------------------------------------- #


def test_set_track_params_sets_and_reads_back(client, project):
    tr = FakeTrack(guid=project._next_guid(), name="bus")
    project.tracks.append(tr)
    res = client.call("set_track_params", track="bus", volume_db=-6.0, pan=-0.4)
    assert res["track_guid"] == tr.guid
    assert res["volume_db"] == pytest.approx(-6.0)
    assert res["pan"] == pytest.approx(-0.4)
    assert tr.volume_db == pytest.approx(-6.0)
    assert tr.pan == pytest.approx(-0.4)


def test_set_track_params_leaves_omitted_fields_untouched(client, project):
    tr = FakeTrack(guid=project._next_guid(), name="bus", volume_db=-3.0, pan=0.25)
    project.tracks.append(tr)
    res = client.call("set_track_params", track="bus", pan=-1.0)
    assert res["volume_db"] == pytest.approx(-3.0)  # untouched, read back
    assert res["pan"] == pytest.approx(-1.0)
    assert tr.volume_db == pytest.approx(-3.0)


def test_set_track_params_clamps_pan(client, project):
    tr = FakeTrack(guid=project._next_guid(), name="bus")
    project.tracks.append(tr)
    assert client.call("set_track_params", track="bus", pan=2.0)["pan"] == pytest.approx(1.0)
    assert client.call("set_track_params", track="bus", pan=-9.0)["pan"] == pytest.approx(-1.0)


def test_set_track_params_floors_zero_volume_at_minus_150(client, project):
    project.tracks.append(FakeTrack(guid=project._next_guid(), name="bus"))
    # Deep enough that the linear gain underflows to 0.0 — the -inf case, floored.
    res = client.call("set_track_params", track="bus", volume_db=-7000.0)
    assert res["volume_db"] == pytest.approx(-150.0)


def test_set_track_params_unknown_track_errors(client, project):
    with pytest.raises(BridgeError, match="no track named"):
        client.call("set_track_params", track="nope", volume_db=0.0)


# --------------------------------------------------------------------------- #
# Tool wrappers — end-to-end through an in-memory MCP client
# --------------------------------------------------------------------------- #


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    bridge_dir = tmp_path / "bridge"
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", bridge_dir)
    mcp = FastMCP(name="OrpheusCloneTest")
    register_tools(mcp, profile="default")
    with FakeReaperBridge(bridge_dir, make_handlers(project)):
        yield Client(mcp)


async def test_clone_tool_end_to_end(mcp_client, project, take_wav):
    _audio_track(project, "take", take_wav, position=1.0)
    async with mcp_client as c:
        res = await c.call_tool(
            "clone_track_items",
            {"src_track": "take", "new_track_name": "take (harmony)",
             "offset_s": 0.03, "item_volume_db": -3.0},
        )
    data = res.data
    assert data["cloned"] == 1
    assert data["skipped_midi"] == 0
    assert data["failed"] == []
    dest = project.tracks[-1]
    assert dest.name == "take (harmony)"
    assert data["track_guid"] == dest.guid
    assert dest.items[0].position == pytest.approx(1.03)
    assert dest.items[0].take_volume == pytest.approx(10.0 ** (-3.0 / 20.0))
    assert dest.items[0].beat_attach_mode == 0


async def test_clone_tool_rejects_empty_track_name(mcp_client, project, take_wav):
    _audio_track(project, "take", take_wav)
    async with mcp_client as c:
        with pytest.raises(Exception, match="must not be empty"):
            await c.call_tool(
                "clone_track_items", {"src_track": "take", "new_track_name": ""}
            )
    assert len(project.tracks) == 1  # validation fired before any bridge write


async def test_set_track_params_tool_end_to_end(mcp_client, project):
    tr = FakeTrack(guid=project._next_guid(), name="bus")
    project.tracks.append(tr)
    async with mcp_client as c:
        res = await c.call_tool(
            "set_track_params", {"track": "bus", "volume_db": -4.5, "pan": 0.6}
        )
    assert res.data["volume_db"] == pytest.approx(-4.5)
    assert res.data["pan"] == pytest.approx(0.6)
    assert tr.pan == pytest.approx(0.6)
