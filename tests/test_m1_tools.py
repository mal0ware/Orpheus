"""M1 contract tests: track / transport / MIDI tools over the real wire protocol.

Two layers, both without REAPER:
  1. *Bridge contract* — the request/response shape every new bridge fn must honour,
     exercised through the real BridgeClient against FakeReaperBridge. This is the
     executable spec the Lua handlers must satisfy (mirrored by tests/lua/test_bridge.lua).
  2. *Tool wrappers* — the actual FastMCP tools, driven end-to-end through an in-memory
     MCP Client, so the Python layer's beats-in/typed-out behaviour is covered too.
"""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    return FakeReaperProject(tempo=120.0)


@pytest.fixture
def client(project, tmp_path):
    """A real BridgeClient pointed at a running FakeReaperBridge over the file channel."""
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path)


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    """An in-memory MCP client whose tools' default BridgeClient hits the fake bridge."""
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="OrpheusTest")
    register_tools(mcp, profile="full")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


# --------------------------------------------------------------------------- #
# Bridge contract — transport
# --------------------------------------------------------------------------- #


def test_set_tempo_contract(client, project):
    assert client.call("set_tempo", bpm=72.0) == {"tempo": 72.0}
    assert project.tempo == 72.0


def test_set_time_signature_contract(client, project):
    assert client.call("set_time_signature", numerator=3, denominator=4) == {
        "time_signature": [3, 4]
    }
    assert (project.ts_num, project.ts_den) == (3, 4)


def test_play_stop_record_contract(client):
    assert client.call("play_stop_record", command="play")["play_state"] == 1
    assert client.call("play_stop_record", command="stop")["play_state"] == 0


def test_unknown_transport_command_is_an_error(client):
    from orpheus_mcp.bridge.client import BridgeError

    with pytest.raises(BridgeError, match="unknown transport command"):
        client.call("play_stop_record", command="rewind")


# --------------------------------------------------------------------------- #
# Bridge contract — tracks + project reads
# --------------------------------------------------------------------------- #


def test_create_track_returns_stable_guid(client):
    res = client.call("create_track", name="Bass")
    assert res["name"] == "Bass"
    assert res["guid"].startswith("{")
    assert res["index"] == 1


def test_list_tracks_reflects_creation(client):
    client.call("create_track", name="Drums")
    client.call("create_track", name="Bass")
    rows = client.call("list_tracks")
    assert [r["name"] for r in rows] == ["Drums", "Bass"]
    assert all(r["guid"].startswith("{") for r in rows)


def test_get_project_info_reports_tempo_and_meter(client):
    client.call("set_tempo", bpm=90.0)
    client.call("set_time_signature", numerator=6, denominator=8)
    info = client.call("get_project_info")
    assert info["tempo"] == 90.0
    assert info["time_signature"] == [6, 8]
    assert info["num_tracks"] == 0


def test_create_track_at_index_inserts(client):
    client.call("create_track", name="A")
    client.call("create_track", name="B")
    client.call("create_track", name="Mid", index=2)
    assert [r["name"] for r in client.call("list_tracks")] == ["A", "Mid", "B"]


# --------------------------------------------------------------------------- #
# Bridge contract — MIDI
# --------------------------------------------------------------------------- #


def test_insert_midi_notes_reports_count(client):
    guid = client.call("create_track", name="Keys")["guid"]
    res = client.call(
        "insert_midi_notes",
        track=guid,
        notes=[
            {"pitch": 60, "start_beat": 0.0, "duration_beats": 1.0, "velocity": 100},
            {"pitch": 62, "start_beat": 1.0, "duration_beats": 1.0, "velocity": 100},
        ],
    )
    assert res["inserted"] == 2
    assert res["track"] == guid


def test_create_midi_item_returns_take_handle(client):
    guid = client.call("create_track", name="Keys")["guid"]
    res = client.call("create_midi_item", track=guid, start_bar=1, length_bars=2)
    assert res["track"] == guid
    assert res["item_index"] == 0


def test_insert_midi_notes_grows_item_for_later_section(client, project):
    """arrange_song writes section 2 at a higher at_bar into the SAME shared track.

    Regression for the live bug: insert_midi_notes sized the media item to the FIRST
    call's notes only; a later call at a higher at_bar wrote notes past the item's right
    edge, where in live REAPER they exist but don't play. FakeTake.item_end_qn models
    the item's right edge (D_POSITION + D_LENGTH) so this is enforceable without REAPER.
    """
    guid = client.call("create_track", name="Keys")["guid"]
    section1 = [{"pitch": 60, "start_beat": 0.0, "duration_beats": 1.0, "velocity": 100}]
    client.call("insert_midi_notes", track=guid, notes=section1, at_bar=1)

    section2 = [{"pitch": 67, "start_beat": 0.0, "duration_beats": 2.0, "velocity": 90}]
    client.call("insert_midi_notes", track=guid, notes=section2, at_bar=5)

    # (a) round-trip still holds: the bar-5 note reads back at beat 0 of bar 5.
    read_bar5 = client.call("get_track_midi", track=guid, at_bar=5)
    assert read_bar5["notes"][-1]["start_beat"] == pytest.approx(0.0)
    assert read_bar5["notes"][-1]["duration_beats"] == pytest.approx(2.0)

    # (b) the take's item now covers the bar-5 notes: item_end_qn >= project-QN end of
    # the last note written (bar 5 start + 2 beats).
    tr = project.resolve_track(guid)
    take = tr.takes[0]
    bar5_note_end_qn = project.bar_start_qn(5) + 2.0
    assert take.item_end_qn >= bar5_note_end_qn - 1e-9


def test_insert_into_unknown_track_errors(client):
    from orpheus_mcp.bridge.client import BridgeError

    with pytest.raises(BridgeError, match="no track"):
        client.call(
            "insert_midi_notes",
            track="{NOPE}",
            notes=[{"pitch": 60, "start_beat": 0.0, "duration_beats": 1.0}],
        )


# --------------------------------------------------------------------------- #
# Tool wrappers — end-to-end through an in-memory MCP client
# --------------------------------------------------------------------------- #


async def test_tool_create_track_then_list(mcp_client):
    async with mcp_client as c:
        created = await c.call_tool("create_track", {"name": "Bass"})
        guid = created.data["guid"]
        assert guid.startswith("{")
        tracks = await c.call_tool("list_tracks", {})
        # list_tracks returns typed TrackSpec models -> attribute access, not subscript.
        assert [t.name for t in tracks.data] == ["Bass"]
        assert tracks.data[0].guid == guid


async def test_tool_set_tempo_roundtrips_through_project_info(mcp_client):
    async with mcp_client as c:
        await c.call_tool("set_tempo", {"bpm": 84.0})
        info = await c.call_tool("get_project_info", {})
        assert info.data["tempo_bpm"] == 84.0


async def test_tool_insert_and_read_notes_in_beats(mcp_client):
    async with mcp_client as c:
        guid = (await c.call_tool("create_track", {"name": "Keys"})).data["guid"]
        await c.call_tool(
            "insert_midi_notes",
            {
                "track": guid,
                "notes": [
                    {"pitch": 60, "start_beat": 0.0, "duration_beats": 1.0, "velocity": 100},
                    {"pitch": 64, "start_beat": 1.5, "duration_beats": 0.5, "velocity": 80},
                ],
            },
        )
        read = await c.call_tool("get_track_midi", {"track": guid})
        notes = read.data["notes"]
        assert [(n["pitch"], n["start_beat"], n["duration_beats"]) for n in notes] == [
            (60, 0.0, 1.0),
            (64, 1.5, 0.5),
        ]


async def test_tool_insert_rejects_oversized_batch(mcp_client):
    from orpheus_mcp.bridge.client import MAX_NOTES_PER_CALL

    async with mcp_client as c:
        guid = (await c.call_tool("create_track", {"name": "Keys"})).data["guid"]
        big = [
            {"pitch": 60, "start_beat": float(i), "duration_beats": 0.5}
            for i in range(MAX_NOTES_PER_CALL + 1)
        ]
        with pytest.raises(Exception, match="at most"):
            await c.call_tool("insert_midi_notes", {"track": guid, "notes": big})


async def test_tool_transpose_moves_pitch(mcp_client):
    async with mcp_client as c:
        guid = (await c.call_tool("create_track", {"name": "Pad"})).data["guid"]
        await c.call_tool(
            "insert_midi_notes",
            {
                "track": guid,
                "notes": [{"pitch": 60, "start_beat": 0.0, "duration_beats": 1.0}],
            },
        )
        res = await c.call_tool("transpose_notes", {"track": guid, "semitones": 5})
        assert res.data["transposed"] == 1
        read = await c.call_tool("get_track_midi", {"track": guid})
        assert read.data["notes"][0]["pitch"] == 65
