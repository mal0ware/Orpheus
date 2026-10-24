"""get_project_timeline and project_changes end to end: in-memory MCP client -> real
BridgeClient -> FakeReaperBridge (spec 2026-10-23 REAPER sync §2.1, §2.3)."""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    p = FakeReaperProject(tempo=120.0, name="song.rpp", path="C:/proj/song.rpp")
    p.meter_changes = [(5, 3, 4)]
    return p


@pytest.fixture
def bridge(project, tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path / "cache"))  # snapshots land here
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path / "bridge")
    with FakeReaperBridge(tmp_path / "bridge", make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path / "bridge")


def _client() -> Client:
    mcp = FastMCP(name="OrpheusSyncTest")
    register_tools(mcp, profile="default")
    return Client(mcp)


async def _call(name: str, args: dict | None = None):
    async with _client() as c:
        return (await c.call_tool(name, args or {})).structured_content


async def test_timeline_reports_identity_meter_changes_markers_and_regions(bridge, project):
    bridge.call("add_marker", name="Bridge", bar=6)
    bridge.call("add_region", name="Chorus", start_s=8.0, end_s=14.0)
    project.play_state = 1
    tl = await _call("get_project_timeline")

    assert tl["project"]["name"] == "song.rpp"
    assert tl["project"]["path"] == "C:/proj/song.rpp"
    assert tl["transport"]["playing"] is True
    assert [(p["bar"], p["meter"]) for p in tl["tempo_map"]] == [(1, [4, 4]), (5, [3, 4])]
    assert tl["markers"] == [{"index": 1, "name": "Bridge", "time_s": 9.5, "bar": 6,
                              "beat": 1.0, "color": None}]
    region = tl["regions"][0]
    assert (region["name"], region["start_bar"], region["end_bar"]) == ("Chorus", 5, 9)


async def test_timeline_with_no_markers_returns_empty_lists(bridge):
    tl = await _call("get_project_timeline")
    assert tl["markers"] == [] and tl["regions"] == []


async def test_project_changes_baseline_then_edits_then_quiet(bridge, project):
    first = await _call("project_changes")
    assert first["baseline_recorded"] is True and first["changes"] == []

    guid = bridge.call("create_track", name="Keys")["guid"]
    bridge.call("insert_midi_notes", track=guid, at_bar=6,
                notes=[{"pitch": 60, "start_beat": 0.0, "duration_beats": 2.0}])
    bridge.call("add_marker", name="Bridge", bar=6)
    second = await _call("project_changes")
    kinds = [c["kind"] for c in second["changes"]]
    assert kinds == ["track_added", "item_added", "marker_added", "project_length_changed"]
    assert second["changes"][1]["summary"] == "New midi item on 'Keys' at bar 6."

    quiet = await _call("project_changes")
    assert quiet["changes"] == [] and "has not moved" in quiet["note"]

    bridge.call("insert_midi_notes", track=guid, at_bar=6,
                notes=[{"pitch": 64, "start_beat": 1.0, "duration_beats": 1.0}])
    third = await _call("project_changes")
    assert [c["kind"] for c in third["changes"]] == ["item_notes_changed"]


async def test_project_changes_baseline_survives_a_server_restart(bridge, project):
    await _call("project_changes")  # baseline, written to disk
    bridge.call("create_track", name="Pad")
    # A new FastMCP instance + client stands in for a fresh Claude session.
    again = await _call("project_changes")
    assert again["baseline_recorded"] is False
    assert [c["kind"] for c in again["changes"]] == ["track_added"]


async def test_each_project_has_its_own_baseline(bridge, project):
    await _call("project_changes")
    project.path = "C:/proj/other.rpp"
    other = await _call("project_changes")
    assert other["baseline_recorded"] is True
