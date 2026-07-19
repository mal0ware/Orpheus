"""Contract + tool tests for the arrange module, over the real wire protocol."""
from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    p = FakeReaperProject(tempo=120.0)
    p.installed_fx = ["VSTi: ReaSynth (Cockos)"]
    return p


@pytest.fixture
def client(project, tmp_path):
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path)


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="OrpheusTest")
    register_tools(mcp, profile="full")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


def test_add_marker_contract(client, project):
    res = client.call("add_marker", name="Verse 1", bar=5)
    assert res["name"] == "Verse 1"
    assert res["bar"] == 5
    assert project.markers == [{"name": "Verse 1", "bar": 5}]


async def test_add_marker_tool(mcp_client, project):
    async with mcp_client as c:
        res = await c.call_tool("add_marker", {"name": "Chorus", "bar": 9})
    assert res.data["bar"] == 9
    assert project.markers[-1]["name"] == "Chorus"


async def test_build_section_lays_all_parts_at_offset(mcp_client, project):
    async with mcp_client as c:
        res = await c.call_tool(
            "build_section",
            {"key": "A", "mode": "minor", "progression": "i-iv-V-i",
             "bars": 4, "at_bar": 1, "drums": "backbeat", "melody": "A4:q C5:q E5:h"},
        )
    names = {t.name for t in project.tracks}
    assert {"drums", "chords", "bass", "lead"}.issubset(names)
    assert len(project.resolve_track("drums").fx) == 3  # stock kit
    assert project.resolve_track("chords").takes[0].notes  # chords written
    assert project.resolve_track("lead").takes[0].notes    # melody written
    assert res.data["at_bar"] == 1 and res.data["bars"] == 4


async def test_build_section_second_section_offsets(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool("build_section",
            {"key": "A", "mode": "minor", "progression": "i-iv", "bars": 2, "at_bar": 1})
        await c.call_tool("build_section",
            {"key": "A", "mode": "minor", "progression": "VI-VII", "bars": 2, "at_bar": 3})
    # chords track holds notes from both sections (bar 1 and bar 3 => beat >= 8)
    starts = [n.start_ppq for n in project.resolve_track("chords").takes[0].notes]
    assert max(starts) >= 8 * 960  # a note at/after bar 3 (8 beats in)


async def test_arrange_song_places_sections_and_markers(mcp_client, project):
    sections = [
        {"name": "Verse", "bars": 2, "progression": "i-iv"},
        {"name": "Chorus", "bars": 2, "progression": "VI-VII", "melody": "A4:q C5:q E5:h"},
    ]
    async with mcp_client as c:
        res = await c.call_tool(
            "arrange_song",
            {"tempo": 72, "key": "A", "mode": "minor", "sections": sections},
        )
    assert 71 <= project.tempo <= 73
    assert [m["name"] for m in project.markers] == ["Verse", "Chorus"]
    assert [m["bar"] for m in project.markers] == [1, 3]  # Verse@1, Chorus after 2 bars
    assert res.data["sections"] == [
        {"name": "Verse", "at_bar": 1, "bars": 2},
        {"name": "Chorus", "at_bar": 3, "bars": 2},
    ]
    assert len(project.resolve_track("drums").fx) == 3


async def test_place_lyric_markers(mcp_client, project):
    async with mcp_client as c:
        res = await c.call_tool(
            "place_lyric_markers",
            {"lines": ["Verse: first line here", "Chorus: hook line here"],
             "at_bars": [1, 9]},
        )
    assert res.data["placed"] == 2
    assert [m["bar"] for m in project.markers] == [1, 9]
    assert "first line" in project.markers[0]["name"]
