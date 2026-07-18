# tests/test_compose.py
"""Compose tools end-to-end through the in-memory MCP client + fake bridge."""
from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    p = FakeReaperProject(tempo=120.0)
    p.installed_fx = ["VSTi: ReaSynth (Cockos)"]
    return p


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="OrpheusTest")
    register_tools(mcp, profile="full")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


async def test_create_chord_progression_writes_notes(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool(
            "create_track", {"name": "keys"}
        )
        res = await c.call_tool(
            "create_chord_progression",
            {"track": "keys", "chords": "i-iv-V-i", "key": "A", "mode": "minor"},
        )
    assert res.data["chords_written"] == 4
    tr = project.resolve_track("keys")
    assert len(tr.takes[0].notes) == 12  # 4 triads * 3 notes
    assert "ReaSynth" in tr.fx  # instrument auto-loaded


async def test_create_bassline_root_style(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool("create_track", {"name": "bass"})
        res = await c.call_tool(
            "create_bassline",
            {"track": "bass", "chords": "Am, Dm, E, Am", "style": "root", "octave": 2},
        )
    tr = project.resolve_track("bass")
    assert res.data["notes_written"] == 4
    # Am root at octave 2 -> A2 = 45
    assert tr.takes[0].notes[0].pitch == 45
    assert "ReaSynth" in tr.fx


async def test_create_drum_pattern_writes_and_loads_kit(mcp_client, project):
    pattern = "kick:  x...x...x...x...\nsnare: ....x.......x...\nhat:   x.x.x.x.x.x.x.x."
    async with mcp_client as c:
        await c.call_tool("create_track", {"name": "drums"})
        res = await c.call_tool(
            "create_drum_pattern", {"track": "drums", "pattern": pattern}
        )
    tr = project.resolve_track("drums")
    assert res.data["hits_written"] == 4 + 2 + 8
    assert len(tr.fx) == 3  # three RS5k voices


async def test_humanize_is_deterministic_and_preserves_count(mcp_client, project):
    async with mcp_client as c:
        await c.call_tool("create_track", {"name": "keys"})
        await c.call_tool(
            "create_chord_progression",
            {"track": "keys", "chords": "i-iv-V-i", "key": "A"},
        )
        before = len(project.resolve_track("keys").takes[0].notes)
        r1 = await c.call_tool("humanize_pass", {"track": "keys", "seed": 42})
    after = len(project.resolve_track("keys").takes[0].notes)
    assert after == before  # replaced, not appended
    assert r1.data["humanized"] == before
