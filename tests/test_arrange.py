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
