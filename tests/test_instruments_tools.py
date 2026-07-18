"""Bridge-contract + tool tests for the instrument verbs, over the real wire protocol."""
from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    p = FakeReaperProject(tempo=120.0)
    p.installed_fx = ["VSTi: ReaSynth (Cockos)", "VSTi: Surge XT (Surge Synth Team)"]
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


def test_list_installed_fx_contract(client):
    res = client.call("list_installed_fx")
    assert "Surge XT" in " ".join(res["fx"])


async def test_list_installed_fx_tool(mcp_client):
    async with mcp_client as c:
        res = await c.call_tool("list_installed_fx", {})
    assert any("ReaSynth" in name for name in res.data["fx"])
