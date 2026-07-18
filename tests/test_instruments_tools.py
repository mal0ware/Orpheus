"""Bridge-contract + tool tests for the instrument verbs, over the real wire protocol."""
from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, FakeTrack, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.registry import register_tools


def _make_track(project, name):
    return FakeTrack(guid=project._next_guid(), name=name)


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


def test_add_synth_contract(client, project):
    project.tracks.append(_make_track(project, "keys"))
    res = client.call("add_instrument", track="keys", kind="named", name="ReaSynth")
    assert res["loaded"] == "ReaSynth"
    assert res["already_present"] is False


def test_add_synth_is_idempotent(client, project):
    project.tracks.append(_make_track(project, "keys"))
    client.call("add_instrument", track="keys", kind="named", name="ReaSynth")
    res = client.call("add_instrument", track="keys", kind="named", name="ReaSynth")
    assert res["already_present"] is True


def test_add_drumkit_loads_three_samplers(client, project):
    project.tracks.append(_make_track(project, "drums"))
    res = client.call(
        "add_instrument", track="drums", kind="drumkit",
        samples={"kick": "/x/kick.wav", "snare": "/x/snare.wav", "hat": "/x/hat.wav"},
    )
    assert res["loaded"] == "drumkit"
    tr = project.resolve_track("drums")
    assert len(tr.fx) == 3


def test_clear_track_midi_empties_take(client, project):
    from fake_reaper import FakeNote, FakeTake, FakeTrack

    tr = FakeTrack(guid=project._next_guid(), name="keys")
    tr.takes.append(FakeTake(notes=[FakeNote(pitch=60, start_ppq=0, end_ppq=480)]))
    project.tracks.append(tr)
    res = client.call("clear_track_midi", track="keys")
    assert res["cleared"] == 1
    assert project.resolve_track("keys").takes[0].notes == []
