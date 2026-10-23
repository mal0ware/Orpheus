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


def test_add_drumkit_is_idempotent(client, project):
    project.tracks.append(_make_track(project, "drums"))
    samples = {"kick": "/x/kick.wav", "snare": "/x/snare.wav", "hat": "/x/hat.wav"}
    first = client.call("add_instrument", track="drums", kind="drumkit", samples=samples)
    assert first["already_present"] is False
    second = client.call("add_instrument", track="drums", kind="drumkit", samples=samples)
    assert second["already_present"] is True
    tr = project.resolve_track("drums")
    assert len(tr.fx) == 3


async def test_install_sound_pack_requires_confirm(mcp_client):
    async with mcp_client as c:
        res = await c.call_tool("install_sound_pack", {})
    assert res.data["installed"] is False
    assert "confirm=True" in res.data["note"]


async def test_install_sound_pack_installs_with_confirm(mcp_client, monkeypatch):
    calls = []

    def fake_install(root=None, fetch=None, manifest=None):
        calls.append(root)
        return {"installed": True, "instruments": ["guitar", "bass", "piano"], "dir": "x",
                "downloaded_files": 106, "already_present": 0}

    monkeypatch.setattr("orpheus_mcp.soundpack.install_sound_pack", fake_install)
    async with mcp_client as c:
        res = await c.call_tool("install_sound_pack", {"confirm": True})
    assert res.data["installed"] is True and calls == [None]  # the default pack folder
    assert res.data["instruments"] == ["guitar", "bass", "piano"]


def _install_piano_sfz(pack_root):
    import shutil
    from pathlib import Path

    fix = Path(__file__).resolve().parent / "fixtures" / "sfz" / "UprightPianoKW-20220221.sfz"
    (pack_root / "piano").mkdir(parents=True, exist_ok=True)
    shutil.copy(fix, pack_root / "piano" / fix.name)


async def test_load_sound_pack_instrument_sends_zones_in_batches(mcp_client, project, monkeypatch,
                                                                 tmp_path):
    monkeypatch.setenv("ORPHEUS_SOUNDPACK_DIR", str(tmp_path / "pack"))
    _install_piano_sfz(tmp_path / "pack")
    project.tracks.append(_make_track(project, "keys"))
    async with mcp_client as c:
        res = await c.call_tool(
            "load_sound_pack_instrument", {"track": "keys", "instrument": "piano"}
        )
    assert res.data["zones_added"] == 36 and res.data["failed"] == []  # loud layer: 32 + 4
    zones = [fx for fx in project.resolve_track("keys").fx if fx.startswith("RS5K:")]
    assert len(zones) == 36


def test_composers_use_the_pack_when_installed(client, project, monkeypatch, tmp_path):
    from orpheus_mcp import soundpack
    from orpheus_mcp.instruments import choose_instrument, load_instrument

    project.installed_fx = ["VSTi: ReaSynth (Cockos)"]  # nothing on the allowlist
    monkeypatch.setenv("ORPHEUS_SOUNDPACK_DIR", str(tmp_path / "pack"))
    piano = next(i for i in soundpack.load_manifest()["instruments"] if i["id"] == "piano")
    monkeypatch.setattr(
        soundpack, "installed_instruments", lambda root=None, manifest=None: [piano]
    )
    _install_piano_sfz(tmp_path / "pack")
    project.tracks.append(_make_track(project, "chords"))

    choice = choose_instrument("keys", project.installed_fx)
    assert choice == {"kind": "pack", "name": "piano", "source": "pack"}
    res = load_instrument(client, "chords", choice)
    assert res["zones_added"] == 36


def test_composers_fall_back_to_stock_without_a_pack(project):
    from orpheus_mcp.instruments import choose_instrument

    assert choose_instrument("keys", ["VSTi: ReaSynth (Cockos)"])["source"] == "stock"


def test_clear_track_midi_empties_take(client, project):
    from fake_reaper import FakeNote, FakeTake, FakeTrack

    tr = FakeTrack(guid=project._next_guid(), name="keys")
    tr.takes.append(FakeTake(notes=[FakeNote(pitch=60, start_ppq=0, end_ppq=480)]))
    project.tracks.append(tr)
    res = client.call("clear_track_midi", track="keys")
    assert res["cleared"] == 1
    assert project.resolve_track("keys").takes[0].notes == []
