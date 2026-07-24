"""§15.2 render_track_region: the bridge verb that bounces one track's region to a WAV.

Two layers, both without REAPER (CI-safe — the live-REAPER smoke is HUMAN-deferred, Task 11):
  1. *Bridge contract* — the real BridgeClient against FakeReaperBridge, inspecting the fake
     project's state to prove the §15.2 guarantees: SIP solo on the target, master-FX bypass,
     full snapshot/restore (success AND error paths), the non-reentrant mutex, and the
     non-silent assertion.
  2. *Tool wrapper* — the actual FastMCP render_track_region tool end-to-end, proving it writes
     into <cache>/renders/ and returns {path, start_s, end_s, sr}.
"""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import (
    FakeReaperBridge,
    FakeReaperProject,
    FakeTake,
    FakeTrack,
    make_handlers,
)
from orpheus_mcp.bridge.client import BridgeClient, BridgeError
from orpheus_mcp.registry import register_tools


@pytest.fixture
def project():
    """Three tracks: track 1 has an item (renders audio), track 2 is pre-soloed in place
    (its solo must survive the render), track 3 is empty (renders silence)."""
    return FakeReaperProject(
        srate=48000,
        tracks=[
            FakeTrack(guid="{T1}", name="Drums", takes=[FakeTake()], i_solo=0),
            FakeTrack(guid="{T2}", name="Bass", takes=[FakeTake()], i_solo=2),
            FakeTrack(guid="{T3}", name="Empty", i_solo=0),
        ],
    )


@pytest.fixture
def client(project, tmp_path):
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path)


def _render(client, tmp_path, track="1", start_s=1.0, end_s=2.5, basename="reg"):
    return client.call(
        "render_track_region",
        track=track,
        start_s=start_s,
        end_s=end_s,
        out_dir=str(tmp_path),
        basename=basename,
    )


# --------------------------------------------------------------------------- #
# Bridge contract
# --------------------------------------------------------------------------- #


def test_render_returns_path_and_srate(client, project, tmp_path):
    res = _render(client, tmp_path)
    assert res["start_s"] == 1.0
    assert res["end_s"] == 2.5
    assert res["sr"] == project.srate
    assert res["path"].endswith("reg.wav")
    assert (tmp_path / "reg.wav").is_file()


def test_render_refuses_while_transport_playing(client, project, tmp_path):
    project.play_state = 1
    with pytest.raises(BridgeError, match="playing"):
        _render(client, tmp_path)


def test_render_solo_state_is_snapshotted_and_restored(client, project, tmp_path):
    # Track 2 is soloed-in-place before the render; track 1 is the target.
    _render(client, tmp_path, track="1")
    t1, t2, t3 = project.tracks
    # SIP (=2) was active on the TARGET during the render, and cleared everywhere else.
    assert project.render_observed_target_solo == 2
    # After the render every track's solo is exactly what it was before.
    assert t1.i_solo == 0
    assert t2.i_solo == 2  # the user's pre-existing SIP solo survived
    assert t3.i_solo == 0


def test_render_master_fx_bypassed_during_then_restored(client, project, tmp_path):
    assert project.master_fxen == 1
    _render(client, tmp_path)
    # Bypass (I_FXEN=0) was in effect while the render ran ...
    assert project.render_observed_master_fxen == 0
    # ... and the master FX chain is re-enabled afterwards.
    assert project.master_fxen == 1


def test_render_settings_restored_after_render(client, project, tmp_path):
    # Seed sentinel render settings and prove none of them leak out of a render.
    project.render_settings = {"RENDER_FILE": "/before", "RENDER_SETTINGS": 3}
    before = dict(project.render_settings)
    _render(client, tmp_path)
    assert project.render_settings == before


def test_second_concurrent_render_errors_busy(client, project, tmp_path):
    # Simulate a render already in flight (the bridge-side non-reentrant mutex is held).
    project.render_mutex_held = True
    with pytest.raises(BridgeError, match="render busy"):
        _render(client, tmp_path)
    # The in-flight render's mutex must NOT have been cleared by the refused call.
    assert project.render_mutex_held is True


def test_silent_render_raises_and_still_restores(client, project, tmp_path):
    # Track 3 has no items -> the render is digital silence -> the non-silent assertion fires.
    with pytest.raises(BridgeError, match="silent"):
        _render(client, tmp_path, track="3", basename="silent")
    # Even though the render failed mid-flight, all global state was restored (§15.2).
    t1, t2, t3 = project.tracks
    assert (t1.i_solo, t2.i_solo, t3.i_solo) == (0, 2, 0)
    assert project.master_fxen == 1
    assert project.render_mutex_held is False


def test_mutex_released_after_successful_render(client, project, tmp_path):
    _render(client, tmp_path)
    assert project.render_mutex_held is False
    # A follow-up render still works (mutex was not left stuck held).
    res = _render(client, tmp_path, basename="again")
    assert res["path"].endswith("again.wav")


# --------------------------------------------------------------------------- #
# Tool wrapper — end-to-end through an in-memory MCP client
# --------------------------------------------------------------------------- #


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    """The real render_track_region tool, whose BridgeClient hits the fake bridge and whose
    render output is redirected under a temp cache root via ORPHEUS_CACHE_DIR."""
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("ORPHEUS_CACHE_DIR", str(cache_root))
    mcp = FastMCP(name="OrpheusRenderTest")
    register_tools(mcp, profile="full")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp), cache_root


async def test_tool_renders_into_cache_renders_dir(mcp_client):
    client, cache_root = mcp_client
    async with client as c:
        res = await c.call_tool(
            "render_track_region", {"track": "1", "start_s": 1.0, "end_s": 3.0}
        )
        data = res.data
        assert data["sr"] == 48000
        assert data["start_s"] == 1.0
        assert data["end_s"] == 3.0
        # Output lands under <cache>/renders/ and the file really exists.
        path = data["path"]
        assert (cache_root / "renders").as_posix() in path.replace("\\", "/")
        from pathlib import Path

        assert Path(path).is_file()


async def test_tool_rejects_nonpositive_region(mcp_client):
    client, _ = mcp_client
    async with client as c:
        with pytest.raises(Exception, match="greater than|end_s"):
            await c.call_tool(
                "render_track_region", {"track": "1", "start_s": 2.0, "end_s": 2.0}
            )
