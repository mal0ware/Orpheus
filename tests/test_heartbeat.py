"""Heartbeat v2 (spec 2026-10-23 REAPER sync §2.4): the file's content carries the open
project's identity; liveness is still its modification time; the old bare-integer form still
reads."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import read_heartbeat
from orpheus_mcp.registry import register_tools

V2 = {"v": 2, "t": 1792851035, "reaper_version": "7.78/win64", "project_name": "song.rpp",
      "project_path": "C:\\proj\\song.rpp", "playing": False, "state_change_count": 42}


def test_reads_v2_json(tmp_path):
    (tmp_path / "heartbeat.lock").write_text(json.dumps(V2), encoding="utf-8")
    assert read_heartbeat(tmp_path) == V2


@pytest.mark.parametrize("content", ["1792851035", "1792851035.25"])
def test_reads_the_old_bare_timestamp(tmp_path, content):
    (tmp_path / "heartbeat.lock").write_text(content, encoding="utf-8")
    assert read_heartbeat(tmp_path) == {"v": 1, "t": float(content)}


@pytest.mark.parametrize("content", ["", "{not json", "[1, 2]"])
def test_unreadable_content_is_none(tmp_path, content):
    (tmp_path / "heartbeat.lock").write_text(content, encoding="utf-8")
    assert read_heartbeat(tmp_path) is None


def test_missing_file_is_none(tmp_path):
    assert read_heartbeat(tmp_path) is None


async def test_connection_status_reports_the_open_project(tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="hb")
    register_tools(mcp, profile="default")
    with FakeReaperBridge(tmp_path, make_handlers(FakeReaperProject())) as fake:
        fake.heartbeat_content = json.dumps(V2)
        fake.wait_for_heartbeat_content()
        async with Client(mcp) as c:
            res = (await c.call_tool("get_connection_status", {})).structured_content
    assert res["connected"] is True
    assert res["project"] == {"name": "song.rpp", "path": "C:\\proj\\song.rpp",
                              "playing": False, "state_change_count": 42}


async def test_connection_status_without_v2_has_no_project(tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="hb1")
    register_tools(mcp, profile="default")
    with FakeReaperBridge(tmp_path, make_handlers(FakeReaperProject())):
        async with Client(mcp) as c:
            res = (await c.call_tool("get_connection_status", {})).structured_content
    assert res["connected"] is True and "project" not in res


def _flaky(monkeypatch, method: str, exc: type[OSError], times: int):
    """Make Path.<method> on heartbeat.lock fail ``times`` times, as it does on Windows while
    the bridge swaps the file (PermissionError mid-replace; FileNotFoundError between the Lua
    writer's remove and rename)."""
    from pathlib import Path

    real, left = getattr(Path, method), [times]

    def patched(self, *a, **kw):
        if self.name == "heartbeat.lock" and left[0] > 0:
            left[0] -= 1
            raise exc("heartbeat.lock is being replaced")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, method, patched)


@pytest.mark.parametrize("exc", [PermissionError, FileNotFoundError])
def test_reading_retries_while_the_bridge_swaps_the_file(tmp_path, monkeypatch, exc):
    (tmp_path / "heartbeat.lock").write_text(json.dumps(V2), encoding="utf-8")
    _flaky(monkeypatch, "read_text", exc, times=2)
    assert read_heartbeat(tmp_path) == V2


@pytest.mark.parametrize("exc", [PermissionError, FileNotFoundError])
def test_liveness_retries_while_the_bridge_swaps_the_file(tmp_path, monkeypatch, exc):
    from orpheus_mcp.bridge.client import BridgeClient

    (tmp_path / "heartbeat.lock").write_text("1", encoding="utf-8")
    _flaky(monkeypatch, "stat", exc, times=2)
    assert BridgeClient(bridge_dir=tmp_path).is_alive() is True


def test_a_heartbeat_that_stays_missing_is_not_alive(tmp_path):
    from orpheus_mcp.bridge.client import BridgeClient

    assert BridgeClient(bridge_dir=tmp_path).is_alive() is False
