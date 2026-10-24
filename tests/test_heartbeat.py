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
