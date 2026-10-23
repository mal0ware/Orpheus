"""Bridge request ids must be unique across every BridgeClient in a process (and, in practice,
across processes), stay numeric, and keep working on the Lua side.

The bug (found 2026-10-22): every BridgeClient started its counter at 0 and every tool call
builds a new client, so two parallel read-only tool calls (FastMCP runs sync tools in a thread
pool; Claude Code runs readOnlyHint tools in parallel) both wrote request_1.json. One request
overwrote the other, and a caller could read the other caller's response.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fake_reaper import FakeReaperBridge
from orpheus_mcp.bridge.client import BridgeClient

REPO = Path(__file__).resolve().parents[1]
BRIDGE_LUA = REPO / "src" / "orpheus_mcp" / "bridge" / "lua" / "orpheus_bridge.lua"


def _slow_echo(params: dict) -> dict:
    time.sleep(0.02)  # keep several requests on disk at once, as a real REAPER tick would
    return {"n": params["n"]}


def test_parallel_clients_each_get_their_own_answer(tmp_path):
    n_threads = 16
    results: dict[int, object] = {}
    errors: list[BaseException] = []
    start = threading.Barrier(n_threads)

    def worker(n: int) -> None:
        client = BridgeClient(bridge_dir=tmp_path, timeout_s=5.0)  # a fresh client per call,
        start.wait()                                               # exactly like the tools
        try:
            results[n] = client.call("echo", n=n)
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    with FakeReaperBridge(tmp_path, {"echo": _slow_echo}):
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    assert errors == []
    assert results == {i: {"n": i} for i in range(n_threads)}


def test_request_ids_numeric_unique_and_increasing(tmp_path, monkeypatch):
    seen: list[int] = []
    real_write = BridgeClient._atomic_write

    def spy(path: Path, payload: dict) -> None:
        if path.name.startswith("request_"):
            assert re.fullmatch(r"request_\d+\.json", path.name)  # the Lua scan pattern
            assert path.name == f"request_{payload['id']}.json"
            seen.append(payload["id"])
        real_write(path, payload)

    monkeypatch.setattr(BridgeClient, "_atomic_write", staticmethod(spy))
    with FakeReaperBridge(tmp_path, {"ping": lambda p: "pong"}):
        for _ in range(3):
            client = BridgeClient(bridge_dir=tmp_path)  # new client each time
            client.call("ping")
            client.call("ping")

    assert len(seen) == 6
    assert all(isinstance(i, int) for i in seen)
    assert seen == sorted(seen) and len(set(seen)) == 6
    # Ids are exchanged as JSON numbers; stay exactly representable as a double.
    assert max(seen) < 2**53


def test_a_later_process_does_not_reuse_this_process_ids(tmp_path):
    with FakeReaperBridge(tmp_path, {"ping": lambda p: "pong"}):
        BridgeClient(bridge_dir=tmp_path).call("ping")
    from orpheus_mcp.bridge import client as client_mod

    mine = client_mod._next_request_id()
    code = "from orpheus_mcp.bridge.client import _next_request_id; print(_next_request_id())"
    theirs = int(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                check=True).stdout)
    assert theirs > mine


def test_lua_bridge_answers_a_large_id_with_an_exact_filename(tmp_path):
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    g = lua.globals()
    bridge_dir = str(tmp_path).replace("\\", "/")
    stub = lua.eval(
        "function(dir) return {"
        " GetExtState = function(a, b) if b == 'bridge_dir' then return dir end return '' end,"
        " GetAppVersion = function() return '7.0/idtest' end,"
        " RecursiveCreateDirectory = function() end, ShowConsoleMsg = function() end,"
        " defer = function() end, EnumerateFiles = function() return nil end,"
        " time_precise = function() return os.clock() end } end"
    )
    g.reaper = stub(bridge_dir)
    g.ORPHEUS_NO_AUTORUN = True
    m = lua.execute(BRIDGE_LUA.read_text(encoding="utf-8"))

    big = 1_761_234_567_890_123  # microseconds-since-epoch scale
    name = f"request_{big}.json"
    (tmp_path / name).write_text(
        json.dumps({"id": big, "fn": "get_connection_status", "params": {}}), encoding="utf-8"
    )
    m.handle_request(name)

    assert (tmp_path / f"response_{big}.json").exists()
    assert not (tmp_path / name).exists()
