"""Cross-language integration: the REAL Python BridgeClient talking to the REAL Lua bridge
running as a subprocess. Proves the two implementations actually interoperate over the wire
(the strongest guard against subtle mismatches like number/id formatting).

Skipped automatically when no `lua` interpreter is installed (e.g. CI without lua)."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from orpheus_mcp.bridge.client import BridgeClient

LUA = shutil.which("lua") or shutil.which("lua5.4") or shutil.which("luajit")
DRIVER = Path(__file__).parent / "lua" / "run_bridge.lua"

pytestmark = pytest.mark.skipif(LUA is None, reason="no lua interpreter installed")


def _wait_alive(client: BridgeClient, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.is_alive():
            return True
        time.sleep(0.05)
    return False


def test_python_client_round_trips_through_real_lua_bridge(tmp_path):
    proc = subprocess.Popen([LUA, str(DRIVER), str(tmp_path)])
    try:
        client = BridgeClient(bridge_dir=tmp_path, timeout_s=3.0)
        assert _wait_alive(client), "lua bridge never produced a heartbeat"

        result = client.call("get_connection_status")
        assert result["reaper_version"] == "7.0/integration"

        batched = client.batch([
            {"fn": "get_connection_status"},
            {"fn": "get_connection_status"},
        ])
        assert len(batched) == 2
        assert batched[0]["reaper_version"] == "7.0/integration"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_lua_m1_handler_suite_passes():
    """Run the self-contained Lua-side M1 handler suite through the real interpreter.

    Verifies the in-REAPER handlers (beats↔PPQ round-trip, transport, track ops) against
    a stubbed `reaper`, so the Lua half of the M1 contract is enforced wherever a lua
    interpreter exists (and auto-skipped where it doesn't, like the cross-language test)."""
    suite = Path(__file__).parent / "lua" / "test_m1_handlers.lua"
    result = subprocess.run([LUA, str(suite)], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Lua M1 handler suite failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "0 failed" in result.stdout
