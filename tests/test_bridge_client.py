"""Tests for the Python side of the REAPER bridge.

These run WITHOUT REAPER: `FakeReaperBridge` is a real, threaded implementation of the
exact file protocol the in-REAPER Lua loop speaks (read request_N.json → dispatch →
write response_N.json atomically → delete request; touch heartbeat.lock). It is the
executable spec for `orpheus_bridge.lua`, not a mock.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from orpheus_mcp.bridge.client import BridgeClient, BridgeError, BridgeTimeout


class FakeReaperBridge:
    """Stand-in for the in-REAPER Lua poll loop. Mirrors the wire protocol exactly."""

    def __init__(self, bridge_dir: Path, handlers: dict, *, beat: bool = True, answer: bool = True):
        self.dir = Path(bridge_dir)
        self.handlers = handlers
        self._beat = beat
        self._answer = answer
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> FakeReaperBridge:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        if self._beat:
            # Single heartbeat writer is the thread; just wait until it's beaten once.
            deadline = time.monotonic() + 1.0
            while not (self.dir / "heartbeat.lock").exists() and time.monotonic() < deadline:
                time.sleep(0.005)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._beat:
                self._atomic_write(self.dir / "heartbeat.lock", str(time.time()))
            if self._answer:
                for req_file in sorted(self.dir.glob("request_*.json")):
                    self._handle(req_file)
            time.sleep(0.005)

    def _handle(self, req_file: Path) -> None:
        try:
            req = json.loads(req_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return  # half-written or already gone — atomicity means we just retry next tick
        result = self._dispatch(req.get("fn"), req.get("params") or {})
        self._atomic_write(self.dir / f"response_{req['id']}.json", json.dumps(result))
        req_file.unlink(missing_ok=True)

    def _dispatch(self, fn: str, params: dict) -> dict:
        if fn == "__batch__":
            try:
                results = [self._dispatch(c["fn"], c.get("params") or {})["result"]
                           for c in params["calls"]]
                return {"ok": True, "result": results}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        handler = self.handlers.get(fn)
        if handler is None:
            return {"ok": False, "error": f"unknown fn: {fn}"}
        try:
            return {"ok": True, "result": handler(params)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Heartbeat / liveness
# --------------------------------------------------------------------------- #


def test_is_alive_false_without_heartbeat(tmp_path):
    client = BridgeClient(bridge_dir=tmp_path)
    assert client.is_alive() is False


def test_is_alive_true_with_running_bridge(tmp_path):
    with FakeReaperBridge(tmp_path, handlers={}):
        client = BridgeClient(bridge_dir=tmp_path)
        assert client.is_alive() is True


def test_is_alive_false_with_stale_heartbeat(tmp_path):
    hb = tmp_path / "heartbeat.lock"
    hb.write_text("old")
    old = time.time() - 60
    os.utime(hb, (old, old))
    client = BridgeClient(bridge_dir=tmp_path)
    assert client.is_alive() is False


# --------------------------------------------------------------------------- #
# call() round-trip
# --------------------------------------------------------------------------- #


def test_call_returns_handler_result(tmp_path):
    handlers = {"get_connection_status": lambda p: {"reaper_version": "7.0/OSX-arm64"}}
    with FakeReaperBridge(tmp_path, handlers):
        client = BridgeClient(bridge_dir=tmp_path)
        assert client.call("get_connection_status") == {"reaper_version": "7.0/OSX-arm64"}


def test_call_passes_params_through(tmp_path):
    handlers = {"echo": lambda p: p}
    with FakeReaperBridge(tmp_path, handlers):
        client = BridgeClient(bridge_dir=tmp_path)
        assert client.call("echo", track="bass", semitones=-2) == {"track": "bass", "semitones": -2}


def test_call_raises_bridge_error_when_handler_fails(tmp_path):
    def boom(_):
        raise ValueError("take has no active MIDI editor")

    with FakeReaperBridge(tmp_path, {"insert_midi_notes": boom}):
        client = BridgeClient(bridge_dir=tmp_path)
        with pytest.raises(BridgeError, match="take has no active MIDI editor"):
            client.call("insert_midi_notes")


def test_call_unknown_fn_raises_bridge_error(tmp_path):
    with FakeReaperBridge(tmp_path, handlers={}):
        client = BridgeClient(bridge_dir=tmp_path)
        with pytest.raises(BridgeError, match="unknown fn"):
            client.call("not_a_real_tool")


def test_call_times_out_when_bridge_dead(tmp_path):
    # No FakeReaperBridge running at all: no heartbeat, no answers.
    client = BridgeClient(bridge_dir=tmp_path, timeout_s=0.3)
    with pytest.raises(BridgeTimeout):
        client.call("get_connection_status")


def test_call_times_out_when_bridge_beats_but_never_answers(tmp_path):
    with FakeReaperBridge(tmp_path, handlers={}, answer=False):
        client = BridgeClient(bridge_dir=tmp_path, timeout_s=0.3)
        with pytest.raises(BridgeTimeout):
            client.call("get_connection_status")


# --------------------------------------------------------------------------- #
# Hygiene: no leftover files, ids increment
# --------------------------------------------------------------------------- #


def test_no_leftover_request_or_response_files(tmp_path):
    with FakeReaperBridge(tmp_path, {"ping": lambda p: "pong"}):
        client = BridgeClient(bridge_dir=tmp_path)
        for _ in range(3):
            assert client.call("ping") == "pong"
        leftovers = [p.name for p in tmp_path.iterdir()
                     if p.name.startswith(("request_", "response_"))]
        assert leftovers == []


def test_request_ids_are_sequential(tmp_path):
    with FakeReaperBridge(tmp_path, {"id_probe": lambda p: "ok"}):
        client = BridgeClient(bridge_dir=tmp_path)
        client.call("id_probe")
        client.call("id_probe")
    # The client must use strictly increasing ids (we assert via the private counter).
    assert client._request_id == 2


# --------------------------------------------------------------------------- #
# batch()
# --------------------------------------------------------------------------- #


def test_batch_runs_each_call_in_order(tmp_path):
    handlers = {
        "create_track": lambda p: {"guid": "{T}", "name": p["name"]},
        "set_tempo": lambda p: {"bpm": p["bpm"]},
    }
    with FakeReaperBridge(tmp_path, handlers):
        client = BridgeClient(bridge_dir=tmp_path)
        results = client.batch([
            {"fn": "create_track", "params": {"name": "Bass"}},
            {"fn": "set_tempo", "params": {"bpm": 72}},
        ])
        assert results == [{"guid": "{T}", "name": "Bass"}, {"bpm": 72}]
