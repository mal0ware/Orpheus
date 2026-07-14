"""Tests for the Python side of the REAPER bridge.

These run WITHOUT REAPER: `FakeReaperBridge` is a real, threaded implementation of the
exact file protocol the in-REAPER Lua loop speaks (read request_N.json → dispatch →
write response_N.json atomically → delete request; touch heartbeat.lock). It is the
executable spec for `orpheus_bridge.lua`, not a mock.
"""

from __future__ import annotations

import os
import time

import pytest

from fake_reaper import FakeReaperBridge
from orpheus_mcp.bridge.client import BridgeClient, BridgeError, BridgeTimeout

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
