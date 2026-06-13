"""THE load-bearing invariant: a note written at beat B reads back at beat B.

PPQ/tempo/take conversion is the single most correctness-sensitive code in Orpheus. This
test guards it (mirrors waveform-MCP's 250 XML round-trip tests) by writing notes through
the REAL BridgeClient + wire protocol into a behavioural fake of REAPER (FakeReaperBridge +
FakeReaperProject), reading them back, and asserting the beats survive the round-trip.
The fake's beats↔PPQ math is the executable spec for the Lua handlers; the Lua-side tests
guard that the in-REAPER code performs the same conversion.
"""

from __future__ import annotations

import pytest

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.models import Note


def _client_against(project, tmp_path) -> tuple:
    bridge = FakeReaperBridge(tmp_path, make_handlers(project))
    return bridge, BridgeClient(bridge_dir=tmp_path)


def _insert(client, track, notes, at_bar=1):
    return client.call(
        "insert_midi_notes",
        track=track,
        notes=[
            {
                "pitch": n.pitch,
                "start_beat": n.start_beat,
                "duration_beats": n.duration_beats,
                "velocity": n.velocity,
            }
            for n in notes
        ],
        at_bar=at_bar,
    )


def _read(client, track, at_bar=1) -> list[Note]:
    res = client.call("get_track_midi", track=track, at_bar=at_bar)
    return [
        Note(
            pitch=n["pitch"],
            start_beat=n["start_beat"],
            duration_beats=n["duration_beats"],
            velocity=n["velocity"],
        )
        for n in res["notes"]
    ]


def test_note_roundtrips_in_beats(tmp_path):
    project = FakeReaperProject()
    bridge, client = _client_against(project, tmp_path)
    written = [
        Note(pitch=60, start_beat=0.0, duration_beats=1.0, velocity=100),
        Note(pitch=64, start_beat=1.5, duration_beats=0.5, velocity=80),
    ]
    with bridge:
        guid = client.call("create_track", name="Keys")["guid"]
        _insert(client, guid, written)
        read_back = _read(client, guid)
    assert read_back == written


def test_fractional_beats_survive_ppq_conversion(tmp_path):
    project = FakeReaperProject()
    bridge, client = _client_against(project, tmp_path)
    # Sixteenth-note grid offsets are the classic place PPQ rounding bites.
    written = [
        Note(pitch=48, start_beat=0.25, duration_beats=0.25, velocity=96),
        Note(pitch=50, start_beat=0.75, duration_beats=0.25, velocity=96),
        Note(pitch=52, start_beat=2.5, duration_beats=1.5, velocity=96),
    ]
    with bridge:
        guid = client.call("create_track", name="Bass")["guid"]
        _insert(client, guid, written)
        read_back = _read(client, guid)
    assert read_back == written


def test_at_bar_offset_is_relative_not_absolute(tmp_path):
    # A note written at beat 0 of bar 3 must read back at beat 0 when queried with the
    # same at_bar anchor — the bar→QN offset lives in the bridge, not the model's beats.
    project = FakeReaperProject()  # 4/4 → 4 beats/bar
    bridge, client = _client_against(project, tmp_path)
    written = [Note(pitch=67, start_beat=0.0, duration_beats=2.0, velocity=90)]
    with bridge:
        guid = client.call("create_track", name="Lead")["guid"]
        _insert(client, guid, written, at_bar=3)
        same_anchor = _read(client, guid, at_bar=3)
        # Read from bar 1: the same note now sits 8 beats later (bars 1,2 = 8 beats).
        from_bar1 = _read(client, guid, at_bar=1)
    assert same_anchor == written
    assert from_bar1[0].start_beat == pytest.approx(8.0)


def test_transpose_shifts_pitch_but_preserves_timing(tmp_path):
    project = FakeReaperProject()
    bridge, client = _client_against(project, tmp_path)
    written = [
        Note(pitch=60, start_beat=0.0, duration_beats=1.0, velocity=100),
        Note(pitch=64, start_beat=1.0, duration_beats=1.0, velocity=100),
    ]
    with bridge:
        guid = client.call("create_track", name="Pad")["guid"]
        _insert(client, guid, written)
        client.call("transpose_notes", track=guid, semitones=-2)
        read_back = _read(client, guid)
    assert [n.pitch for n in read_back] == [58, 62]
    assert [n.start_beat for n in read_back] == [0.0, 1.0]
    assert [n.duration_beats for n in read_back] == [1.0, 1.0]
