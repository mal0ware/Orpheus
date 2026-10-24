"""Bars follow REAPER's tempo/meter map (spec 2026-10-23 REAPER sync §2.2), Python half.

The fake mirrors the Lua bridge: with bars 1-4 in 4/4 and bar 5 on in 3/4, bar 6 starts at
quarter note 4*4 + 3 = 19. The old single-meter formula said 20.
"""

from __future__ import annotations

import pytest

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient


@pytest.fixture
def project():
    p = FakeReaperProject()
    p.meter_changes = [(5, 3, 4)]
    return p


@pytest.fixture
def client(project, tmp_path):
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path)


def test_bar_starts_follow_meter_changes(project):
    assert [project.bar_start_qn(b) for b in (1, 4, 5, 6, 7)] == [0.0, 12.0, 16.0, 19.0, 22.0]


def test_notes_at_a_bar_after_a_meter_change(client, project):
    guid = client.call("create_track", name="keys")["guid"]
    client.call("insert_midi_notes", track=guid, at_bar=6,
                notes=[{"pitch": 64, "start_beat": 0.0, "duration_beats": 1.0}])
    assert client.call("get_track_midi", track=guid, at_bar=1)["notes"][0]["start_beat"] == 19.0
    assert client.call("get_track_midi", track=guid, at_bar=6)["notes"][0]["start_beat"] == 0.0


def test_midi_item_length_counts_each_bar_in_its_own_meter(client, project):
    guid = client.call("create_track", name="pad")["guid"]
    item = client.call("create_midi_item", track=guid, start_bar=4, length_bars=2)
    take = project.resolve_track(guid).takes[item["item_index"]]
    assert (take.item_start_qn, take.item_end_qn) == (12.0, 19.0)
