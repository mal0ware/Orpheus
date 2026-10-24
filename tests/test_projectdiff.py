"""project_changes' pure diff (spec 2026-10-23 REAPER sync §2.3): one test per change kind."""

from __future__ import annotations

import copy

import pytest

from orpheus_mcp.analysis import projectdiff
from orpheus_mcp.analysis.projectdiff import diff_snapshots


def _snap() -> dict:
    return {
        "project": {"name": "song.rpp", "path": "C:/p/song.rpp", "length_s": 60.0,
                    "state_change_count": 10},
        "tempo_map": [{"time_s": 0.0, "bar": 1, "beat": 1.0, "bpm": 120.0, "meter": [4, 4],
                       "linear": False}],
        "markers": [{"index": 1, "name": "Verse", "time_s": 2.0, "bar": 2, "beat": 1.0}],
        "regions": [{"index": 1, "name": "Chorus", "start_s": 8.0, "end_s": 16.0,
                     "start_bar": 5, "start_beat": 1.0, "end_bar": 9, "end_beat": 1.0}],
        "tracks": [
            {"guid": "{A}", "index": 1, "name": "Keys", "mute": False, "solo": False,
             "volume_db": 0.0, "pan": 0.0, "fx_count": 1,
             "items": [{"guid": "{i1}", "position_s": 0.0, "length_s": 8.0, "bar": 1,
                        "beat": 1.0, "kind": "midi", "midi_hash": "h1"}]},
            {"guid": "{B}", "index": 2, "name": "Bass", "mute": False, "solo": False,
             "volume_db": -6.0, "pan": 0.0, "fx_count": 0, "items": []},
            {"guid": "{C}", "index": 3, "name": "Vox", "mute": False, "solo": False,
             "volume_db": 0.0, "pan": 0.0, "fx_count": 0,
             "items": [{"guid": "{i2}", "position_s": 8.0, "length_s": 4.0, "bar": 5,
                        "beat": 1.0, "kind": "audio", "source": "C:/x/vox.wav"}]},
        ],
    }


def _kinds(changes) -> list[str]:
    return [c.kind for c in changes]


def _track(snap, guid):
    return next(t for t in snap["tracks"] if t["guid"] == guid)


def test_identical_snapshots_have_no_changes():
    assert diff_snapshots(_snap(), _snap()) == []


def test_float_noise_is_ignored():
    new = _snap()
    _track(new, "{B}")["volume_db"] = -6.0 + 0.01
    new["tracks"][0]["items"][0]["position_s"] = 0.0004
    new["markers"][0]["time_s"] = 2.0005
    assert diff_snapshots(_snap(), new) == []


def test_track_added_removed_renamed():
    new = _snap()
    new["tracks"].append({"guid": "{D}", "index": 4, "name": "Pad", "items": []})
    new["tracks"] = [t for t in new["tracks"] if t["guid"] != "{B}"]
    _track(new, "{C}")["name"] = "Lead Vox"
    changes = diff_snapshots(_snap(), new)
    assert set(_kinds(changes)) == {"track_added", "track_removed", "track_renamed"}
    assert any(c.summary == "Track 'Vox' renamed to 'Lead Vox'." for c in changes)


def test_moving_one_track_reports_only_that_track():
    new = _snap()
    a, b, c = new["tracks"]
    a["index"], b["index"], c["index"] = 3, 1, 2
    new["tracks"] = [b, c, a]
    changes = diff_snapshots(_snap(), new)
    assert _kinds(changes) == ["track_moved"]
    assert changes[0].track == "Keys" and (changes[0].before, changes[0].after) == (1, 3)


def test_mixer_changes():
    new = _snap()
    keys = _track(new, "{A}")
    keys.update(mute=True, solo=True, volume_db=-3.5, pan=-0.3, fx_count=2)
    changes = diff_snapshots(_snap(), new)
    assert _kinds(changes) == ["track_mute", "track_solo", "track_volume", "track_pan",
                               "track_fx_count"]
    assert changes[2].summary == "Track 'Keys' volume 0.0 dB -> -3.5 dB."


def test_item_changes():
    new = _snap()
    keys_item = _track(new, "{A}")["items"][0]
    keys_item.update(midi_hash="h2", length_s=12.0)
    vox_item = _track(new, "{C}")["items"][0]
    vox_item.update(position_s=10.0, bar=6, source="C:/x/vox_take2.wav")
    _track(new, "{B}")["items"].append({"guid": "{i3}", "position_s": 16.0, "length_s": 8.0,
                                        "bar": 9, "beat": 1.0, "kind": "midi",
                                        "midi_hash": "h9"})
    changes = diff_snapshots(_snap(), new)
    by_kind = {c.kind: c for c in changes}
    assert set(by_kind) == {"item_added", "item_moved", "item_resized", "item_notes_changed",
                            "item_source_changed"}
    assert by_kind["item_added"].summary == "New midi item on 'Bass' at bar 9."
    assert by_kind["item_notes_changed"].summary == (
        "Notes changed in the MIDI item on 'Keys' at bar 1."
    )
    assert by_kind["item_moved"].summary == "Audio item moved on 'Vox': bar 5 -> bar 6."
    assert "vox_take2.wav" in by_kind["item_source_changed"].summary


def test_item_moved_between_tracks_and_removed():
    new = _snap()
    item = _track(new, "{A}")["items"].pop()
    _track(new, "{B}")["items"].append(item)
    _track(new, "{C}")["items"].clear()
    changes = diff_snapshots(_snap(), new)
    assert _kinds(changes) == ["item_removed", "item_moved"]
    assert "from 'Keys' to 'Bass'" in changes[1].summary
    assert changes[0].summary == "Audio item removed from 'Vox' (was at bar 5)."


def test_marker_and_region_changes():
    new = _snap()
    new["markers"][0].update(name="Verse 1", time_s=4.0, bar=3)
    new["markers"].append({"index": 2, "name": "Outro", "time_s": 50.0, "bar": 26,
                           "beat": 3.0})
    new["regions"] = []
    changes = diff_snapshots(_snap(), new)
    assert _kinds(changes) == ["marker_renamed", "marker_moved", "marker_added",
                               "region_removed"]
    assert changes[2].summary == "Marker 'Outro' added at bar 26 beat 3."
    assert changes[1].summary == "Marker 'Verse 1' moved: bar 2 -> bar 3."


def test_region_end_move_counts_as_moved():
    new = _snap()
    new["regions"][0]["end_s"] = 18.0
    assert _kinds(diff_snapshots(_snap(), new)) == ["region_moved"]


def test_tempo_map_and_length_changes():
    new = _snap()
    new["tempo_map"].append({"time_s": 8.0, "bar": 5, "beat": 1.0, "bpm": 120.0,
                             "meter": [3, 4], "linear": False})
    new["project"]["length_s"] = 64.0
    changes = diff_snapshots(_snap(), new)
    assert _kinds(changes) == ["tempo_map_changed", "project_length_changed"]
    assert "120 BPM 3/4 at bar 5" in changes[0].summary


def test_empty_lua_tables_arrive_as_dicts_and_are_lists_here():
    old, new = _snap(), copy.deepcopy(_snap())
    for s in (old, new):
        s["markers"] = {}
        s["tracks"][1]["items"] = {}
    assert diff_snapshots(old, new) == []


def test_snapshot_store_round_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(projectdiff, "snapshot_dir", lambda: tmp_path / "snapshots")
    key = projectdiff.project_key({"path": "C:/p/song.rpp", "name": "song.rpp"})
    assert projectdiff.load_snapshot(key) is None
    projectdiff.save_snapshot(key, _snap())
    assert projectdiff.load_snapshot(key) == _snap()
    unsaved = projectdiff.project_key({"path": "", "name": "untitled"})
    assert unsaved != key


@pytest.mark.parametrize(("bar", "beat", "text"), [(9, 1.0, "bar 9"), (9, 2.5, "bar 9 beat 2.5"),
                                                    (None, None, "")])
def test_positions_read_like_the_ruler(bar, beat, text):
    assert projectdiff._where(bar, beat) == text
