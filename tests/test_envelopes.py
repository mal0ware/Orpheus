"""``set_envelope_points`` / ``get_envelope_points``: the automation-envelope verbs.

Bridge-contract tests (the real BridgeClient against FakeReaperBridge, same shape as
tests/test_clone_and_track_params.py). Wire semantics under test: volume points travel
in dB and round-trip through the envelope's native scale; pan points clamp to -1..1;
FX-param points stay NORMALIZED 0..1 (unit translation is python-side, the same split
set_fx_param uses); ``clear=True`` replaces only the [min, max] time range of the new
points; a missing envelope READS as empty but a volume/pan WRITE to an invisible
envelope is a clean human-path error (the API cannot create those without touching the
user's selection state).
"""

from __future__ import annotations

import pytest

from fake_reaper import FakeReaperBridge, FakeReaperProject, FakeTrack, make_handlers
from orpheus_mcp.bridge.client import BridgeClient, BridgeError


@pytest.fixture
def project():
    return FakeReaperProject()


@pytest.fixture
def client(project, tmp_path):
    with FakeReaperBridge(tmp_path / "bridge", make_handlers(project)):
        yield BridgeClient(bridge_dir=tmp_path / "bridge")


def _track(project, name: str, *, volume_env: bool = False, pan_env: bool = False) -> FakeTrack:
    tr = FakeTrack(guid=project._next_guid(), name=name)
    if volume_env:
        tr.envelopes["volume"] = []  # the envelope is "visible" in REAPER terms
    if pan_env:
        tr.envelopes["pan"] = []
    project.tracks.append(tr)
    return tr


# --------------------------------------------------------------------------- #
# Volume: dB in, dB out — native scale is an implementation detail
# --------------------------------------------------------------------------- #


def test_volume_points_round_trip_in_db(client, project):
    _track(project, "vox", volume_env=True)
    res = client.call(
        "set_envelope_points", track="vox", envelope="volume",
        points=[{"time_s": 0.0, "value": -6.0}, {"time_s": 2.0, "value": 0.0}],
    )
    assert res["inserted"] == 2
    assert res["cleared"] is True
    assert res["envelope"] == "volume"

    got = client.call("get_envelope_points", track="vox", envelope="volume")
    assert [p["time_s"] for p in got["points"]] == [0.0, 2.0]
    assert got["points"][0]["value"] == pytest.approx(-6.0)
    assert got["points"][1]["value"] == pytest.approx(0.0)


def test_volume_write_to_invisible_envelope_is_a_clean_human_path_error(client, project):
    _track(project, "vox")  # no volume envelope pre-created
    with pytest.raises(BridgeError, match="volume envelope not visible"):
        client.call(
            "set_envelope_points", track="vox", envelope="volume",
            points=[{"time_s": 0.0, "value": -6.0}],
        )


def test_missing_envelope_reads_as_empty_not_an_error(client, project):
    _track(project, "vox")
    got = client.call("get_envelope_points", track="vox", envelope="volume")
    assert got["points"] == []


# --------------------------------------------------------------------------- #
# Pan: -1..1, clamped
# --------------------------------------------------------------------------- #


def test_pan_points_are_clamped(client, project):
    _track(project, "vox", pan_env=True)
    client.call(
        "set_envelope_points", track="vox", envelope="pan",
        points=[{"time_s": 0.0, "value": 2.5}, {"time_s": 1.0, "value": -9.0}],
    )
    got = client.call("get_envelope_points", track="vox", envelope="pan")
    assert got["points"][0]["value"] == pytest.approx(1.0)
    assert got["points"][1]["value"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# FX param envelopes: normalized 0..1, auto-created on write
# --------------------------------------------------------------------------- #


def test_fx_param_envelope_auto_creates_and_round_trips_normalized(client, project):
    _track(project, "vox")
    client.call("add_fx_by_name", track="vox", name="ReaEQ")
    spec = {"fx_index": 0, "param_index": 4}
    res = client.call(
        "set_envelope_points", track="vox", envelope=spec,
        points=[{"time_s": 10.0, "value": 0.25}, {"time_s": 12.0, "value": 1.7}],
    )
    assert res["envelope"] == "fx 0 param 4"
    got = client.call("get_envelope_points", track="vox", envelope=spec)
    assert got["points"][0]["value"] == pytest.approx(0.25)
    assert got["points"][1]["value"] == pytest.approx(1.0)  # clamped to normalized range


def test_fx_envelope_points_live_in_the_params_native_range(client, project):
    """JSFX sliders keep their own range; REAPER envelopes store that native value. The wire
    stays normalized (2026-07-31 live bug: 1.0 written raw landed as 0.01 of a 0..100 slider)."""
    tr = _track(project, "vox")
    tr.fx_param_ranges[(0, 2)] = (0.0, 100.0)  # e.g. a JSFX percentage slider
    tr.fx_param_ranges[(0, 3)] = (20.0, 20000.0)  # e.g. a JSFX cutoff in Hz
    pct, hz = {"fx_index": 0, "param_index": 2}, {"fx_index": 0, "param_index": 3}
    client.call("set_envelope_points", track="vox", envelope=pct,
                points=[{"time_s": 0.0, "value": 1.0}, {"time_s": 1.0, "value": 0.5}])
    client.call("set_envelope_points", track="vox", envelope=hz,
                points=[{"time_s": 0.0, "value": 0.5}])

    assert [pt["value"] for pt in tr.envelopes[("fx", 0, 2)]] == [100.0, 50.0]
    assert tr.envelopes[("fx", 0, 3)][0]["value"] == pytest.approx(10010.0)
    got = client.call("get_envelope_points", track="vox", envelope=pct)
    assert [pt["value"] for pt in got["points"]] == pytest.approx([1.0, 0.5])
    got = client.call("get_envelope_points", track="vox", envelope=hz)
    assert got["points"][0]["value"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# clear semantics
# --------------------------------------------------------------------------- #


def test_clear_true_replaces_only_the_new_points_time_range(client, project):
    _track(project, "vox", volume_env=True)
    client.call(
        "set_envelope_points", track="vox", envelope="volume",
        points=[{"time_s": t, "value": -3.0} for t in (0.0, 5.0, 10.0)],
    )
    client.call(
        "set_envelope_points", track="vox", envelope="volume",
        points=[{"time_s": 4.0, "value": -12.0}, {"time_s": 6.0, "value": -12.0}],
    )
    got = client.call("get_envelope_points", track="vox", envelope="volume")
    # The point at 5.0 fell inside [4, 6] and was replaced; 0.0 and 10.0 survive.
    assert [p["time_s"] for p in got["points"]] == [0.0, 4.0, 6.0, 10.0]
    assert got["points"][0]["value"] == pytest.approx(-3.0)
    assert got["points"][1]["value"] == pytest.approx(-12.0)


def test_clear_false_stacks_points(client, project):
    _track(project, "vox", volume_env=True)
    client.call(
        "set_envelope_points", track="vox", envelope="volume",
        points=[{"time_s": 5.0, "value": -3.0}],
    )
    client.call(
        "set_envelope_points", track="vox", envelope="volume",
        points=[{"time_s": 5.0, "value": -12.0}], clear=False,
    )
    got = client.call("get_envelope_points", track="vox", envelope="volume")
    assert len(got["points"]) == 2  # both survive — no range was cleared


# --------------------------------------------------------------------------- #
# Guards + undo discipline
# --------------------------------------------------------------------------- #


def test_point_cap_errors(client, project):
    _track(project, "vox", volume_env=True)
    with pytest.raises(BridgeError, match="too many envelope points"):
        client.call(
            "set_envelope_points", track="vox", envelope="volume",
            points=[{"time_s": float(i), "value": 0.0} for i in range(513)],
        )


def test_empty_points_and_bad_spec_error(client, project):
    _track(project, "vox", volume_env=True)
    with pytest.raises(BridgeError, match="points is required"):
        client.call("set_envelope_points", track="vox", envelope="volume", points=[])
    with pytest.raises(BridgeError, match="envelope must be"):
        client.call(
            "set_envelope_points", track="vox", envelope="sidechain",
            points=[{"time_s": 0.0, "value": 0.0}],
        )
    with pytest.raises(BridgeError, match="envelope must be"):
        client.call("get_envelope_points", track="vox", envelope="sidechain")


def test_unknown_track_errors(client, project):
    with pytest.raises(BridgeError, match="no track named"):
        client.call(
            "set_envelope_points", track="ghost", envelope="volume",
            points=[{"time_s": 0.0, "value": 0.0}],
        )


def test_each_set_call_is_one_undo_block(client, project):
    _track(project, "vox", volume_env=True)
    client.call(
        "set_envelope_points", track="vox", envelope="volume",
        points=[{"time_s": 0.0, "value": -6.0}],
    )
    pairs = [e for e in project.undo_events if "envelope points" in e[1]]
    assert [kind for kind, _ in pairs] == ["begin", "end"]
