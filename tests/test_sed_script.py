"""``scripts/pick_sed_head.py`` (Task 2.2, §15.6) — CI-safe tests on hand-built arrays.

No model, no fixtures, no GPU: the real desktop scoring run (``score_fixture_set``) needs
torch/panns_inference + the ``~/.orpheus_fixtures`` set and is exercised only by hand on the
desktop box, per the brief. Everything here is pure math + pure-Python/file-I/O logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.pick_sed_head import (
    CANDIDATE_HEADS,
    ROLE_CLASS_MAP,
    TIE_BREAK_HEAD,
    FixtureSetStale,
    activity_frames_from_mono,
    frame_counts,
    frame_f1,
    mapped_roles,
    pick_winner,
    predicted_frames,
    rms_dbfs,
    verify_manifest_integrity,
)

from orpheus_mcp.models import Timeline

# --------------------------------------------------------------------------- #
# frame_counts / f1_from_counts / frame_f1
# --------------------------------------------------------------------------- #


def test_frame_f1_perfect_agreement_is_one():
    pred = [True, False, True, True, False]
    truth = [True, False, True, True, False]
    assert frame_f1(pred, truth) == pytest.approx(1.0)


def test_frame_f1_all_true_negative_is_one_not_undefined():
    # No positives on either side -- trivial perfect agreement, not a 0/0 blowup.
    pred = [False, False, False]
    truth = [False, False, False]
    assert frame_f1(pred, truth) == pytest.approx(1.0)


def test_frame_f1_no_overlap_is_zero():
    pred = [True, True, False, False]
    truth = [False, False, True, True]
    assert frame_f1(pred, truth) == pytest.approx(0.0)


def test_frame_f1_partial_overlap_matches_hand_computation():
    # tp=2 (idx 0,2), fp=1 (idx 1), fn=1 (idx 3) -> F1 = 2*2 / (2*2+1+1) = 4/6
    pred = [True, True, True, False]
    truth = [True, False, True, True]
    assert frame_counts(pred, truth) == (2, 1, 1)
    assert frame_f1(pred, truth) == pytest.approx(4 / 6)


def test_frame_f1_missed_all_truth_is_zero_no_zero_division():
    pred = [False, False, False]
    truth = [True, True, True]
    assert frame_counts(pred, truth) == (0, 0, 3)
    assert frame_f1(pred, truth) == pytest.approx(0.0)


def test_frame_counts_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        frame_counts([True, False], [True])


# --------------------------------------------------------------------------- #
# pick_winner tie-break (§15.6 -> DecisionLevelAtt)
# --------------------------------------------------------------------------- #


def test_pick_winner_clear_best_no_tie():
    f1 = {"DecisionLevelMax": 0.50, "DecisionLevelAvg": 0.62, "DecisionLevelAtt": 0.55}
    assert pick_winner(f1) == "DecisionLevelAvg"


def test_pick_winner_three_way_tie_breaks_to_att():
    f1 = dict.fromkeys(CANDIDATE_HEADS, 0.70)
    assert pick_winner(f1) == TIE_BREAK_HEAD == "DecisionLevelAtt"


def test_pick_winner_two_way_tie_including_att_breaks_to_att():
    f1 = {"DecisionLevelMax": 0.40, "DecisionLevelAvg": 0.61, "DecisionLevelAtt": 0.61}
    assert pick_winner(f1) == "DecisionLevelAtt"


def test_pick_winner_tie_without_att_falls_back_to_candidate_order():
    # Att was never scored (e.g. its checkpoint genuinely unavailable) -- deterministic
    # fallback to the first tied head in CANDIDATE_HEADS order, never an arbitrary pick.
    f1 = {"DecisionLevelMax": 0.61, "DecisionLevelAvg": 0.61}
    assert pick_winner(f1) == "DecisionLevelMax"


def test_pick_winner_raises_on_empty_input():
    with pytest.raises(ValueError, match="no candidate heads"):
        pick_winner({})


# --------------------------------------------------------------------------- #
# Truth-activity: RMS -> dBFS -> per-1s-frame boolean series
# --------------------------------------------------------------------------- #


def test_rms_dbfs_full_scale_sine_is_about_minus_three_db():
    t = np.linspace(0, 1.0, 1000, endpoint=False)
    sine = np.sin(2 * np.pi * 10 * t).astype(np.float32)
    assert rms_dbfs(sine) == pytest.approx(-3.0103, abs=0.01)


def test_rms_dbfs_silence_is_negative_infinity():
    assert rms_dbfs(np.zeros(100, dtype=np.float32)) == float("-inf")


def test_rms_dbfs_empty_chunk_is_negative_infinity():
    assert rms_dbfs(np.array([], dtype=np.float32)) == float("-inf")


def test_activity_frames_loud_then_silent():
    sr = 1000
    loud = (0.5 * np.ones(sr, dtype=np.float32))  # 0.5 RMS -> ~-6 dBFS, well above -50
    silent = np.zeros(sr, dtype=np.float32)
    mono = np.concatenate([loud, silent])
    frames = activity_frames_from_mono(mono, sr, n_frames=2)
    assert frames == [True, False]


def test_activity_frames_quiet_below_threshold_is_inactive():
    sr = 1000
    # -60 dBFS RMS ~= 0.001 amplitude -- below the -50 dBFS truth floor.
    quiet = (0.001 * np.ones(sr, dtype=np.float32))
    frames = activity_frames_from_mono(quiet, sr, n_frames=1, threshold_dbfs=-50.0)
    assert frames == [False]


def test_activity_frames_trailing_short_stem_reads_as_silence():
    sr = 1000
    mono = 0.5 * np.ones(sr // 2, dtype=np.float32)  # only half a second of audio
    frames = activity_frames_from_mono(mono, sr, n_frames=2)
    assert frames == [True, False]  # frame 1 runs past the end -> silence, not a crash


# --------------------------------------------------------------------------- #
# predicted_frames / mapped_roles
# --------------------------------------------------------------------------- #


def _tl(frames: list[list[float]], classes: list[str]) -> Timeline:
    return Timeline(classes=classes, hop_s=1.0, frames=frames)


def test_predicted_frames_thresholds_max_across_mapped_classes():
    tl = _tl(
        frames=[[0.9, 0.0, 0.1], [0.1, 0.05, 0.0], [0.0, 0.0, 0.9]],
        classes=["Drum kit", "Drum", "Bass guitar"],
    )
    pred = predicted_frames(tl, ["Drum kit", "Drum"], threshold=0.30)
    assert pred == [True, False, False]  # frame 2's 0.9 is on "Bass guitar", not mapped in


def test_predicted_frames_unmapped_classes_are_all_inactive():
    tl = _tl(frames=[[0.9], [0.9]], classes=["Bass guitar"])
    assert predicted_frames(tl, ["Drum kit", "Drum"], threshold=0.30) == [False, False]


def test_mapped_roles_only_keeps_roles_with_a_present_class():
    tl_classes = ["Drum kit", "Bass guitar"]
    # "lead" isn't in ROLE_CLASS_MAP at all; "chords" (-> "Piano") has no matching class here.
    roles = mapped_roles(["drums", "bass", "chords", "lead"], tl_classes)
    assert roles == ["drums", "bass"]


def test_role_class_map_has_no_lead_mapping():
    # Locks the honest scope-exclusion documented in the module docstring.
    assert "lead" not in ROLE_CLASS_MAP


# --------------------------------------------------------------------------- #
# Fixture-set manifest integrity (Task step 4: STOP on a stale set)
# --------------------------------------------------------------------------- #


def _write_fixture_set(root: Path) -> None:
    fixture = root / "genreA_dense"
    (fixture / "stems").mkdir(parents=True)
    (fixture / "mix.flac").write_bytes(b"mix-bytes")
    (fixture / "stems" / "drums.flac").write_bytes(b"drum-bytes")
    (fixture / "truth.json").write_text(
        json.dumps({"roles": {"drums": "stems/drums.flac"}}), encoding="utf-8"
    )

    import hashlib

    files = {}
    for p in sorted(fixture.rglob("*")):
        if p.is_file():
            files[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    set_hash = hashlib.sha256("".join(sorted(files.values())).encode("utf-8")).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps({"files": files, "fixture_set_hash": set_hash}), encoding="utf-8"
    )


def test_verify_manifest_integrity_passes_for_a_consistent_set(tmp_path):
    _write_fixture_set(tmp_path)
    manifest = verify_manifest_integrity(tmp_path)
    assert manifest["fixture_set_hash"]


def test_verify_manifest_integrity_raises_on_missing_manifest(tmp_path):
    with pytest.raises(FixtureSetStale, match="No manifest.json"):
        verify_manifest_integrity(tmp_path)


def test_verify_manifest_integrity_raises_on_file_hash_drift(tmp_path):
    _write_fixture_set(tmp_path)
    (tmp_path / "genreA_dense" / "mix.flac").write_bytes(b"TAMPERED")
    with pytest.raises(FixtureSetStale, match="hash drift"):
        verify_manifest_integrity(tmp_path)


def test_verify_manifest_integrity_raises_on_missing_file(tmp_path):
    _write_fixture_set(tmp_path)
    (tmp_path / "genreA_dense" / "stems" / "drums.flac").unlink()
    with pytest.raises(FixtureSetStale, match="missing"):
        verify_manifest_integrity(tmp_path)


def test_verify_manifest_integrity_raises_on_corrupt_set_hash(tmp_path):
    _write_fixture_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["fixture_set_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FixtureSetStale, match="fixture_set_hash"):
        verify_manifest_integrity(tmp_path)
