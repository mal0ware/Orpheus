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
    ScoreReport,
    activity_frames_from_mono,
    append_pin_run_d_section,
    build_honest_caveat,
    build_vocab_note,
    describe_tie,
    frame_counts,
    frame_f1,
    main,
    mapped_roles,
    pick_winner,
    predicted_frames,
    rms_dbfs,
    verify_manifest_integrity,
    write_registry_head,
)

from orpheus_mcp.analysis.model_registry import RegistryError
from orpheus_mcp.models import Timeline

_REAL_PIN_REPORT = Path("docs/pins/2026-07-model-pins.md")

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


# --------------------------------------------------------------------------- #
# describe_tie -- any tie among the top scorers pick_winner resolved, named (§ item 3)
# --------------------------------------------------------------------------- #


def test_describe_tie_no_tie_is_empty():
    f1 = {"DecisionLevelMax": 0.50, "DecisionLevelAvg": 0.62, "DecisionLevelAtt": 0.55}
    assert describe_tie(f1, pick_winner(f1)) == ""


def test_describe_tie_three_way_names_all_three():
    f1 = dict.fromkeys(CANDIDATE_HEADS, 0.70)
    winner = pick_winner(f1)
    note = describe_tie(f1, winner)
    assert "three-way tie" in note
    for head in CANDIDATE_HEADS:
        assert f"`{head}`" in note
    assert f"`{TIE_BREAK_HEAD}`" in note
    assert "§15.6 breaks to" in note


def test_describe_tie_two_way_including_att_names_both_and_breaks_to_att():
    f1 = {"DecisionLevelMax": 0.40, "DecisionLevelAvg": 0.61, "DecisionLevelAtt": 0.61}
    winner = pick_winner(f1)
    assert winner == "DecisionLevelAtt"
    note = describe_tie(f1, winner)
    assert "two-way tie" in note
    assert "`DecisionLevelAvg`" in note
    assert "`DecisionLevelAtt`" in note
    assert "`DecisionLevelMax`" not in note
    assert "§15.6 breaks to `DecisionLevelAtt`" in note


def test_describe_tie_two_way_without_att_names_both_and_falls_back():
    f1 = {"DecisionLevelMax": 0.61, "DecisionLevelAvg": 0.61}
    winner = pick_winner(f1)
    assert winner == "DecisionLevelMax"
    note = describe_tie(f1, winner)
    assert "two-way tie" in note
    assert "`DecisionLevelMax`" in note
    assert "`DecisionLevelAvg`" in note
    assert "was never scored" in note
    assert "`DecisionLevelMax`" in note.split("breaking to")[1]


def test_describe_tie_single_candidate_is_empty():
    assert describe_tie({"DecisionLevelMax": 0.5}, "DecisionLevelMax") == ""


# --------------------------------------------------------------------------- #
# build_vocab_note -- derived IN CODE from the registry, "Drum machine" called out (item 1)
# --------------------------------------------------------------------------- #


def _write_registry(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "model_registry.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_build_vocab_note_against_real_registry_mentions_drum_machine():
    # No registry_path override -- exercises the real, shipped data/model_registry.json,
    # which the previous commit (24f8b7c) already fixed to carry "Drum machine" everywhere
    # "Drum kit" appears.
    note = build_vocab_note()
    assert '"Drum machine"' in note
    assert "4stem-a/bsroformer" in note
    assert "MISSING" not in note


def test_build_vocab_note_flags_a_missing_drum_machine_entry(tmp_path):
    entries = [
        {"id": "a/one", "tagger_classes": ["Drum kit", "Drum machine"]},
        {"id": "b/two", "tagger_classes": ["Drum kit"]},  # missing Drum machine
        {"id": "c/three", "tagger_classes": ["Piano"]},  # not drum-family at all
    ]
    path = _write_registry(tmp_path, entries)
    note = build_vocab_note(registry_path=path)
    assert "a/one" in note and "b/two" in note
    assert "c/three" not in note
    assert "MISSING from `b/two`" in note


def test_build_vocab_note_no_drum_family_entries_is_empty(tmp_path):
    entries = [{"id": "x/y", "tagger_classes": ["Piano"]}]
    path = _write_registry(tmp_path, entries)
    assert build_vocab_note(registry_path=path) == ""


# --------------------------------------------------------------------------- #
# build_honest_caveat -- derived from actual scores, fires only when overall F1 < 0.1
# --------------------------------------------------------------------------- #


def _report(**overrides) -> ScoreReport:
    base = dict(
        f1_table={h: {"bass": 0.0, "chords": 0.0, "drums": 0.071} for h in CANDIDATE_HEADS},
        counts={h: {"__overall__": [18, 0, 1282]} for h in CANDIDATE_HEADS},
        overall_f1=dict.fromkeys(CANDIDATE_HEADS, 0.027),
        winner="DecisionLevelAtt",
        n_fixtures=24,
        roles_scored=["bass", "chords", "drums"],
        heads_byte_identical=True,
    )
    base.update(overrides)
    return ScoreReport(**base)


def test_build_honest_caveat_fires_below_threshold_and_names_zero_roles():
    report = _report()
    caveat = build_honest_caveat(report)
    assert "Honest caveat" in caveat
    assert "`bass`" in caveat and "`chords`" in caveat
    assert "`drums` only 0.071" in caveat
    assert "CFG.TIMELINE_ACTIVITY_THRESHOLD" in caveat
    assert "scripts/calibrate_gates.py" in caveat
    assert "byte-identical activations" in caveat  # heads_byte_identical=True


def test_build_honest_caveat_silent_above_threshold():
    report = _report(overall_f1=dict.fromkeys(CANDIDATE_HEADS, 0.55))
    assert build_honest_caveat(report) == ""


def test_build_honest_caveat_empty_scores_is_empty():
    assert build_honest_caveat(ScoreReport()) == ""


def test_build_honest_caveat_non_identical_heads_uses_different_clause():
    report = _report(heads_byte_identical=False)
    caveat = build_honest_caveat(report)
    assert "distinct scores" in caveat
    assert "byte-identical activations" not in caveat


# --------------------------------------------------------------------------- #
# append_pin_run_d_section -- idempotent round-trip against a COPY of the real pin report
# (item 1's required verification: no GPU, recorded scores, all four blocks survive).
# --------------------------------------------------------------------------- #


def test_append_pin_run_d_section_reproduces_all_four_content_blocks(tmp_path):
    pin_copy = tmp_path / "2026-07-model-pins.md"
    pin_copy.write_text(_REAL_PIN_REPORT.read_text(encoding="utf-8"), encoding="utf-8")

    report = _report()
    vocab_note = build_vocab_note()
    fixture_hash = "f941582f7b246bf718cebace52a383f82d81761eb611a541b0153dea6a713289"

    append_pin_run_d_section(
        report, fixture_hash, vocab_note=vocab_note, pin_report_path=pin_copy
    )
    text = pin_copy.read_text(encoding="utf-8")

    # Block 1: table.
    assert "DecisionLevelMax" in text and "0.071" in text and "18/0/1282" in text
    # Block 2: scope-limit paragraph (conditional -- present because heads_byte_identical).
    assert "Known scope limit" in text
    # Block 3: vocab note (Drum machine called out).
    assert '"Drum machine"' in text
    # Block 4: honest caveat (overall F1 0.027 < 0.1 threshold).
    assert "Honest caveat" in text
    # Pin run (a) (the other section already in the file) must survive untouched.
    assert "## Pin run (a): separation + dereverb engines" in text
    assert "4stem-a/bsroformer" in text

    # Idempotent: re-appending replaces, never duplicates, the (d) section.
    append_pin_run_d_section(
        report, fixture_hash, vocab_note=vocab_note, pin_report_path=pin_copy
    )
    text_again = pin_copy.read_text(encoding="utf-8")
    assert text_again.count("## Pin run (d): SED head selection") == 1
    assert text_again.count("Honest caveat") == 1
    assert text_again.count('"Drum machine"') == text.count('"Drum machine"')


def test_append_pin_run_d_section_omits_scope_limit_when_heads_not_identical(tmp_path):
    pin_copy = tmp_path / "pins.md"
    report = _report(heads_byte_identical=False)
    append_pin_run_d_section(report, "deadbeef" * 8, pin_report_path=pin_copy)
    text = pin_copy.read_text(encoding="utf-8")
    assert "Known scope limit" not in text


# --------------------------------------------------------------------------- #
# write_registry_head -- synthetic registry (item 4)
# --------------------------------------------------------------------------- #


def test_write_registry_head_replaces_not_duplicates(tmp_path):
    entries = [
        {"id": "other/entry", "head": "keep-me"},
        {"id": "tagger/panns-sed", "head": "DecisionLevelMax"},
    ]
    path = _write_registry(tmp_path, entries)
    write_registry_head("DecisionLevelAtt", registry_path=path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw) == 2  # no new entry appended
    tagger_entries = [e for e in raw if e["id"] == "tagger/panns-sed"]
    assert len(tagger_entries) == 1  # not duplicated
    assert tagger_entries[0]["head"] == "DecisionLevelAtt"
    assert raw[0]["head"] == "keep-me"  # untouched sibling entry


def test_write_registry_head_missing_tagger_id_raises(tmp_path):
    path = _write_registry(tmp_path, [{"id": "other/entry", "head": "x"}])
    with pytest.raises(RegistryError, match="tagger/panns-sed"):
        write_registry_head("DecisionLevelAtt", registry_path=path)


# --------------------------------------------------------------------------- #
# main()'s BLOCKED path -- exit code 2, "BLOCKED:" printed (item 4)
# --------------------------------------------------------------------------- #


def test_main_blocked_path_exits_2_and_prints_blocked(tmp_path, monkeypatch, capsys):
    def _raise(_fixtures_dir):
        raise FixtureSetStale("synthetic staleness for the test")

    monkeypatch.setattr("scripts.pick_sed_head.verify_manifest_integrity", _raise)

    code = main(["--fixtures-dir", str(tmp_path)])

    assert code == 2
    out = capsys.readouterr().out
    assert "BLOCKED:" in out
    assert "synthetic staleness for the test" in out
