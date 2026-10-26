"""Task 5.3 — §15.1 gate-calibration pipeline + D7 config-override loading (all CI-safe).

No real model weights, no GPU, no network: SDR math on constructed signals, the ROC-knee picker
on synthetic score distributions (clean / overlapping / degenerate), the >20% absent-fire guard,
the config override loader + provenance schema + ``verify_config_hash`` busting, and the
``--ci-smoke`` end-to-end run over the 2 committed fixtures via the stub engines/tagger.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from scripts import calibrate_gates as cg

from orpheus_mcp.analysis import cachefs, config
from orpheus_mcp.analysis.config import CFG

# --------------------------------------------------------------------------- #
# SDR math on known signals
# --------------------------------------------------------------------------- #


def _sine(n: int, freq: float, *, sr: int = 8000, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64).reshape(-1, 1)


def test_sdr_perfect_reconstruction_is_ceiling() -> None:
    truth = _sine(8000, 220.0)
    assert cg.sdr_db(truth, truth.copy()) == cg._SDR_CEILING_DB


def test_sdr_known_half_error_ratio() -> None:
    # est = truth scaled by 0.5 -> error == 0.5*truth -> rms(truth)/rms(err) = 2 -> 20log10(2).
    truth = _sine(8000, 220.0)
    est = 0.5 * truth
    assert cg.sdr_db(truth, est) == pytest.approx(20.0 * math.log10(2.0), abs=1e-9)


def test_sdr_silent_truth_is_neg_inf() -> None:
    truth = np.zeros((8000, 1), dtype=np.float64)
    est = _sine(8000, 220.0)
    assert cg.sdr_db(truth, est) == -math.inf


def test_sdr_aligns_to_shorter_and_downmixes() -> None:
    truth = _sine(9000, 110.0)  # longer, mono-in-2d
    est = truth[:8000].copy()
    # Overlap is a perfect copy over the shared length -> ceiling (no error where compared).
    assert cg.sdr_db(truth, est) == cg._SDR_CEILING_DB


# --------------------------------------------------------------------------- #
# ROC-knee picker: clean, overlapping, degenerate
# --------------------------------------------------------------------------- #


def test_roc_knee_clean_separation_le() -> None:
    # Lower is better: goods cluster low, bads high, no overlap -> J == 1.0, knee between.
    values = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
    labels = [True, True, True, False, False, False]
    thr, quality = cg.roc_knee(values, labels, direction="le")
    assert quality == 1.0
    assert thr is not None and 0.2 < thr < 0.8


def test_roc_knee_clean_separation_ge() -> None:
    # Higher is better (agreement r): goods high, bads low.
    values = [0.95, 0.9, 0.85, 0.2, 0.1, 0.05]
    labels = [True, True, True, False, False, False]
    thr, quality = cg.roc_knee(values, labels, direction="ge")
    assert quality == 1.0
    assert thr is not None and 0.2 < thr < 0.85


def test_roc_knee_overlapping_returns_threshold_below_one() -> None:
    values = [0.0, 0.3, 0.6, 0.4, 0.7, 1.0]
    labels = [True, True, True, False, False, False]  # 0.6 (good) > 0.4 (bad): overlap
    thr, quality = cg.roc_knee(values, labels, direction="le")
    assert thr is not None
    assert 0.0 < quality < 1.0


def test_roc_knee_all_good_is_degenerate() -> None:
    thr, quality = cg.roc_knee([0.1, 0.2, 0.3], [True, True, True], direction="le")
    assert thr is None and quality == 0.0


def test_roc_knee_all_bad_is_degenerate() -> None:
    thr, quality = cg.roc_knee([0.1, 0.2, 0.3], [False, False, False], direction="le")
    assert thr is None and quality == 0.0


def test_label_by_sdr_bands() -> None:
    assert cg.label_by_sdr(9.0, good=7.0, bad=4.0) is True
    assert cg.label_by_sdr(2.0, good=7.0, bad=4.0) is False
    assert cg.label_by_sdr(5.0, good=7.0, bad=4.0) is None  # ambiguous band excluded


# --------------------------------------------------------------------------- #
# Absent-fire guard (§3.1c) — logic + the fit-level fallback it drives
# --------------------------------------------------------------------------- #


def test_absent_fire_fraction_counts_suppressed_active_frames() -> None:
    acts = [0.1, 0.5, 0.05, 0.9, 0.2]
    truth = [True, True, True, False, True]  # 4 truth-active
    # threshold 0.3: truth-active acts <= 0.3 are 0.1, 0.05, 0.2 -> 3 of 4 = 0.75.
    assert cg.absent_fire_fraction(acts, truth, 0.3) == pytest.approx(0.75)


def test_absent_fire_fraction_no_active_frames_is_zero() -> None:
    assert cg.absent_fire_fraction([0.1, 0.2], [False, False], 0.5) == 0.0


def test_fit_activity_threshold_guard_forces_fallback() -> None:
    # The best-separating knee lands HIGH (~0.35): the 3 low truth-active frames (0.05) are
    # inseparable from inactives at low thresholds, so the knee catches only the 0.6 actives —
    # suppressing 30% of truth-active frames (> 20% guard) despite a clean J=0.7. Must fall back.
    frames = (
        [cg.FrameSample(activation=0.6, truth_active=True) for _ in range(7)]
        + [cg.FrameSample(activation=0.05, truth_active=True) for _ in range(3)]
        + [cg.FrameSample(activation=0.1, truth_active=False) for _ in range(5)]
        + [cg.FrameSample(activation=0.0, truth_active=False) for _ in range(5)]
    )
    fit = cg.fit_activity_threshold(frames)
    assert fit.fallback is True
    assert fit.shipped == pytest.approx(fit.initial)
    assert fit.fitted is not None and fit.fitted > 0.2  # the guard-tripping high knee
    assert "absent" in fit.note.lower()


def test_fit_activity_threshold_ships_clean_guard_passing_fit() -> None:
    # Truth-active frames all fire strongly; inactive stay near zero -> clean knee, guard passes.
    frames = (
        [cg.FrameSample(activation=0.8, truth_active=True) for _ in range(10)]
        + [cg.FrameSample(activation=0.02, truth_active=False) for _ in range(10)]
    )
    fit = cg.fit_activity_threshold(frames)
    assert fit.fallback is False
    assert fit.fitted is not None and 0.02 < fit.fitted < 0.8


# --------------------------------------------------------------------------- #
# fit_gate fallback logic
# --------------------------------------------------------------------------- #


def _samples(pairs: list[tuple[float, float]]) -> list[cg.MetricSample]:
    return [cg.MetricSample(value=v, sdr=s) for v, s in pairs]


def test_fit_gate_ships_clean_fit() -> None:
    spec = cg.GateMetricSpec("GATE_BLEED_PASS", "bleed", "le")
    # good runs (SDR 9) have low bleed; bad runs (SDR 2) high bleed.
    samples = _samples([(0.05, 9.0), (0.1, 9.0), (0.08, 9.0), (0.6, 2.0), (0.7, 2.0), (0.8, 2.0)])
    fit = cg.fit_gate(spec, samples, good=7.0, bad=4.0)
    assert fit.fallback is False
    assert fit.fitted is not None and 0.1 < fit.fitted < 0.6


def test_fit_gate_too_few_samples_falls_back() -> None:
    spec = cg.GateMetricSpec("GATE_BLEED_PASS", "bleed", "le")
    samples = _samples([(0.05, 9.0), (0.6, 2.0)])  # 1 good, 1 bad
    fit = cg.fit_gate(spec, samples, good=7.0, bad=4.0)
    assert fit.fallback is True
    assert fit.shipped == pytest.approx(float(CFG.GATE_BLEED_PASS))


def test_fit_gate_overlap_below_quality_falls_back() -> None:
    spec = cg.GateMetricSpec("GATE_BLEED_PASS", "bleed", "le")
    # Heavily overlapping metric vs SDR -> low Youden J -> untrusted -> fallback.
    samples = _samples([
        (0.5, 9.0), (0.5, 9.0), (0.5, 9.0), (0.5, 2.0), (0.5, 2.0), (0.5, 2.0),
    ])
    fit = cg.fit_gate(spec, samples, good=7.0, bad=4.0)
    assert fit.fallback is True


# --------------------------------------------------------------------------- #
# Config override loading + provenance schema + verify_config_hash busting (D7)
# --------------------------------------------------------------------------- #


@pytest.fixture
def cfg_cal_snapshot() -> Iterator[None]:
    """Snapshot every # cal field around a test and restore it, regardless of what any committed
    calibrated_thresholds.json loaded at import — so override tests never contaminate the suite."""
    names = config.cal_field_names()
    saved = {n: getattr(CFG, n) for n in names}
    yield
    for n, v in saved.items():
        setattr(CFG, n, v)


def test_load_overrides_applies_only_cal_fields(tmp_path: Path, cfg_cal_snapshot: None) -> None:
    payload = {
        "constants": {
            "GATE_BLEED_PASS": {"value": 0.055, "fit_domain": "synthetic-fixtures",
                                "fixture_set_hash": "abc"},
            "KEY_WINDOW_BARS": {"value": 999},   # not a # cal field -> must be ignored
            "NOT_A_FIELD": {"value": 1.0},        # unknown -> ignored
        }
    }
    f = tmp_path / "calibrated_thresholds.json"
    f.write_text(json.dumps(payload), encoding="utf-8")

    # the two non-cal keys must each be warned about (drift observability) AND ignored;
    # asserting the warnings also keeps pytest's summary pristine.
    with pytest.warns(UserWarning) as caught:
        applied = config.load_calibration_overrides(f)
    warned = " ".join(str(w.message) for w in caught)
    assert "KEY_WINDOW_BARS" in warned and "NOT_A_FIELD" in warned
    assert applied == ["GATE_BLEED_PASS"]
    assert pytest.approx(0.055) == CFG.GATE_BLEED_PASS
    assert CFG.KEY_WINDOW_BARS == 16  # untouched


def test_load_overrides_accepts_bare_scalar(tmp_path: Path, cfg_cal_snapshot: None) -> None:
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"constants": {"GATE_BLEED_PASS": 0.077}}), encoding="utf-8")
    assert config.load_calibration_overrides(f) == ["GATE_BLEED_PASS"]
    assert pytest.approx(0.077) == CFG.GATE_BLEED_PASS


def test_load_overrides_missing_file_is_noop() -> None:
    assert config.load_calibration_overrides(Path("does/not/exist.json")) == []


def test_load_overrides_malformed_file_is_noop(tmp_path: Path) -> None:
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    assert config.load_calibration_overrides(f) == []


def test_load_overrides_unknown_key_warns_and_applies_nothing_for_it(
    tmp_path: Path, cfg_cal_snapshot: None
) -> None:
    # A typo'd constant name: not a # cal field, not a field at all -> must warn (naming the
    # key) and be skipped, rather than silently dropped.
    payload = {"constants": {"GATE_BLEED_PSAS": {"value": 0.5}}}
    f = tmp_path / "c.json"
    f.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.warns(UserWarning, match="GATE_BLEED_PSAS"):
        applied = config.load_calibration_overrides(f)
    assert applied == []


def test_load_overrides_malformed_value_warns_skips_and_continues(
    tmp_path: Path, cfg_cal_snapshot: None
) -> None:
    # A malformed "value" for a known # cal constant (a string that can't coerce to float) must
    # not raise and must not abort the rest of the file's constants.
    before = CFG.GATE_BLEED_PASS
    payload = {
        "constants": {
            "GATE_BLEED_PASS": {"value": "not-a-number"},
            "GATE_SILENCE_PASS_DB": {"value": -30.0},
        }
    }
    f = tmp_path / "c.json"
    f.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.warns(UserWarning, match="GATE_BLEED_PASS"):
        applied = config.load_calibration_overrides(f)

    assert applied == ["GATE_SILENCE_PASS_DB"]
    assert pytest.approx(-30.0) == CFG.GATE_SILENCE_PASS_DB
    assert before == CFG.GATE_BLEED_PASS  # malformed constant left untouched, not raised


def test_load_overrides_malformed_value_never_raises_at_import_call_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_cal_snapshot: None
) -> None:
    # Mirrors the exact call shape used at module import (`load_calibration_overrides()` with no
    # explicit path, resolved via `calibrated_thresholds_path()`) so the "never raise, the whole
    # import graph depends on this" contract is pinned against the real entry point, not just the
    # explicit-path test helper above.
    f = tmp_path / "calibrated_thresholds.json"
    f.write_text(json.dumps({"constants": {"GATE_AGREEMENT_PASS": {"value": []}}}), "utf-8")
    monkeypatch.setattr(config, "calibrated_thresholds_path", lambda: f)

    with pytest.warns(UserWarning, match="GATE_AGREEMENT_PASS"):
        applied = config.load_calibration_overrides()  # must not raise
    assert applied == []


def test_override_busts_verify_config_hash(tmp_path: Path, cfg_cal_snapshot: None) -> None:
    before = cachefs.verify_config_hash()
    f = tmp_path / "c.json"
    # A value guaranteed to differ from the live GATE_BLEED_PASS.
    new_val = round(float(CFG.GATE_BLEED_PASS) + 0.037, 6)
    f.write_text(json.dumps({"constants": {"GATE_BLEED_PASS": {"value": new_val}}}), "utf-8")

    config.load_calibration_overrides(f)
    after = cachefs.verify_config_hash()
    assert after != before  # the D7 loader flows straight into the cache key

    config.reset_calibration()
    assert cachefs.verify_config_hash() == cachefs.verify_config_hash()  # stable after reset


# --------------------------------------------------------------------------- #
# build_json provenance + merge coexistence across domains
# --------------------------------------------------------------------------- #


def test_build_json_carries_per_constant_provenance() -> None:
    fits = cg.fit_all(cg.CalibrationSamples())  # empty -> everything falls back
    payload = cg.build_json(fits, "HASH123", "synthetic-fixtures")
    assert payload["fit_domain"] == "synthetic-fixtures"
    assert payload["fixture_set_hash"] == "HASH123"
    for spec in cg.GATE_SPECS:
        entry = payload["constants"][spec.constant]
        assert set(entry) >= {"value", "fit_domain", "fixture_set_hash", "fallback"}
        assert entry["fit_domain"] == "synthetic-fixtures"
    assert payload["constants"]["TIMELINE_ACTIVITY_THRESHOLD"]["fallback"] is True
    assert payload["constants"]["EXPOSURE_THRESHOLD"]["fallback"] is True


def test_build_json_merge_preserves_other_domain_constants() -> None:
    existing = {
        "constants": {
            "GATE_BLEED_PASS": {"value": 0.123, "fit_domain": "owned-audio",
                                "fixture_set_hash": "OWN"},
        }
    }
    # This run fits nothing (empty samples) but should NOT clobber an owned-audio constant it
    # didn't itself produce... except GATE_BLEED_PASS IS one it produces (fallback) -> overwrite.
    fits = [f for f in cg.fit_all(cg.CalibrationSamples()) if f.constant != "GATE_BLEED_PASS"]
    payload = cg.build_json(fits, "SYN", "synthetic-fixtures", existing=existing)
    # GATE_BLEED_PASS was not re-fit this run -> preserved from the owned-audio pass.
    assert payload["constants"]["GATE_BLEED_PASS"]["fit_domain"] == "owned-audio"
    assert payload["constants"]["GATE_BLEED_PASS"]["value"] == pytest.approx(0.123)


# --------------------------------------------------------------------------- #
# --ci-smoke end-to-end via stubs (ZERO downloads, full pipeline)
# --------------------------------------------------------------------------- #


@pytest.fixture
def _plenty_of_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin free_vram_gb so the VRAM guard never imports real torch (mirrors
    test_reference_tools.py's hermeticity fixture) — the stub separation needs no GPU."""
    from orpheus_mcp.analysis import model_registry

    monkeypatch.setattr(model_registry, "free_vram_gb", lambda: 24.0)


@pytest.mark.analysis
def test_ci_smoke_runs_full_pipeline(
    capsys: pytest.CaptureFixture[str], _plenty_of_vram: None
) -> None:
    rc = cg.main(["--ci-smoke"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ZERO downloads" in out
    assert "GATE_BLEED_PASS" in out  # the fit table rendered
    assert "4 separations" in out  # both stub engines ran on both fixtures


@pytest.mark.analysis
def test_ci_smoke_collects_samples_and_emits_constants(_plenty_of_vram: None) -> None:
    ci_root = Path(__file__).resolve().parent / "fixtures" / "audio"
    manifest = cg.verify_manifest_integrity(ci_root)
    fixture_dirs = sorted(
        ci_root / n for n in {r.split("/", 1)[0] for r in manifest["files"]}
    )
    from stub_engines import StubTagger

    samples = cg.collect_samples(fixture_dirs, cg._ci_smoke_engine_pairs(), StubTagger())
    assert samples.n_fixtures == 2
    assert samples.n_separations == 4  # 2 fixtures x 2 stub engines

    payload = cg.build_json(cg.fit_all(samples), manifest["fixture_set_hash"], "ci-smoke")
    # Every gate constant + the two guarded constants are present with provenance.
    for spec in cg.GATE_SPECS:
        assert spec.constant in payload["constants"]
    assert "TIMELINE_ACTIVITY_THRESHOLD" in payload["constants"]
    assert "EXPOSURE_THRESHOLD" in payload["constants"]


@pytest.mark.analysis
def test_ci_smoke_standalone_subprocess_matches_ci_argv() -> None:
    """Pins the standalone import path (not just the pytest-side-effect one): invokes the script
    exactly as CI does (`.github/workflows/ci.yml`'s "Gate-calibration smoke" step runs
    `uv run python scripts/calibrate_gates.py --ci-smoke`) in a fresh subprocess with the repo
    root as cwd, so the bare `from stub_engines import ...` imports must resolve on their own —
    with no help from pytest's tests/-on-sys.path collection side-effect."""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/calibrate_gates.py", "--ci-smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"standalone --ci-smoke failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "ZERO downloads" in result.stdout


def test_subset_covers_all_genres() -> None:
    names = [f"{g}_{d}" for g in
             ["classical", "hiphop", "lofi", "house", "rock", "jazz", "pop", "cinematic"]
             for d in ["sparse", "medium", "dense"]]
    dirs = [Path(n) for n in names]
    picked = cg._subset(dirs, 12)
    assert len(picked) == 12
    genres = {p.name.split("_")[0] for p in picked}
    assert len(genres) == 8  # all genres represented
