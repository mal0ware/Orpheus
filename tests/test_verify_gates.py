"""``analysis/verify.py`` (Task 4.2, §5): the three constructed-signal-testable
verification gates (mix consistency, bleed, silence honesty), the §5 overall decision
rule, and ``run_gates``'s composition of all four (gate 4/agreement is a Phase-5 stub —
always N/A here). Every numeric case below is driven by hand-built arrays/Timelines, no
real model weights, matching the Task 4.1 CI-safe convention.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.model_registry import ModelEntry, classes_for_target
from orpheus_mcp.analysis.verify import (
    ensemble_avg_wave,
    gate_agreement,
    gate_bleed,
    gate_mix_consistency,
    gate_silence_honesty,
    overall_verdict,
    run_gates,
)
from orpheus_mcp.models import GateResult, Timeline
from stub_engines import StubTagger

# --------------------------------------------------------------------------- #
# Gate 1 — mix consistency (§5.1): multi-stem only, residual_db = 20log10(rms(mix -
# sum(stems)) / rms(mix)); pass iff <= CFG.GATE_MIX_RESIDUAL_PASS_DB (-20 dB default).
# --------------------------------------------------------------------------- #


def _sine(n: int, freq: float, *, sr: int = 8000, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64).reshape(-1, 1)


def test_gate_mix_consistency_perfect_sum_passes():
    n = 8000
    vocals = _sine(n, 220.0)
    drums = _sine(n, 55.0, amp=0.6)
    mix = vocals + drums  # exact reconstruction, same float ops both sides

    result = gate_mix_consistency(mix, {"vocals": vocals, "drums": drums}, True)

    assert isinstance(result, GateResult)
    assert result.value is not None
    assert result.value < -60.0
    assert result.verdict == "pass"


def test_gate_mix_consistency_one_stem_zeroed_fails():
    n = 8000
    vocals = _sine(n, 220.0)
    drums = _sine(n, 55.0, amp=0.6)
    mix = vocals + drums

    # "drums" reported as all-zero — the residual now equals the whole missing stem,
    # nowhere near negligible relative to the mix.
    result = gate_mix_consistency(mix, {"vocals": vocals, "drums": np.zeros_like(drums)}, True)

    assert result.value is not None
    assert result.value > CFG.GATE_MIX_RESIDUAL_PASS_DB
    assert result.verdict == "fail"


def test_gate_mix_consistency_single_target_is_na():
    n = 800
    mix = _sine(n, 220.0)
    stems = {"vocals": mix}

    result = gate_mix_consistency(mix, stems, False)

    assert result.verdict == "na"
    assert result.value is None


# --------------------------------------------------------------------------- #
# Gate 2 — bleed (§5.2): tagger-on-stem activations; active frames = target > CFG.
# TIMELINE_ACTIVITY_THRESHOLD; value = mean over active frames of max(non-target)/
# (target+eps), clipped [0,1]; pass < CFG.GATE_BLEED_PASS (strict <).
# --------------------------------------------------------------------------- #


def _activations(rows: list[tuple[float, float]]) -> np.ndarray:
    """Two-column (target, non-target) activation matrix, one row per frame."""
    return np.asarray(rows, dtype=np.float64)


def test_gate_bleed_ratio_below_threshold_passes():
    # 0.15 / (0.9 + eps) ~= 0.1667 < 0.20 -> pass.
    arr = _activations([(0.9, 0.15)] * 4)
    result = gate_bleed(arr, target_idx=[0], mapped_idx=[0, 1])

    assert result.value is not None
    assert result.value == pytest.approx(0.15 / (0.9 + 1e-6), abs=1e-6)
    assert result.verdict == "pass"


def test_gate_bleed_ratio_above_threshold_fails():
    # 0.25 / (0.9 + eps) ~= 0.2778 >= 0.20 -> fail.
    arr = _activations([(0.9, 0.25)] * 4)
    result = gate_bleed(arr, target_idx=[0], mapped_idx=[0, 1])

    assert result.value is not None
    assert result.value == pytest.approx(0.25 / (0.9 + 1e-6), abs=1e-6)
    assert result.verdict == "fail"


def test_gate_bleed_zero_active_frames_is_na():
    # target never exceeds CFG.TIMELINE_ACTIVITY_THRESHOLD (0.30 default) -> no active
    # frames to average over -> N/A, never a division over an empty set.
    arr = _activations([(0.1, 0.05)] * 5)
    result = gate_bleed(arr, target_idx=[0], mapped_idx=[0, 1])

    assert result.verdict == "na"
    assert result.value is None


def test_gate_bleed_empty_target_idx_is_na():
    arr = _activations([(0.9, 0.15)] * 4)
    result = gate_bleed(arr, target_idx=[], mapped_idx=[0, 1])

    assert result.verdict == "na"
    assert result.value is None


def test_gate_bleed_ignores_columns_outside_mapped_idx():
    # A huge activation in column 2 must NOT count toward the bleed ratio because it is
    # outside mapped_idx — only mapped, non-target columns are eligible "bleed" sources.
    arr = np.asarray([[0.9, 0.05, 0.99]] * 4, dtype=np.float64)
    result = gate_bleed(arr, target_idx=[0], mapped_idx=[0, 1])

    assert result.value is not None
    assert result.value == pytest.approx(0.05 / (0.9 + 1e-6), abs=1e-6)
    assert result.verdict == "pass"


# --------------------------------------------------------------------------- #
# Gate 3 — silence honesty (§5.3): inactive windows minus a tail allowance after each
# active region are compared (RMS, dB) against active-window RMS; no qualifying inactive
# window -> N/A (wall-to-wall).
# --------------------------------------------------------------------------- #

_SR = 1000  # simple round-number rate — every second is exactly 1000 samples


def _const_stem(levels: list[tuple[float, float]], *, sr: int = _SR) -> np.ndarray:
    """Concatenate ``(seconds, amplitude)`` constant-value chunks into a (samples, 1) stem."""
    chunks = [np.full((int(round(secs * sr)), 1), amp, dtype=np.float64) for secs, amp in levels]
    return np.concatenate(chunks, axis=0)


def _flat_target_timeline(active_flags: list[bool]) -> Timeline:
    frames = [[0.9, 0.0] if flag else [0.05, 0.0] for flag in active_flags]
    return Timeline(classes=["target", "other"], hop_s=1.0, frames=frames)


def test_gate_silence_honesty_reverb_tail_within_allowance_passes():
    # Active note for 3s, then a 3s "reverb tail" (excluded by the default 3s tail
    # allowance, so its content is irrelevant), then true near-silence from t=6..10.
    stem = _const_stem([(3.0, 1.0), (3.0, 0.5), (4.0, 0.001)])
    active_flags = [True, True, True, False, False, False, False, False, False, False]
    tl = _flat_target_timeline(active_flags)

    result = gate_silence_honesty(stem, _SR, tl, target_idx=[0])

    assert result.value is not None
    assert result.value <= CFG.GATE_SILENCE_PASS_DB
    assert result.verdict == "pass"


def test_gate_silence_honesty_energy_past_tail_allowance_fails():
    # Same active note + reverb-tail shape, but real (non-negligible) energy persists
    # past the 3s tail allowance (t=6..10, i.e. ~5s after the note ended at t=3) —
    # exactly the "ghost of everything everywhere" failure mode §5.3 exists to catch.
    stem = _const_stem([(3.0, 1.0), (3.0, 0.5), (4.0, 0.3)])
    active_flags = [True, True, True, False, False, False, False, False, False, False]
    tl = _flat_target_timeline(active_flags)

    result = gate_silence_honesty(stem, _SR, tl, target_idx=[0])

    assert result.value is not None
    assert result.value > CFG.GATE_SILENCE_PASS_DB
    assert result.verdict == "fail"


def test_gate_silence_honesty_wall_to_wall_is_na():
    stem = _const_stem([(10.0, 1.0)])
    active_flags = [True] * 10
    tl = _flat_target_timeline(active_flags)

    result = gate_silence_honesty(stem, _SR, tl, target_idx=[0])

    assert result.verdict == "na"
    assert result.value is None


def test_gate_silence_honesty_zero_active_frames_is_na():
    # The target never plays at all -- there is no active-window RMS to normalize
    # against, so this is N/A rather than a division against an empty reference.
    stem = _const_stem([(10.0, 0.01)])
    active_flags = [False] * 10
    tl = _flat_target_timeline(active_flags)

    result = gate_silence_honesty(stem, _SR, tl, target_idx=[0])

    assert result.verdict == "na"
    assert result.value is None


def test_gate_silence_honesty_empty_target_idx_is_na():
    stem = _const_stem([(10.0, 1.0)])
    tl = _flat_target_timeline([True] * 3 + [False] * 7)

    result = gate_silence_honesty(stem, _SR, tl, target_idx=[])

    assert result.verdict == "na"
    assert result.value is None


# --------------------------------------------------------------------------- #
# Gate 4 — cross-model agreement (§5.4, Task 5.1): log-mel (128 bands, hop 1024) Pearson
# r over flattened frames between two engines' takes on the same target; pass iff
# r >= CFG.GATE_AGREEMENT_PASS (0.90 default).
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
def test_gate_agreement_identical_stems_pass_r_near_one():
    stem = _sine(8000, 220.0, sr=8000)

    result = gate_agreement(stem, stem.copy(), sr=8000)

    assert result.value is not None
    assert result.value == pytest.approx(1.0, abs=1e-6)
    assert result.verdict == "pass"


@pytest.mark.analysis
def test_gate_agreement_true_stereo_input_downmixes_and_passes():
    """(n, 2) stereo input exercises the mean-axis downmix path directly: a stereo stem
    compared against its own mono downmix must correlate ~1.0 (same signal content)."""
    mono = _sine(8000, 220.0, sr=8000)
    stereo = np.concatenate([mono, mono], axis=1)
    assert stereo.shape == (8000, 2)

    result = gate_agreement(stereo, mono, sr=8000)

    assert result.value == pytest.approx(1.0, abs=1e-6)
    assert result.verdict == "pass"


@pytest.mark.analysis
def test_gate_agreement_noise_vs_sine_fails():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((8000, 1))
    sine = _sine(8000, 220.0, sr=8000)

    result = gate_agreement(noise, sine, sr=8000)

    assert result.value is not None
    assert result.value < CFG.GATE_AGREEMENT_PASS
    assert result.verdict == "fail"


@pytest.mark.analysis
def test_gate_agreement_one_silent_one_active_fails_zero_not_nan():
    # Pearson r is undefined (0/0) when one side has zero variance -- must read as an
    # explicit "fail" at value 0.0, never NaN and never a crash.
    silent = np.zeros((8000, 1))
    active = _sine(8000, 220.0, sr=8000)

    result = gate_agreement(silent, active, sr=8000)

    assert result.value is not None
    assert math.isfinite(result.value)
    assert result.value == 0.0
    assert result.verdict == "fail"


def test_gate_agreement_both_silent_is_na():
    silent_a = np.zeros((8000, 1))
    silent_b = np.zeros((8000, 1))

    result = gate_agreement(silent_a, silent_b, sr=8000)

    assert result.verdict == "na"
    assert result.value is None


@pytest.mark.analysis
def test_gate_agreement_length_mismatch_aligns_to_shorter_before_scoring():
    stem_a = _sine(8000, 220.0, sr=8000)
    # stem_b repeats stem_a's content then appends unrelated noise past stem_a's length --
    # aligning to the shorter length before mel extraction must crop the noise tail out
    # entirely, leaving a near-perfect agreement despite the raw length mismatch.
    rng = np.random.default_rng(1)
    stem_b = np.concatenate([stem_a, rng.standard_normal((500, 1))], axis=0)
    assert stem_a.shape[0] != stem_b.shape[0]

    result = gate_agreement(stem_a, stem_b, sr=8000)

    assert result.value is not None
    assert result.value == pytest.approx(1.0, abs=1e-6)
    assert result.verdict == "pass"


# --------------------------------------------------------------------------- #
# avg_wave ensemble (§5.4, Task 5.1): length-aligned element-wise mean across stems.
# --------------------------------------------------------------------------- #


def test_ensemble_avg_wave_dc_offset_copies_is_exact_mean():
    base = _sine(4000, 220.0, sr=4000)
    a = base + 1.0
    b = base - 1.0

    result = ensemble_avg_wave([a, b])

    np.testing.assert_allclose(result, base, atol=1e-9)


def test_ensemble_avg_wave_different_lengths_aligns_then_averages():
    a = np.full((10, 1), 2.0)
    b = np.full((6, 1), 4.0)

    result = ensemble_avg_wave([a, b])

    assert result.shape[0] == 6
    np.testing.assert_allclose(result, np.full((6, 1), 3.0))


def test_ensemble_avg_wave_preserves_channel_count():
    a = np.zeros((10, 2))
    b = np.ones((10, 2))

    result = ensemble_avg_wave([a, b])

    assert result.shape == (10, 2)
    np.testing.assert_allclose(result, np.full((10, 2), 0.5))


# --------------------------------------------------------------------------- #
# §5 overall decision rule truth table: fail if ANY applicable gate fails; unverifiable
# iff ALL gates are N/A; N/A gates excluded otherwise.
# --------------------------------------------------------------------------- #

_PASS = GateResult(verdict="pass", value=-40.0)
_FAIL = GateResult(verdict="fail", value=0.0)
_NA = GateResult(verdict="na", value=None)


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        pytest.param([_PASS, _PASS, _PASS, _PASS], "pass", id="all-pass"),
        pytest.param([_PASS, _FAIL, _PASS, _PASS], "fail", id="one-fail"),
        pytest.param([_PASS, _NA, _PASS, _NA], "pass", id="mixed-pass-and-na-excluded"),
        pytest.param([_NA, _NA, _NA, _NA], "unverifiable", id="all-na"),
        pytest.param([_NA, _FAIL, _NA, _NA], "fail", id="fail-beats-na"),
        pytest.param([], "unverifiable", id="empty-gate-list"),
    ],
)
def test_overall_verdict_truth_table(gates, expected):
    assert overall_verdict(gates) == expected


# --------------------------------------------------------------------------- #
# run_gates: composes gates 1-3 + the Phase-5 gate-4 stub into one GateSet.
# --------------------------------------------------------------------------- #


def test_run_gates_composes_gateset_with_agreement_stubbed_na():
    n = 4000
    vocals = _sine(n, 220.0, sr=4000)
    drums = _sine(n, 55.0, sr=4000, amp=0.6)
    mix = vocals + drums
    stems = {"vocals": vocals, "drums": drums}

    # StubTagger with fixed activations: "Singing" (target) active every frame, low
    # non-target -- bleed should read a real, non-N/A number.
    tagger = StubTagger(
        classes=["Singing", "Speech"], hop_s=1.0, activations=[[0.9, 0.05]] * 1
    )
    mix_timeline = Timeline(classes=["Singing", "Speech"], hop_s=1.0, frames=[[0.9, 0.05]])

    gates = run_gates(
        mix=mix,
        stems=stems,
        target="vocals",
        sr=4000,
        multi_stem=True,
        tagger=tagger,
        target_tagger_classes=["Singing"],
        mix_timeline=mix_timeline,
    )

    assert gates.agreement.verdict == "na"
    assert gates.agreement.value is None
    assert gates.mix_consistency.verdict == "pass"  # vocals+drums reconstructs mix exactly
    assert gates.bleed.verdict in {"pass", "fail", "na"}
    assert gates.silence_honesty.verdict in {"pass", "fail", "na"}
    assert gates.overall in {"pass", "fail", "unverifiable"}


@pytest.mark.analysis
def test_run_gates_computes_agreement_when_other_stem_provided():
    # Task 5.1: passing an ``other_stem`` (a second engine's take on the same target)
    # switches gate 4 from the Phase-5 na stub to a real computed verdict, while every
    # other gate stays exactly as it was.
    n = 4000
    vocals = _sine(n, 220.0, sr=4000)
    drums = _sine(n, 55.0, sr=4000, amp=0.6)
    mix = vocals + drums
    stems = {"vocals": vocals, "drums": drums}

    tagger = StubTagger(
        classes=["Singing", "Speech"], hop_s=1.0, activations=[[0.9, 0.05]] * 1
    )
    mix_timeline = Timeline(classes=["Singing", "Speech"], hop_s=1.0, frames=[[0.9, 0.05]])

    gates = run_gates(
        mix=mix,
        stems=stems,
        target="vocals",
        sr=4000,
        multi_stem=True,
        tagger=tagger,
        target_tagger_classes=["Singing"],
        mix_timeline=mix_timeline,
        other_stem=vocals.copy(),
    )

    assert gates.agreement.verdict == "pass"
    assert gates.agreement.value is not None
    assert gates.agreement.value == pytest.approx(1.0, abs=1e-6)
    assert gates.overall in {"pass", "fail"}


# --------------------------------------------------------------------------- #
# D13: the bleed gate must see PER-TARGET tagger classes, not a multi-target entry's
# flat tagger_classes union -- otherwise vocal-class activation bleeding into an
# isolated drums stem reads as "target" energy and the gate stays blind to it.
# --------------------------------------------------------------------------- #


def test_run_gates_bleed_gate_d13_per_target_classes_catches_cross_target_bleed():
    # A multi-stem entry whose flat tagger_classes is the pre-D13 union across every
    # target (vocals + drums); tagger_classes_by_target scopes each target to its own
    # family, exactly like the real seed data's 4-stem entries.
    entry = ModelEntry(
        id="4stem-fake/bleed-fixture",
        task="multi-stem",
        targets=["vocals", "drums"],
        engine="audio-separator",
        checkpoint_url=None,
        sha256=None,
        vram_gb=0.0,
        rt_factor=None,
        benchmark=None,
        license="MIT",
        rank=1,
        tagger_classes=["Singing", "Drum kit"],  # the pre-D13 flat union
        tagger_classes_by_target={"vocals": ["Singing"], "drums": ["Drum kit"]},
    )

    n = 4000
    drums_stem = _sine(n, 55.0, sr=4000, amp=0.6)

    # Vocal ("Singing") bleed into the isolated drums stem reads exactly as high as the
    # genuine drums-family ("Drum kit") activation -- the scenario the fix must catch.
    tagger = StubTagger(classes=["Singing", "Drum kit"], hop_s=1.0, activations=[[0.9, 0.9]])
    mix_timeline = Timeline(classes=["Singing", "Drum kit"], hop_s=1.0, frames=[[0.9, 0.9]])

    gates = run_gates(
        mix=drums_stem,
        stems={"drums": drums_stem},
        target="drums",
        sr=4000,
        multi_stem=False,
        tagger=tagger,
        # D13 fix: classes_for_target scopes to ["Drum kit"] only -- "Singing" correctly
        # counts as non-target bleed.
        target_tagger_classes=classes_for_target(entry, "drums"),
        mix_timeline=mix_timeline,
    )

    assert gates.bleed.value is not None
    assert gates.bleed.value == pytest.approx(1.0, abs=1e-3)
    assert gates.bleed.verdict == "fail"

    # NOT asserted -- documentation only: had the caller instead passed the OLD flat
    # union (entry.tagger_classes == ["Singing", "Drum kit"]) as target_tagger_classes,
    # BOTH columns would count as "target", leaving nothing in mapped_idx outside the
    # target set -- nontarget_vals defaults to zeros, the ratio collapses to 0.0, and the
    # bleed gate would have read "pass" on a stem with genuine vocal bleed. That silent
    # pass is exactly the D13 defect this test proves is fixed.
