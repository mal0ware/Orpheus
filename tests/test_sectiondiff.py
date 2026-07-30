"""Section-diff (analysis/sectiondiff.py): the minute-details differential pass.

All fixtures are SYNTHETIC (deterministic numpy, no real songs, no decode stack): a
"section" is a repeatable 4 s pseudo-music bed (kick pattern + tones + a fixed noise
floor) planted into a quiet timeline at exact sample offsets, so every test controls
precisely what changed between repetition A and repetition B — and the module's flags
must find exactly that, and stay SILENT when nothing changed (an empty flag list is a
measurement, per the module contract).
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus_mcp.analysis import sectiondiff

SR = 22050
SECTION_S = 4.0


def _tone(freq: float, dur_s: float, amp: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(round(dur_s * sr))) / sr
    return amp * np.sin(2.0 * np.pi * freq * t)


def _kick(sr: int = SR) -> np.ndarray:
    """A 60 Hz decaying thump, ~150 ms — enough low-band flux to register as an onset."""
    t = np.arange(int(0.15 * sr)) / sr
    return (0.8 * np.sin(2.0 * np.pi * 60.0 * t) * np.exp(-t * 30.0)).astype(np.float64)


def make_section(kick_every_s: float = 0.5) -> np.ndarray:
    """The base 4 s section: kicks on a grid, a 440 Hz bed, a 4 kHz sparkle, and a FIXED
    (seeded) noise floor so every band carries energy — a filter cut must be measurable.
    Deterministic: two calls with the same args return identical arrays."""
    n = int(round(SECTION_S * SR))
    rng = np.random.default_rng(1234)
    y = 0.02 * rng.standard_normal(n)
    y += _tone(440.0, SECTION_S, 0.25)
    y += _tone(4000.0, SECTION_S, 0.08)
    k = _kick()
    at = 0.0
    while at < SECTION_S - 0.16:
        i = int(round(at * SR))
        y[i : i + len(k)] += k
        at += kick_every_s
    return y


def make_song(placements: list[tuple[float, np.ndarray]], total_s: float = 24.0) -> np.ndarray:
    """A quiet timeline with sections planted at exact sample offsets (a faint seeded
    noise bed keeps 'silence' realistic without correlating with anything)."""
    rng = np.random.default_rng(99)
    y = 0.001 * rng.standard_normal(int(round(total_s * SR)))
    for start_s, seg in placements:
        i = int(round(start_s * SR))
        y[i : i + len(seg)] += seg
    return y


# --------------------------------------------------------------------------- #
# 1. identical repetitions -> zero flags
# --------------------------------------------------------------------------- #


def test_identical_sections_produce_zero_flags():
    s = make_section()
    y = make_song([(2.0, s), (12.0, s)])
    diff = sectiondiff.diff_sections(y, SR, (2.0, 6.0), (12.0, 16.0))
    assert sectiondiff.describe_diff(diff) == []
    assert diff.residual_peaks == []


# --------------------------------------------------------------------------- #
# 2. a layer entering halfway -> residual peak at the right time + band flag
# --------------------------------------------------------------------------- #


def test_added_layer_flags_band_and_localizes_entry():
    s = make_section()
    layered = s.copy()
    half = int(round(2.0 * SR))
    layered[half:] += _tone(3000.0, SECTION_S - 2.0, 0.3)
    y = make_song([(2.0, s), (12.0, layered)])
    diff = sectiondiff.diff_sections(y, SR, (2.0, 6.0), (12.0, 16.0))
    flags = sectiondiff.describe_diff(diff)

    assert any("presence 2.5-6k" in f and "+" in f for f in flags), flags
    assert diff.residual_peaks, "the entering layer must produce residual peaks"
    # every flagged residual window sits in the second half (layer enters at +2.0 s)
    for peak in diff.residual_peaks:
        assert 1.5 <= peak.offset_s <= 3.5, diff.residual_peaks
    assert any("residual layer" in f for f in flags)


# --------------------------------------------------------------------------- #
# 3. doubled kick pattern -> bassline onset-density flag
# --------------------------------------------------------------------------- #


def test_doubled_kick_pattern_flags_bass_onset_density():
    a = make_section(kick_every_s=0.5)
    b = make_section(kick_every_s=0.25)
    y = make_song([(2.0, a), (12.0, b)])
    diff = sectiondiff.diff_sections(y, SR, (2.0, 6.0), (12.0, 16.0))
    flags = sectiondiff.describe_diff(diff)

    assert diff.low_onset_density_b > diff.low_onset_density_a * 1.5
    assert any("bassline pattern change" in f for f in flags), flags


# --------------------------------------------------------------------------- #
# 4. lowpassed repetition -> presence/air cut flags
# --------------------------------------------------------------------------- #


def test_lowpassed_section_flags_high_band_cuts():
    s = make_section()
    spec = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(len(s), 1.0 / SR)
    spec[freqs > 2000.0] = 0.0  # brickwall lowpass at 2 kHz
    b = np.fft.irfft(spec, n=len(s))
    y = make_song([(2.0, s), (12.0, b)])
    diff = sectiondiff.diff_sections(y, SR, (2.0, 6.0), (12.0, 16.0))
    flags = sectiondiff.describe_diff(diff)

    assert any("presence 2.5-6k" in f and "filter cut" in f for f in flags), flags
    assert any("air 6-12k" in f and "filter cut" in f for f in flags), flags
    assert diff.centroid_b_hz < diff.centroid_a_hz


# --------------------------------------------------------------------------- #
# 5. lag alignment: a 0.3 s-late window compares clean WITH the lag, dirty without
# --------------------------------------------------------------------------- #


def test_lag_alignment_matters():
    s = make_section()
    # content planted at exactly 12.0 s, but B's WINDOW starts 0.3 s early
    y = make_song([(2.0, s), (12.0, s)])
    aligned = sectiondiff.diff_sections(y, SR, (2.0, 6.0), (11.7, 15.7), lag_s=0.3)
    assert sectiondiff.describe_diff(aligned) == []

    misaligned = sectiondiff.diff_sections(y, SR, (2.0, 6.0), (11.7, 15.7), lag_s=0.0)
    assert sectiondiff.describe_diff(misaligned) != []


# --------------------------------------------------------------------------- #
# 6. match_repeated_sections — grouping and discovery
# --------------------------------------------------------------------------- #


def test_candidates_group_repeats_and_isolate_unrelated():
    s = make_section()
    unrelated = _tone(880.0, SECTION_S, 0.3) + _tone(1320.0, SECTION_S, 0.2)
    # second repeat's content starts 0.3 s INTO its candidate window
    y = make_song([(2.0, s), (12.0, s), (18.0, unrelated)])
    groups = sectiondiff.match_repeated_sections(
        y, SR, candidates=[(2.0, 6.0), (11.7, 15.7), (18.0, 22.0)]
    )
    assert len(groups) == 2
    repeats = groups[0]
    assert len(repeats.members) == 2
    assert repeats.members[1].best_lag_s == pytest.approx(0.3, abs=0.03)
    assert repeats.correlation >= 0.6
    assert len(groups[1].members) == 1


def test_discovery_finds_planted_repeats():
    """Discovery slides fixed anchors, so a found group's WINDOWS may straddle the
    planted section edges — the honest discovered property is the repeat PERIOD: the
    two plants sit 10 s apart, and some group must contain two members 10 s apart."""
    s = make_section()
    y = make_song([(2.0, s), (12.0, s)])
    groups = sectiondiff.match_repeated_sections(y, SR, candidates=None, min_len_s=4.0)
    assert groups, "discovery must find the planted repeat"
    periods = [
        m.start_s - g.members[0].start_s for g in groups for m in g.members[1:]
    ]
    assert any(abs(p - 10.0) <= 0.5 for p in periods), (groups, periods)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


def test_empty_window_raises():
    y = make_song([(2.0, make_section())])
    with pytest.raises(ValueError, match="empty"):
        sectiondiff.diff_sections(y, SR, (100.0, 104.0), (2.0, 6.0))


def test_level_ride_alone_is_not_a_residual_layer():
    """A pure fader move (B = 1.3x A) must flag LEVEL, never a phantom 'new layer' —
    the per-window gain matching exists precisely so a ride doesn't read as content."""
    s = make_section()
    y = make_song([(2.0, s), (12.0, 1.3 * s)])
    diff = sectiondiff.diff_sections(y, SR, (2.0, 6.0), (12.0, 16.0))
    assert diff.residual_peaks == []
    flags = sectiondiff.describe_diff(diff)
    assert not any("residual layer" in f for f in flags)
