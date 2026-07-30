"""Differential section analysis ("section-diff") — the minute-details pass.

The broad-brush pipeline (recon/timeline/chords/descriptors) characterizes a song
globally. This module answers a different question: *what changed between two
repetitions of the same section?* — the production details that make a song special:
a background layer entering in the second chorus, a bassline pattern change in a
unique section, a filter/EQ move, a level ride.

Method: align repetition B onto repetition A (an explicit lag, since sung/played
repeats rarely land sample-exact), then measure per-dimension deltas — band-energy
shares, onset densities (broadband + a low-band bass-pattern proxy), RMS, spectral
centroid — plus a **residual-layer score**: after per-window gain-matching A onto B,
the energy of ``B − g·A`` relative to B, whose peaks localize WHERE new material
enters that A does not explain.

Pure numpy — no torch, no model stack, no cache writes. Fast enough to run on every
section pair of a song in the base env.

Honest limits (v1): the residual score compares aligned SAMPLES, so it localizes new
layers well on tightly-repeated (loop-based / programmed) material but degrades on
performances with free timing drift inside the section; content REMOVED from B also
shows up as residual (the score finds "unexplained difference", not strictly
"addition" — the band deltas disambiguate direction). Tempo *bends* between
repetitions are not warped away — only a constant lag is compensated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SectionMember",
    "SectionGroup",
    "ResidualPeak",
    "SectionDiff",
    "match_repeated_sections",
    "diff_sections",
    "describe_diff",
]

# 10 ms envelope grid — fine enough to localize a lag to a beat subdivision, coarse
# enough that a whole song's envelope is a few thousand frames.
_HOP_S = 0.010
_ENV_FLOOR_DB = -60.0

# The §8-style band split the rest of the analysis stack speaks (compare_character's
# spectral bands) — deltas here must be phrased in the same vocabulary.
_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub<80", 20.0, 80.0),
    ("low 80-250", 80.0, 250.0),
    ("lowmid 250-800", 250.0, 800.0),
    ("mid 800-2.5k", 800.0, 2500.0),
    ("presence 2.5-6k", 2500.0, 6000.0),
    ("air 6-12k", 6000.0, 12000.0),
)

_FFT_N = 4096
_FLUX_N = 1024
_FLUX_HOP = 512
_LOW_ONSET_MAX_HZ = 250.0
_RESIDUAL_WIN_S = 1.0
_RESIDUAL_HOP_S = 0.5
# A residual window quieter than this (relative to B) is "A explains B here".
_RESIDUAL_FLAG_DB = -6.0
_RESIDUAL_FLOOR_DB = -80.0
_MAX_RESIDUAL_PEAKS = 3
# Onset-density flags need BOTH a ratio and an absolute change — a 0.1/s -> 0.2/s
# "doubling" is noise, not a pattern change.
_ONSET_RATIO_FLAG = 1.5
_ONSET_ABS_FLAG = 0.5
_CENTROID_REL_FLAG = 0.20


@dataclass(frozen=True)
class SectionMember:
    """One occurrence of a repeated section: its window and the lag (seconds INTO the
    window where the group-anchor-aligned content starts — 0.0 for the anchor itself)."""

    start_s: float
    end_s: float
    best_lag_s: float


@dataclass(frozen=True)
class SectionGroup:
    """A group of section windows judged to be repetitions of the same material.
    ``correlation`` is the mean envelope correlation of the non-anchor members against
    the anchor (1.0 for a single-member group, degenerate but explicit)."""

    members: list[SectionMember]
    correlation: float


@dataclass(frozen=True)
class ResidualPeak:
    """One ~1 s window where B carries energy A does not explain. ``offset_s`` is into
    the section (0 == section start); ``residual_db`` is the unexplained energy relative
    to B's own energy in that window (0 dB == B is entirely new material there)."""

    offset_s: float
    residual_db: float


@dataclass(frozen=True)
class SectionDiff:
    """The measured deltas of section B relative to section A (positive == more in B)."""

    a: tuple[float, float]
    b: tuple[float, float]
    lag_s: float
    band_deltas_db: dict[str, float]  # keyed by the _BANDS names; delta of band SHARE
    rms_delta_db: float
    centroid_a_hz: float
    centroid_b_hz: float
    onset_density_a: float  # broadband spectral-flux onsets per second
    onset_density_b: float
    low_onset_density_a: float  # <250 Hz onsets per second — the bassline proxy
    low_onset_density_b: float
    residual_peaks: list[ResidualPeak]  # loudest first, at most _MAX_RESIDUAL_PEAKS


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _mono(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def _envelope_db(y: np.ndarray, sr: int) -> np.ndarray:
    """10 ms RMS envelope in dB, floored at ``_ENV_FLOOR_DB`` so digital silence does
    not dominate a correlation with -inf outliers."""
    hop = max(1, int(round(_HOP_S * sr)))
    n = len(y) // hop
    if n == 0:
        return np.full(1, _ENV_FLOOR_DB)
    frames = y[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return np.maximum(db, _ENV_FLOOR_DB)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized (Pearson) correlation of two equal-length windows; 0.0 when either
    side is flat (silence against silence must not read as a perfect repeat)."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _best_lag(
    env: np.ndarray, a_f: int, b_f: int, length_f: int, max_lag_f: int
) -> tuple[int, float]:
    """Scan integer-frame lags in ``[-max_lag_f, +max_lag_f]`` for the lag maximizing the
    envelope correlation of ``env[a_f : a_f+length_f]`` against ``env[b_f+lag : ...]``.
    Out-of-bounds lags are skipped, not clamped (a clamped window would correlate
    misaligned content and could still win)."""
    ref = env[a_f : a_f + length_f]
    best_lag, best_corr = 0, -1.0
    for lag in range(-max_lag_f, max_lag_f + 1):
        lo = b_f + lag
        if lo < 0 or lo + length_f > len(env):
            continue
        c = _ncc(ref, env[lo : lo + length_f])
        if c > best_corr:
            best_lag, best_corr = lag, c
    return best_lag, best_corr


# --------------------------------------------------------------------------- #
# match_repeated_sections
# --------------------------------------------------------------------------- #


def match_repeated_sections(
    y: np.ndarray,
    sr: int,
    candidates: list[tuple[float, float]] | None = None,
    min_len_s: float = 4.0,
    corr_threshold: float = 0.6,
) -> list[SectionGroup]:
    """Group repeated sections by 10 ms-envelope correlation.

    With ``candidates`` (start_s, end_s windows — e.g. from markers or recon sections),
    each candidate joins the first existing group whose ANCHOR (first member) it
    correlates with at >= ``corr_threshold`` under a ±0.5 s lag search; otherwise it
    founds a new group. Windows shorter than ``min_len_s`` are ignored.

    With ``candidates=None``, repeats are DISCOVERED by envelope self-similarity: fixed
    ``min_len_s`` anchor windows slide at half-window stride; each un-covered anchor is
    correlated against every later position (vectorized sliding windows) and positions
    scoring >= ``corr_threshold`` (local peaks, at least one window apart) become
    members. Only groups with >= 2 members are returned — deliberately simple, honest
    discovery, not a structure segmenter.
    """
    mono = _mono(y)
    env = _envelope_db(mono, sr)
    max_lag_f = int(round(0.5 / _HOP_S))

    if candidates is not None:
        min_len_f = int(round(min_len_s / _HOP_S))
        groups: list[dict] = []  # {"anchor_f", "len_f", "members", "corrs"}
        for start_s, end_s in candidates:
            s_f = int(round(start_s / _HOP_S))
            e_f = int(round(end_s / _HOP_S))
            length_f = e_f - s_f
            if length_f < min_len_f or s_f < 0 or e_f > len(env):
                continue
            placed = False
            for g in groups:
                cmp_len = min(g["len_f"], length_f)
                lag_f, corr = _best_lag(env, g["anchor_f"], s_f, cmp_len, max_lag_f)
                if corr >= corr_threshold:
                    g["members"].append(
                        SectionMember(start_s=start_s, end_s=end_s, best_lag_s=lag_f * _HOP_S)
                    )
                    g["corrs"].append(corr)
                    placed = True
                    break
            if not placed:
                groups.append(
                    {
                        "anchor_f": s_f,
                        "len_f": length_f,
                        "members": [
                            SectionMember(start_s=start_s, end_s=end_s, best_lag_s=0.0)
                        ],
                        "corrs": [],
                    }
                )
        return [
            SectionGroup(
                members=g["members"],
                correlation=float(np.mean(g["corrs"])) if g["corrs"] else 1.0,
            )
            for g in groups
        ]

    # -- discovery mode -------------------------------------------------- #
    length_f = int(round(min_len_s / _HOP_S))
    if length_f < 2 or len(env) < 2 * length_f:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(env, length_f)
    centered = windows - windows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    covered = np.zeros(len(env), dtype=bool)
    out: list[SectionGroup] = []
    for a_f in range(0, len(env) - length_f + 1, length_f // 2):
        if covered[a_f : a_f + length_f].any():
            continue
        anchor = centered[a_f]
        anchor_norm = norms[a_f]
        if anchor_norm < 1e-9:
            continue  # silence anchors match nothing
        corr = (centered @ anchor) / np.maximum(norms * anchor_norm, 1e-12)
        corr[max(0, a_f - length_f + 1) : a_f + length_f] = -1.0  # exclude self-overlap
        members = [SectionMember(a_f * _HOP_S, (a_f + length_f) * _HOP_S, 0.0)]
        member_corrs: list[float] = []
        taken = covered.copy()
        taken[a_f : a_f + length_f] = True
        order = np.argsort(corr)[::-1]
        for p in order:
            if corr[p] < corr_threshold:
                break
            if taken[p : p + length_f].any():
                continue
            members.append(SectionMember(p * _HOP_S, (p + length_f) * _HOP_S, 0.0))
            member_corrs.append(float(corr[p]))
            taken[p : p + length_f] = True
        if len(members) >= 2:
            members.sort(key=lambda m: m.start_s)
            out.append(SectionGroup(members=members, correlation=float(np.mean(member_corrs))))
            covered = taken
    return out


# --------------------------------------------------------------------------- #
# diff_sections
# --------------------------------------------------------------------------- #


def _band_shares_and_centroid(seg: np.ndarray, sr: int) -> tuple[dict[str, float], float]:
    """Per-band POWER SHARES in dB (band / broadband — level-independent, so a fader
    difference between repetitions doesn't masquerade as a filter move) plus the
    power-spectrum centroid in Hz."""
    n = _FFT_N
    if len(seg) < n:
        n = max(256, 1 << int(np.floor(np.log2(max(len(seg), 257)))))
    window = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    count = 0
    for i in range(0, len(seg) - n + 1, n // 2):
        acc += np.abs(np.fft.rfft(seg[i : i + n] * window)) ** 2
        count += 1
    if count == 0:
        acc += 1e-18
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = float(acc.sum()) + 1e-18
    shares = {}
    for name, lo, hi in _BANDS:
        m = (freqs >= lo) & (freqs < hi)
        shares[name] = 10.0 * np.log10(float(acc[m].sum()) / total + 1e-12)
    centroid = float(np.sum(freqs * acc) / total)
    return shares, centroid


def _onset_density(seg: np.ndarray, sr: int, *, max_hz: float | None = None) -> float:
    """Spectral-flux onset density (onsets/second). ``max_hz`` restricts the flux to
    bins below it — the <250 Hz variant is the bassline-pattern proxy. Peak picking:
    local maxima above mean + 1 std of the flux, at least 100 ms apart."""
    n, hop = _FLUX_N, _FLUX_HOP
    if len(seg) < 2 * n:
        return 0.0
    window = np.hanning(n)
    frames = []
    for i in range(0, len(seg) - n + 1, hop):
        frames.append(np.abs(np.fft.rfft(seg[i : i + n] * window)))
    mags = np.array(frames)
    if max_hz is not None:
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mags = mags[:, freqs <= max_hz]
    flux = np.sum(np.maximum(mags[1:] - mags[:-1], 0.0), axis=1)
    if len(flux) == 0:
        return 0.0
    thresh = flux.mean() + flux.std()
    min_gap = max(1, int(round(0.1 * sr / hop)))
    onsets = 0
    last = -min_gap
    for i in range(1, len(flux) - 1):
        is_peak = flux[i] >= thresh and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]
        if is_peak and i - last >= min_gap:
            onsets += 1
            last = i
    return onsets / (len(seg) / sr)


def _residual_peaks(a: np.ndarray, b: np.ndarray, sr: int) -> list[ResidualPeak]:
    """Per-window gain-matched residual: for each ~1 s window, the g minimizing
    ``||b − g·a||`` is applied and the residual energy is reported relative to B's own
    energy there. Per-window (not global) gain matching keeps a level RIDE from reading
    as a new layer — only content A cannot explain at any gain survives."""
    win = max(1, int(round(_RESIDUAL_WIN_S * sr)))
    hop = max(1, int(round(_RESIDUAL_HOP_S * sr)))
    peaks: list[ResidualPeak] = []
    for i in range(0, len(a) - win + 1, hop):
        wa, wb = a[i : i + win], b[i : i + win]
        ea = float(np.dot(wa, wa))
        eb = float(np.dot(wb, wb)) + 1e-18
        g = float(np.dot(wa, wb)) / ea if ea > 1e-18 else 0.0
        resid = wb - g * wa
        er = float(np.dot(resid, resid))
        db = 10.0 * np.log10(er / eb + 1e-12)
        peaks.append(ResidualPeak(offset_s=i / sr, residual_db=max(db, _RESIDUAL_FLOOR_DB)))
    flagged = [p for p in peaks if p.residual_db >= _RESIDUAL_FLAG_DB]
    flagged.sort(key=lambda p: p.residual_db, reverse=True)
    return flagged[:_MAX_RESIDUAL_PEAKS]


def diff_sections(
    y: np.ndarray,
    sr: int,
    a: tuple[float, float],
    b: tuple[float, float],
    lag_s: float = 0.0,
) -> SectionDiff:
    """Measure section B against section A. ``lag_s`` is seconds INTO b's window where
    the content aligned with a's start actually begins (the ``best_lag_s`` a
    :class:`SectionMember` carries) — B's slice starts at ``b.start + lag_s``. Both
    slices are truncated to the shorter aligned length; every delta is B − A
    (positive == more of it in B)."""
    mono = _mono(y)
    a0 = int(round(a[0] * sr))
    a1 = int(round(a[1] * sr))
    b0 = int(round((b[0] + lag_s) * sr))
    b1 = int(round(b[1] * sr))
    seg_a = mono[max(0, a0) : max(0, a1)]
    seg_b = mono[max(0, b0) : max(0, b1)]
    n = min(len(seg_a), len(seg_b))
    if n == 0:
        raise ValueError("diff_sections: a section window is empty after alignment")
    seg_a, seg_b = seg_a[:n], seg_b[:n]

    shares_a, centroid_a = _band_shares_and_centroid(seg_a, sr)
    shares_b, centroid_b = _band_shares_and_centroid(seg_b, sr)
    rms_a = float(np.sqrt(np.mean(seg_a**2))) + 1e-12
    rms_b = float(np.sqrt(np.mean(seg_b**2))) + 1e-12

    return SectionDiff(
        a=a,
        b=b,
        lag_s=lag_s,
        band_deltas_db={name: shares_b[name] - shares_a[name] for name, _, _ in _BANDS},
        rms_delta_db=20.0 * np.log10(rms_b / rms_a),
        centroid_a_hz=centroid_a,
        centroid_b_hz=centroid_b,
        onset_density_a=_onset_density(seg_a, sr),
        onset_density_b=_onset_density(seg_b, sr),
        low_onset_density_a=_onset_density(seg_a, sr, max_hz=_LOW_ONSET_MAX_HZ),
        low_onset_density_b=_onset_density(seg_b, sr, max_hz=_LOW_ONSET_MAX_HZ),
        residual_peaks=_residual_peaks(seg_a, seg_b, sr),
    )


# --------------------------------------------------------------------------- #
# describe_diff
# --------------------------------------------------------------------------- #


def _onset_flag(da: float, db_: float) -> bool:
    lo, hi = min(da, db_), max(da, db_)
    if hi - lo < _ONSET_ABS_FLAG:
        return False
    return lo <= 1e-9 or (hi / max(lo, 1e-9)) >= _ONSET_RATIO_FLAG


def describe_diff(diff: SectionDiff, threshold_db: float = 3.0) -> list[str]:
    """Human-readable anomaly flags, one per dimension over threshold. An EMPTY list is
    a real measurement ("repetition B measured the same as A"), so every threshold errs
    quiet — a flag must mean something audible changed."""
    flags: list[str] = []
    for name, delta in diff.band_deltas_db.items():
        if delta >= threshold_db:
            flags.append(
                f"{name} band {delta:+.1f} dB in repetition B "
                "(possible added layer or filter move)"
            )
        elif delta <= -threshold_db:
            flags.append(
                f"{name} band {delta:+.1f} dB in repetition B "
                "(possible removed layer or filter cut)"
            )
    if abs(diff.rms_delta_db) >= threshold_db:
        flags.append(f"overall level {diff.rms_delta_db:+.1f} dB in repetition B")
    if _onset_flag(diff.low_onset_density_a, diff.low_onset_density_b):
        flags.append(
            f"low-band onset density {diff.low_onset_density_a:.1f}/s -> "
            f"{diff.low_onset_density_b:.1f}/s (bassline pattern change)"
        )
    elif _onset_flag(diff.onset_density_a, diff.onset_density_b):
        # broadband-only change (low band steady): rhythmic density moved elsewhere
        flags.append(
            f"onset density {diff.onset_density_a:.1f}/s -> "
            f"{diff.onset_density_b:.1f}/s (rhythmic pattern change)"
        )
    if diff.centroid_a_hz > 1e-9:
        rel = abs(diff.centroid_b_hz - diff.centroid_a_hz) / diff.centroid_a_hz
        if rel >= _CENTROID_REL_FLAG:
            direction = "brighter" if diff.centroid_b_hz > diff.centroid_a_hz else "darker"
            flags.append(
                f"spectral centroid {diff.centroid_a_hz:.0f} Hz -> "
                f"{diff.centroid_b_hz:.0f} Hz ({direction} — filter/EQ move?)"
            )
    for peak in diff.residual_peaks:
        flags.append(
            f"residual layer at +{peak.offset_s:.1f}s into the section "
            f"(new element enters, residual {peak.residual_db:+.1f} dB vs repetition B)"
        )
    return flags
