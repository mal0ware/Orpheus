"""``analysis/recon.py`` (§3.1 T1): decode-adjacent, CPU-only "recon" pass over a whole
track — LUFS, tempo/beat grid + downbeat/meter + grid confidence, tuning offset, key +
alternates, structure sections, and a coarse band-energy sketch.

Locked implementation choices (task 1.3 brief, do not re-decide here):
- tempo/beats via ``librosa.beat.beat_track``; downbeats/meter via onset-strength
  periodicity over beat groups of {3, 4}, argmax correlation picks the meter.
- tuning via ``librosa.estimate_tuning``; ALL chroma downstream is tuning-corrected
  (``tuning=`` param) — 432 Hz tapes and detuned recordings must not smear key detection.
- key via Krumhansl-Schmuckler profile correlation on tuning-corrected chroma; best + 2
  alternates.
- sections via self-similarity novelty (Foote checkerboard kernel) on beat-synchronous
  chroma; files under 30 s degrade to a single section (§T1 short-file rule).
- band sketch: 8 log-spaced bands 60 Hz-16 kHz, named ``"<lo>-<hi>"``.

librosa is imported lazily inside functions — the core server must import without it.
"""

from __future__ import annotations

import math

import numpy as np

from orpheus_mcp.analysis.cachefs import Pcm
from orpheus_mcp.models import BeatGrid, KeyEstimate, Recon, Window

_EPS = 1e-9
_BAND_FLOOR_DB = -120.0

# 8 log-spaced bands, 60 Hz-16 kHz (locked names, e.g. "60-120").
_BAND_EDGES_HZ = tuple(float(e) for e in np.geomspace(60.0, 16000.0, 9))

# §T1 short-file rule: below this duration, structure segmentation is skipped outright
# and the whole file is a single section (novelty over a handful of beats is noise).
_SECTION_MIN_DURATION_S = 30.0
_SECTION_MIN_BEATS = 8

# Krumhansl-Kessler key profiles (index 0 = tonic weight), rotated per candidate root.
_KS_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_TEMPO_MIN_BPM = 60.0
_TEMPO_MAX_BPM = 180.0
_DOWNBEAT_MIN_BEATS = 6
_DOWNBEAT_GROUPS = (3, 4)
_NOVELTY_KERNEL_RADIUS = 8

# §T1: below this many detected beats, the mean-inter-beat-interval tempo estimate is too
# noisy to trust over librosa's own scalar tempo (Task 1.3 brief, `_tempo_from_beats`).
_MIN_BEATS_FOR_TEMPO = 4

# Foote checkerboard-novelty peak-picking (librosa.util.peak_pick) — named per Task 1.3
# review (minor finding: bare literals at the call site). Kept as individual constants
# (not a single dict) so librosa's typed stub for peak_pick's per-arg int/bool params
# stays mypy-clean at the call site.
_PEAK_PICK_PRE_MAX = 3
_PEAK_PICK_POST_MAX = 3
_PEAK_PICK_PRE_AVG = 3
_PEAK_PICK_POST_AVG = 3
_PEAK_PICK_DELTA = 0.05
_PEAK_PICK_WAIT = 4


def run_recon(pcm: Pcm) -> Recon:
    """The T1 pass: everything ``recon_reference`` caches under kind ``"recon"``."""
    import librosa

    mono = _to_mono(pcm.samples)
    sr = pcm.sample_rate

    lufs = _integrated_lufs(pcm.samples, sr)

    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
    tempo_raw, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    tempo_bpm = _normalize_tempo(_tempo_from_beats(beat_times, float(np.atleast_1d(tempo_raw)[0])))

    meter, downbeat_idx, alignment_score, meter_confidence = _meter_and_confidence(
        beat_frames, onset_env
    )
    downbeats_s = [beat_times[i] for i in downbeat_idx if i < len(beat_times)]
    grid = BeatGrid(
        times_s=beat_times,
        downbeats_s=downbeats_s,
        meter=meter,
        confidence=alignment_score,
        meter_confidence=meter_confidence,
        kind="beat",
    )

    tuning = float(librosa.estimate_tuning(y=mono, sr=sr))
    tuning_offset_cents = tuning * 100.0

    key, key_alternates = _key_estimates(mono, sr, tuning)

    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, tuning=tuning)
    sections = _estimate_sections(chroma, sr, beat_frames, pcm.duration_s)
    band_energy_db = _band_energy_sketch(mono, sr)

    return Recon(
        duration_s=pcm.duration_s,
        sample_rate=sr,
        lufs_integrated=lufs,
        tempo_bpm=tempo_bpm,
        grid=grid,
        tuning_offset_cents=tuning_offset_cents,
        key=key,
        key_alternates=key_alternates,
        sections=sections,
        band_energy_db=band_energy_db,
    )


# --------------------------------------------------------------------------- #
# Mono downmix + LUFS (mirrors analysis/audio.py's shape, own copy — recon operates on
# the already-decoded Pcm, not a WAV path).
# --------------------------------------------------------------------------- #


def _to_mono(samples: np.ndarray) -> np.ndarray:
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    return np.ascontiguousarray(mono, dtype=np.float32)


def _integrated_lufs(samples: np.ndarray, sample_rate: int) -> float | None:
    import pyloudnorm

    data = samples.astype(np.float64)
    try:
        lufs = float(pyloudnorm.Meter(sample_rate).integrated_loudness(data))
    except ValueError:
        return None  # shorter than one 400 ms gating block
    return lufs if math.isfinite(lufs) else None  # silence gates to -inf


# --------------------------------------------------------------------------- #
# Tempo octave normalization (standard practice — beat trackers commonly halve/double).
# --------------------------------------------------------------------------- #


def _tempo_from_beats(beat_times: list[float], fallback_bpm: float) -> float:
    """The mean inter-beat interval, not librosa's own scalar tempo estimate: the DP beat
    tracker's frame-hop quantization makes consecutive intervals alternate a frame short/
    long around the true period (e.g. 21 vs 22 frames for a period that isn't an integer
    number of hops) — the scalar tempo picks up that bias, the mean interval cancels it.
    Falls back to librosa's own estimate when there are too few beats to average."""
    if len(beat_times) < _MIN_BEATS_FOR_TEMPO:
        return fallback_bpm
    diffs = np.diff(beat_times)
    mean_interval = float(np.mean(diffs))
    if mean_interval <= _EPS:
        return fallback_bpm
    return 60.0 / mean_interval


def _normalize_tempo(bpm: float) -> float:
    if bpm <= 0:
        return bpm
    while bpm >= _TEMPO_MAX_BPM:
        bpm /= 2.0
    while bpm < _TEMPO_MIN_BPM:
        bpm *= 2.0
    return bpm


# --------------------------------------------------------------------------- #
# Downbeats/meter: onset-strength periodicity over beat groups {3, 4}, argmax correlation.
# Grid confidence = mean beat-onset alignment score (onset strength at each detected beat,
# normalized by the track's peak onset strength), independent of the meter pick.
#
# meter_confidence (D10, controller decision) is a SEPARATE signal from grid confidence: the
# margin ratio between the winning {3,4}-group's best-phase correlation and the runner-up's,
# i.e. `(best - second) / max(best, eps)`, bounded [0, 1]. A rock-solid grid (every beat
# lands exactly on an onset) can still have an ambiguous METER (grouping by 3 vs 4 scores
# nearly identically, e.g. an unaccented uniform click) — the two fields must not be
# conflated into one.
# --------------------------------------------------------------------------- #


def _meter_and_confidence(
    beat_frames: np.ndarray, onset_env: np.ndarray
) -> tuple[tuple[int, int], list[int], float, float]:
    n = len(beat_frames)
    if n == 0 or len(onset_env) == 0:
        return (4, 4), [], 0.0, 0.0

    idx = np.clip(beat_frames, 0, len(onset_env) - 1)
    strengths = onset_env[idx]
    peak = float(onset_env.max())
    alignment_score = float(np.clip(np.mean(strengths) / peak, 0.0, 1.0)) if peak > _EPS else 0.0

    if n < _DOWNBEAT_MIN_BEATS:
        return (4, 4), [], alignment_score, 0.0

    best_group = _DOWNBEAT_GROUPS[0]
    best_phase = 0
    best_score = -np.inf
    group_best_scores: dict[int, float] = {}
    for group in _DOWNBEAT_GROUPS:
        group_score = -np.inf
        group_phase = 0
        for phase in range(group):
            phase_idx = np.arange(phase, n, group)
            score = float(np.mean(strengths[phase_idx]))
            if score > group_score:
                group_score = score
                group_phase = phase
        group_best_scores[group] = group_score
        if group_score > best_score:
            best_score = group_score
            best_group = group
            best_phase = group_phase

    ranked_scores = sorted(group_best_scores.values(), reverse=True)
    best = ranked_scores[0]
    second = ranked_scores[1] if len(ranked_scores) > 1 else ranked_scores[0]
    meter_confidence = float(np.clip((best - second) / max(best, _EPS), 0.0, 1.0))

    downbeat_idx = list(range(best_phase, n, best_group))
    return (best_group, 4), downbeat_idx, alignment_score, meter_confidence


# --------------------------------------------------------------------------- #
# Key: Krumhansl-Schmuckler on tuning-corrected chroma — global (whole-track) reading,
# best + 2 alternates by correlation. Harmonic-percussive separated (drums/onsets would
# otherwise leak broadband energy into every chroma bin and wash out the tonal profile)
# and per-pitch MEDIAN rather than mean across frames — a per-frame-normalized mean lets a
# long sustained chord dominate the profile as much as several shorter ones combined;
# the median is far less sensitive to that duration skew.
# --------------------------------------------------------------------------- #


def _key_estimates(
    mono: np.ndarray, sr: int, tuning: float
) -> tuple[KeyEstimate, list[KeyEstimate]]:
    import librosa

    harmonic = librosa.effects.harmonic(mono)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, tuning=tuning, norm=None)
    mean_chroma = np.median(chroma, axis=1)
    candidates: list[tuple[float, str, str]] = []
    for mode_name, profile in (("major", _KS_MAJOR_PROFILE), ("minor", _KS_MINOR_PROFILE)):
        for root_idx in range(12):
            rotated = np.roll(profile, root_idx)
            corr = _safe_corrcoef(mean_chroma, rotated)
            candidates.append((corr, _PITCH_CLASSES[root_idx], mode_name))
    candidates.sort(key=lambda c: c[0], reverse=True)

    def to_estimate(c: tuple[float, str, str]) -> KeyEstimate:
        corr, root, mode = c
        return KeyEstimate(root=root, mode=mode, confidence=float(min(1.0, max(0.0, corr))))

    return to_estimate(candidates[0]), [to_estimate(c) for c in candidates[1:3]]


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < _EPS or np.std(b) < _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Sections: self-similarity novelty (Foote checkerboard kernel) over beat-synchronous
# tuning-corrected chroma. Short files (< 30 s) or too-sparse beat grids skip straight to
# a single section — novelty over a handful of beats is noise, not structure.
# --------------------------------------------------------------------------- #


def _estimate_sections(
    chroma: np.ndarray, sr: int, beat_frames: np.ndarray, duration_s: float
) -> list[Window]:
    whole_file = [Window(start_s=0.0, end_s=duration_s)]
    if duration_s < _SECTION_MIN_DURATION_S or len(beat_frames) < _SECTION_MIN_BEATS:
        return whole_file

    import librosa

    chroma_sync = librosa.util.sync(chroma, beat_frames.tolist(), aggregate=np.median)
    if chroma_sync.shape[1] < _SECTION_MIN_BEATS:
        return whole_file

    recurrence = librosa.segment.recurrence_matrix(chroma_sync, mode="affinity", sym=True)
    novelty = _checkerboard_novelty(np.asarray(recurrence, dtype=np.float64))
    if novelty.max(initial=0.0) <= _EPS:
        return whole_file

    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=_PEAK_PICK_PRE_MAX,
        post_max=_PEAK_PICK_POST_MAX,
        pre_avg=_PEAK_PICK_PRE_AVG,
        post_avg=_PEAK_PICK_POST_AVG,
        delta=_PEAK_PICK_DELTA,
        wait=_PEAK_PICK_WAIT,
    )
    if len(peaks) == 0:
        return whole_file

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    boundary_times = sorted(
        {0.0, float(duration_s), *(float(beat_times[p]) for p in peaks if p < len(beat_times))}
    )
    windows = [
        Window(start_s=boundary_times[i], end_s=boundary_times[i + 1])
        for i in range(len(boundary_times) - 1)
    ]
    return windows or whole_file


def _checkerboard_novelty(
    recurrence: np.ndarray, radius: int = _NOVELTY_KERNEL_RADIUS
) -> np.ndarray:
    """Foote's checkerboard-kernel novelty curve over a self-similarity matrix."""
    n = recurrence.shape[0]
    novelty = np.zeros(n)
    if n < 4:
        return novelty

    r = max(2, min(radius, n // 2 - 1))
    axis = np.arange(-r, r) + 0.5  # avoid a zero row/col at the kernel center
    gauss = np.exp(-0.5 * (axis / (r / 2.0)) ** 2)
    checker = np.sign(np.outer(axis, axis))
    kernel = checker * np.outer(gauss, gauss)

    for t in range(r, n - r):
        block = recurrence[t - r : t + r, t - r : t + r]
        novelty[t] = float(np.sum(block * kernel))

    lo, hi = novelty.min(), novelty.max()
    if hi > lo:
        novelty = (novelty - lo) / (hi - lo)
    return novelty


# --------------------------------------------------------------------------- #
# Band-energy sketch: 8 log-spaced bands, 60 Hz-16 kHz.
# --------------------------------------------------------------------------- #


def _band_energy_sketch(mono: np.ndarray, sample_rate: int) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(mono)) / max(len(mono), 1)
    power = spectrum**2
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)

    bands: dict[str, float] = {}
    for lo, hi in zip(_BAND_EDGES_HZ, _BAND_EDGES_HZ[1:], strict=False):
        label = f"{int(lo)}-{int(hi)}"
        band_power = float(power[(freqs >= lo) & (freqs < hi)].sum())
        # x2: rfft folds negative frequencies; RMS^2 == total spectral power (Parseval).
        linear_rms = math.sqrt(2.0 * band_power)
        db = 20.0 * math.log10(linear_rms) if linear_rms > _EPS else None
        bands[label] = _BAND_FLOOR_DB if db is None else max(db, _BAND_FLOOR_DB)
    return bands
