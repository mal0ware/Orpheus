"""``analysis/harmony.py`` (§13): the always-available baseline chord lane — no torch,
ships in ``[analysis]``. Beat-synchronous template matching over HPSS-harmonic chroma with
Viterbi smoothing, then three refinement layers: §13.3 extensions, §13.2 bass-informed
slash/inversion detection, §13.4 music21 theory (local-key windows + Roman numerals).

Locked algorithm (task 3.1 brief — do not re-decide):
- Vocabulary: 12 roots x {maj, min, 7, maj7, min7, sus2, sus4, dim, aug} + ``N`` = 109
  states. Binary 12-d templates, L2-normalized; emission = cosine(chroma_beat, template);
  the ``N`` (no-chord) template is a flat 1/12 vector, its emission scaled 0.85 so it only
  wins on genuinely flat/quiet beats.
- Viterbi: self-transition ``CFG.CHORD_SELF_TRANSITION``, the remainder spread uniformly
  across the other 108 states; log-space.
- §13.3 extensions / §13.2 slash / §13.4 theory: see each function's docstring.

The recognizer core (templates + Viterbi + segmentation) is a pure function of a
``(beats, 12)`` chroma matrix — audio only enters at ``beat_chroma`` / ``bass_pc_lane``, so
the constructed-chroma unit tests never touch librosa.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from orpheus_mcp.analysis.cachefs import Pcm
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import BeatGrid, ChordSegment, ChordsLane, Recon

_EPS = 1e-9

_PITCH_CLASSES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)  # fmt: skip
_ROOT_INDEX: dict[str, int] = {name: i for i, name in enumerate(_PITCH_CLASSES)}

# The §13.1 vocabulary — intervals (semitones above the root) present in each quality.
_QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
}
_QUALITIES: tuple[str, ...] = tuple(_QUALITY_INTERVALS)
_N_STATE = len(_QUALITIES) * 12  # 108 — the flat no-chord state sits last
_N_STATES = _N_STATE + 1  # 109

# Label suffix per quality (root prepended, extensions/bass appended by _compose_label).
_QUALITY_SUFFIX: dict[str, str] = {
    "maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7",
    "sus2": "sus2", "sus4": "sus4", "dim": "dim", "aug": "aug", "N": "N",
}  # fmt: skip

# §13.3 candidate extensions: (interval-above-root, extension name), in label order. An
# interval already in the base triad's template is skipped (it is a chord tone, not an
# extension). "6" is rewritten to "13" when a seventh is also present (a sixth over a
# seventh chord is the thirteenth).
_EXTENSION_CANDIDATES: tuple[tuple[int, str], ...] = (
    (11, "maj7"),
    (10, "7"),
    (2, "add9"),
    (9, "6"),
    (6, "#11"),
)

# Krumhansl-Kessler profiles (tonic-first), for the §13.4 local-key windows.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


# --------------------------------------------------------------------------- #
# Templates + emission + Viterbi — the pure recognizer core.
# --------------------------------------------------------------------------- #


def _template_matrix() -> np.ndarray:
    """The (109, 12) L2-normalized template matrix. Rows 0..107 are binary chord templates
    (state = root*9 + quality index); row 108 is the flat no-chord template."""
    templates = np.zeros((_N_STATES, 12), dtype=np.float64)
    for root in range(12):
        for q_idx, quality in enumerate(_QUALITIES):
            row = root * len(_QUALITIES) + q_idx
            for interval in _QUALITY_INTERVALS[quality]:
                templates[row, (root + interval) % 12] = 1.0
    templates[_N_STATE, :] = 1.0 / 12.0
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    return templates / np.clip(norms, _EPS, None)


_TEMPLATES = _template_matrix()


def _emission(chroma: np.ndarray) -> np.ndarray:
    """Per-beat cosine(chroma, template) over the 109 chord states, sharpened by
    ``CFG.CHORD_EMISSION_SHARPNESS`` (a temperature) and with an energy-gated no-chord state.

    Why the sharpness exponent: the transition model spreads ``1 - CHORD_SELF_TRANSITION``
    uniformly across the *other 108* states, so any chord change costs ~log(0.1/108) ≈ -7
    nats. Raw cosine emissions sit in a narrow [~0.5, 1.0] band (log-differences ~0.1), so
    without sharpening the winning path never switches — every track collapses to one chord.
    Raising cosine to the 8th power restores the dynamic range a discriminative emission
    needs while leaving the argmax (the recognized chord) untouched. The ``N`` (no-chord)
    state is scored purely by beat energy — a flat 1/12 template has high cosine with *any*
    smeared chroma, so cosine-scoring ``N`` would make it dominate loud, tonally-rich beats;
    instead ``N`` wins only on genuinely quiet beats (energy below
    ``CHORD_QUIET_ENERGY_RATIO`` x the track's median beat energy), which is exactly the
    "beats quiet frames only" intent."""
    norms = np.linalg.norm(chroma, axis=1)
    unit = chroma / np.clip(norms[:, None], _EPS, None)
    cos = np.clip(unit @ _TEMPLATES[:_N_STATE].T, 0.0, None)  # (beats, 108), in [0, 1]

    emis = np.zeros((chroma.shape[0], _N_STATES), dtype=np.float64)
    emis[:, :_N_STATE] = cos ** float(CFG.CHORD_EMISSION_SHARPNESS)

    active = norms > _EPS
    median_energy = float(np.median(norms[active])) if active.any() else 0.0
    quiet = norms < CFG.CHORD_QUIET_ENERGY_RATIO * median_energy
    emis[quiet, :_N_STATE] = 0.0
    emis[quiet, _N_STATE] = 0.85  # the flat no-chord template (scaled 0.85), for quiet beats
    return emis


def _viterbi(chroma: np.ndarray) -> np.ndarray:
    """Log-space Viterbi over the 109-state chain. The transition matrix has
    ``CFG.CHORD_SELF_TRANSITION`` on the diagonal and the remaining mass spread uniformly
    across the other 108 states — so the per-step choice is only ever "stay" vs "jump to
    the globally best predecessor", which collapses the O(S^2) recursion to O(S)."""
    emis = _emission(chroma)
    n_frames = emis.shape[0]
    if n_frames == 0:
        return np.empty(0, dtype=np.int64)

    log_emis = np.log(emis + _EPS)
    p_stay = float(CFG.CHORD_SELF_TRANSITION)
    log_stay = math.log(p_stay)
    log_jump = math.log((1.0 - p_stay) / (_N_STATES - 1))

    delta = log_emis[0].copy()
    psi = np.zeros((n_frames, _N_STATES), dtype=np.int64)
    for t in range(1, n_frames):
        best_prev = int(np.argmax(delta))
        jump_score = float(delta[best_prev]) + log_jump
        stay_score = delta + log_stay
        take_stay = stay_score >= jump_score
        psi[t] = np.where(take_stay, np.arange(_N_STATES), best_prev)
        delta = np.where(take_stay, stay_score, jump_score) + log_emis[t]

    states = np.zeros(n_frames, dtype=np.int64)
    states[-1] = int(np.argmax(delta))
    for t in range(n_frames - 1, 0, -1):
        states[t - 1] = psi[t, states[t]]
    return states


def _state_to_chord(state: int) -> tuple[str, str]:
    """(root name, quality) for a Viterbi state; the no-chord state maps to ("N", "N")."""
    if state == _N_STATE:
        return "N", "N"
    return _PITCH_CLASSES[state // len(_QUALITIES)], _QUALITIES[state % len(_QUALITIES)]


# --------------------------------------------------------------------------- #
# Chroma extraction (audio in) — beat-synchronous and fixed 1 s windows.
# --------------------------------------------------------------------------- #


def _to_mono(samples: np.ndarray) -> np.ndarray:
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    return np.ascontiguousarray(mono, dtype=np.float32)


def _chroma_and_times(pcm: Pcm, tuning_cents: float) -> tuple[np.ndarray, np.ndarray]:
    """HPSS-harmonic CQT chroma (12, frames) + per-frame times. ``tuning_cents`` is the
    recon tuning offset in cents; ``chroma_cqt`` takes tuning in fractional chroma bins
    (1 bin == 1 semitone == 100 cents), so the conversion is a straight /100."""
    import librosa

    mono = _to_mono(pcm.samples)
    sr = pcm.sample_rate
    harmonic = librosa.effects.harmonic(mono)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, tuning=tuning_cents / 100.0)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)
    return chroma, frame_times


def _aggregate(
    chroma: np.ndarray, frame_times: np.ndarray, boundaries_s: np.ndarray, end_s: float
) -> np.ndarray:
    """Median-aggregate a (12, frames) chroma into one (n, 12) row per interval
    ``[boundaries[i], boundaries[i+1])`` (the last interval runs to ``end_s``). Intervals
    with no frames stay all-zero — quiet spans the recognizer reads as no-chord."""
    n = len(boundaries_s)
    out = np.zeros((n, 12), dtype=np.float64)
    for i in range(n):
        lo = boundaries_s[i]
        hi = boundaries_s[i + 1] if i + 1 < n else end_s
        mask = (frame_times >= lo) & (frame_times < hi)
        if mask.any():
            out[i] = np.median(chroma[:, mask], axis=1)
    return out


def beat_chroma(pcm: Pcm, grid: BeatGrid, tuning_cents: float) -> np.ndarray:
    """(beats, 12) HPSS-harmonic chroma, one row per beat interval on ``grid.times_s``."""
    chroma, frame_times = _chroma_and_times(pcm, tuning_cents)
    return _aggregate(chroma, frame_times, np.asarray(grid.times_s, dtype=float), pcm.duration_s)


def _fixed_window_chroma(pcm: Pcm, tuning_cents: float) -> tuple[np.ndarray, np.ndarray]:
    """(windows, 12) chroma on fixed 1 s windows + the window start times (§13.4 rubato
    fallback). One window per whole second of the track, plus a trailing partial second."""
    chroma, frame_times = _chroma_and_times(pcm, tuning_cents)
    n_win = max(1, int(math.ceil(pcm.duration_s)))
    starts = np.arange(n_win, dtype=float)
    return _aggregate(chroma, frame_times, starts, pcm.duration_s), starts


# --------------------------------------------------------------------------- #
# Segmentation + the public baseline lane.
# --------------------------------------------------------------------------- #


def _compose_label(root: str, quality: str, extensions: list[str], bass: str | None) -> str:
    if quality == "N":
        return "N"
    label = root + _QUALITY_SUFFIX[quality]
    for ext in extensions:
        if ext == "pedal":  # a bass annotation, not part of the chord spelling
            continue
        label += ext
    if bass:
        label += "/" + bass
    return label


def _segments_from_states(
    states: np.ndarray,
    emis_per_state_conf: np.ndarray,
    starts_s: np.ndarray,
    end_s: float,
    *,
    is_beat: bool,
) -> list[ChordSegment]:
    """Collapse consecutive equal Viterbi states into ``ChordSegment``s. ``starts_s`` is the
    start time of each frame (beat or fixed window); ``is_beat`` decides whether the
    ``start_beat``/``end_beat`` frame-index fields are populated (``None`` in fixed mode)."""
    segments: list[ChordSegment] = []
    n = len(states)
    i = 0
    while i < n:
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        root, quality = _state_to_chord(int(states[i]))
        confidence = float(np.clip(np.mean(emis_per_state_conf[i:j]), 0.0, 1.0))
        seg_start = float(starts_s[i])
        seg_end = float(starts_s[j]) if j < n else end_s
        segments.append(
            ChordSegment(
                start_beat=float(i) if is_beat else None,
                end_beat=float(j) if is_beat else None,
                start_s=seg_start,
                end_s=seg_end,
                root=root,
                bass=None,
                quality=quality,
                extensions=[],
                label=_compose_label(root, quality, [], None),
                confidence=confidence,
                produced_by="baseline",
            )
        )
        i = j
    return segments


def chord_lane_baseline(pcm: Pcm, recon: Recon) -> ChordsLane:
    """The §13.1 baseline lane: templates + Viterbi over beat-synchronous HPSS chroma. When
    recon's beat-grid confidence is below ``CFG.BEAT_GRID_CONFIDENCE_MIN`` the whole lane
    switches to fixed 1 s windows and ``ChordsLane.grid`` becomes ``"fixed"`` (§13.4)."""
    grid = recon.grid
    if grid.confidence < CFG.BEAT_GRID_CONFIDENCE_MIN:
        chroma, starts = _fixed_window_chroma(pcm, recon.tuning_offset_cents)
        lane_grid = "fixed"
        is_beat = False
    else:
        chroma = beat_chroma(pcm, grid, recon.tuning_offset_cents)
        starts = np.asarray(grid.times_s, dtype=float)
        lane_grid = "beat"
        is_beat = True

    states = _viterbi(chroma)
    emis = _emission(chroma)
    conf = emis[np.arange(len(states)), states] if len(states) else np.empty(0)
    segments = _segments_from_states(states, conf, starts, pcm.duration_s, is_beat=is_beat)
    return ChordsLane(grid=lane_grid, segments=segments)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# §13.3 extension refinement.
# --------------------------------------------------------------------------- #


def _segment_frames(seg: ChordSegment, n_frames: int) -> list[int]:
    """Frame indices a segment spans — beat indices in beat mode, second-indices (== fixed
    window indices) in fixed mode."""
    if seg.start_beat is not None and seg.end_beat is not None:
        lo, hi = int(round(seg.start_beat)), int(round(seg.end_beat))
    else:
        lo, hi = int(round(seg.start_s)), int(round(seg.end_s))
    return list(range(max(0, lo), min(n_frames, hi)))


def refine_extensions(lane: ChordsLane, chroma: np.ndarray, grid: BeatGrid) -> ChordsLane:  # noqa: ARG001
    """§13.3: per segment >= ``CHORD_EXTENSION_MIN_BEATS`` beats, take the residual chroma
    after the base triad and promote pitch classes whose energy exceeds
    ``CHORD_EXTENSION_ENERGY_RATIO`` x the triad energy AND that are stable in
    >= ``CHORD_STABILITY_RATIO`` of the segment's beats to extensions. A 1-beat segment can
    never upgrade (any single observation is trivially 100% stable — noise would win)."""
    n_frames = chroma.shape[0]
    ratio = float(CFG.CHORD_EXTENSION_ENERGY_RATIO)
    out: list[ChordSegment] = []
    for seg in lane.segments:
        if seg.quality == "N":
            out.append(seg.model_copy())
            continue
        idx = _segment_frames(seg, n_frames)
        if len(idx) < CFG.CHORD_EXTENSION_MIN_BEATS:
            out.append(seg.model_copy())
            continue

        seg_chroma = chroma[idx]  # (beats, 12)
        root_idx = _ROOT_INDEX[seg.root]
        triad_intervals = set(_QUALITY_INTERVALS[seg.quality])
        triad_pcs = [(root_idx + iv) % 12 for iv in triad_intervals]
        per_beat_triad = seg_chroma[:, triad_pcs].sum(axis=1)  # (beats,)
        if per_beat_triad.sum() <= _EPS:
            out.append(seg.model_copy())
            continue

        exts: list[str] = []
        for interval, name in _EXTENSION_CANDIDATES:
            if interval in triad_intervals:
                continue
            pc = (root_idx + interval) % 12
            present = seg_chroma[:, pc] > ratio * np.clip(per_beat_triad, _EPS, None)
            if float(np.mean(present)) >= CFG.CHORD_STABILITY_RATIO:
                exts.append(name)
        if "6" in exts and ("7" in exts or "maj7" in exts):
            exts[exts.index("6")] = "13"

        out.append(
            seg.model_copy(
                update={
                    "extensions": [*exts, *(e for e in seg.extensions if e == "pedal")],
                    "label": _compose_label(seg.root, seg.quality, exts, seg.bass),
                }
            )
        )
    return ChordsLane(grid=lane.grid, segments=out)


# --------------------------------------------------------------------------- #
# §13.2 bass-informed slash / inversion detection.
# --------------------------------------------------------------------------- #


def detect_slash(lane: ChordsLane, bass_pc_per_beat: list[int | None]) -> ChordsLane:
    """§13.2: emit a slash chord only when a single non-root bass pitch class is stable
    across >= ``CHORD_STABILITY_RATIO`` of a segment's beats AND is a chord tone. A stable
    non-chord-tone bass is annotated ``pedal``; unstable bass movement is left alone — a
    walking bass under one C-major bar (C-D-E-G) must stay ``C``, not ``C, C/D, C/E, C/G``."""
    n_beats = len(bass_pc_per_beat)
    out: list[ChordSegment] = []
    for seg in lane.segments:
        if seg.quality == "N":
            out.append(seg.model_copy())
            continue
        idx = [b for b in _segment_frames(seg, n_beats)]
        if not idx:
            out.append(seg.model_copy())
            continue

        root_idx = _ROOT_INDEX[seg.root]
        counts = Counter(
            bass_pc_per_beat[b]
            for b in idx
            if bass_pc_per_beat[b] is not None and bass_pc_per_beat[b] != root_idx
        )
        bass: str | None = seg.bass
        extensions = list(seg.extensions)
        if counts:
            pc, cnt = counts.most_common(1)[0]
            assert pc is not None
            if cnt / len(idx) >= CFG.CHORD_STABILITY_RATIO:
                chord_pcs = {(root_idx + iv) % 12 for iv in _QUALITY_INTERVALS[seg.quality]}
                if pc in chord_pcs:
                    bass = _PITCH_CLASSES[pc]
                elif "pedal" not in extensions:
                    extensions.append("pedal")
        out.append(
            seg.model_copy(
                update={
                    "bass": bass,
                    "extensions": extensions,
                    "label": _compose_label(
                        seg.root, seg.quality, extensions, bass
                    ),
                }
            )
        )
    return ChordsLane(grid=lane.grid, segments=out)


def bass_pc_lane(pcm: Pcm, grid: BeatGrid) -> list[int | None]:
    """Per-beat bass pitch class (or ``None`` when unvoiced): low-pass the mix below 250 Hz
    and run pYIN, then take the median voiced f0 inside each beat interval (§13.2 — the
    no-separation path; a real bass stem replaces this when one exists)."""
    import librosa
    from scipy.signal import butter, sosfiltfilt

    mono = _to_mono(pcm.samples)
    sr = pcm.sample_rate
    nyq = sr / 2.0
    sos = butter(4, min(250.0, nyq * 0.99) / nyq, btype="low", output="sos")
    low = sosfiltfilt(sos, mono).astype(np.float32)

    f0, _voiced, _prob = librosa.pyin(y=low, sr=sr, fmin=41.0, fmax=250.0)
    f0_times = librosa.times_like(f0, sr=sr)

    result: list[int | None] = []
    times = list(grid.times_s)
    for i, start in enumerate(times):
        end = times[i + 1] if i + 1 < len(times) else pcm.duration_s
        mask = (f0_times >= start) & (f0_times < end)
        voiced = f0[mask]
        voiced = voiced[np.isfinite(voiced)]
        if voiced.size == 0:
            result.append(None)
        else:
            midi = float(librosa.hz_to_midi(float(np.median(voiced))))
            result.append(int(round(midi)) % 12)
    return result


# --------------------------------------------------------------------------- #
# §13.4 theory layer — local-key windows + hysteresis + music21 Roman numerals.
# --------------------------------------------------------------------------- #


def _bar_starts(lane: ChordsLane, grid: BeatGrid) -> list[float]:
    """Bar start times: recon downbeats on a confident beat grid, else fixed
    ``FIXED_GRID_BARS_TO_SECONDS``-second bars over the lane's span (§13.4 rubato)."""
    if lane.grid == "beat" and len(grid.downbeats_s) >= 2:
        return list(grid.downbeats_s)
    span = max((s.end_s for s in lane.segments), default=0.0)
    step = float(CFG.FIXED_GRID_BARS_TO_SECONDS)
    n = max(1, int(math.ceil(span / step)))
    return [i * step for i in range(n)]


def _window_key(lane: ChordsLane, lo_s: float, hi_s: float) -> tuple[str, str] | None:
    """Krumhansl on the segments overlapping ``[lo_s, hi_s)``, weighting each segment's
    chord tones by its duration."""
    profile = np.zeros(12)
    for seg in lane.segments:
        if seg.quality == "N" or seg.end_s <= lo_s or seg.start_s >= hi_s:
            continue
        weight = min(seg.end_s, hi_s) - max(seg.start_s, lo_s)
        if weight <= 0:
            continue
        root_idx = _ROOT_INDEX[seg.root]
        for interval in _QUALITY_INTERVALS[seg.quality]:
            profile[(root_idx + interval) % 12] += weight
    if profile.sum() <= _EPS:
        return None
    best: tuple[float, str, str] | None = None
    for mode_name, ks in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        for root in range(12):
            corr = _safe_corr(profile, np.roll(ks, root))
            if best is None or corr > best[0]:
                best = (corr, _PITCH_CLASSES[root], mode_name)
    assert best is not None
    return best[1], best[2]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < _EPS or np.std(b) < _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _apply_hysteresis(keys: list[tuple[str, str] | None]) -> list[tuple[str, str] | None]:
    """A modulation is accepted only after ``CFG.KEY_HYSTERESIS_WINDOWS`` consecutive
    windows vote the new key (borrowed-chord passages must not read as key changes)."""
    accepted: list[tuple[str, str] | None] = []
    current: tuple[str, str] | None = None
    pending: tuple[str, str] | None = None
    pending_count = 0
    for k in keys:
        if k is None:
            accepted.append(current)
            continue
        if current is None:
            current = k
        elif k == current:
            pending, pending_count = None, 0
        elif k == pending:
            pending_count += 1
            if pending_count >= CFG.KEY_HYSTERESIS_WINDOWS:
                current, pending, pending_count = k, None, 0
        else:
            pending, pending_count = k, 1
        accepted.append(current)
    return accepted


def theory_layer(lane: ChordsLane, grid: BeatGrid) -> ChordsLane:
    """§13.4: sliding local-key windows (``KEY_WINDOW_BARS`` wide, ``KEY_HOP_BARS`` hop) with
    hysteresis, then a music21 Roman numeral per segment against its *local* key. Sets each
    segment's ``key_context`` (``"C major"``) and ``roman`` (``"IV"``); leaves the headline
    key to recon (§13.4 precedence)."""
    bars = _bar_starts(lane, grid)
    n_bars = len(bars)
    span_end = max((s.end_s for s in lane.segments), default=0.0)
    win, hop = int(CFG.KEY_WINDOW_BARS), int(CFG.KEY_HOP_BARS)

    window_starts_s: list[float] = []
    window_keys: list[tuple[str, str] | None] = []
    start_bar = 0
    while start_bar < n_bars:
        lo = bars[start_bar]
        end_bar = start_bar + win
        hi = bars[end_bar] if end_bar < n_bars else span_end
        window_starts_s.append(lo)
        window_keys.append(_window_key(lane, lo, hi))
        if end_bar >= n_bars:
            break
        start_bar += hop
    if not window_keys:  # no bar grid at all — one global window
        window_starts_s = [0.0]
        window_keys = [_window_key(lane, 0.0, span_end)]

    accepted = _apply_hysteresis(window_keys)

    def key_at(t: float) -> tuple[str, str] | None:
        chosen: tuple[str, str] | None = accepted[0] if accepted else None
        for w_start, key in zip(window_starts_s, accepted, strict=True):
            if w_start <= t + _EPS:
                chosen = key
        return chosen

    out: list[ChordSegment] = []
    for seg in lane.segments:
        local = key_at(seg.start_s)
        key_context = f"{local[0]} {local[1]}" if local else None
        roman = _roman(seg, local) if (local and seg.quality != "N") else None
        out.append(seg.model_copy(update={"key_context": key_context, "roman": roman}))
    return ChordsLane(grid=lane.grid, segments=out)


def _roman(seg: ChordSegment, key: tuple[str, str]) -> str | None:
    """A music21 Roman numeral for one segment against ``key`` (built per segment; failures
    degrade to ``None`` rather than sinking the whole lane)."""
    try:
        from music21 import chord as m21chord
        from music21 import key as m21key
        from music21 import roman as m21roman

        root_idx = _ROOT_INDEX[seg.root]
        # Voice root-position (root in the bass) so the Roman numeral reflects the chord's
        # function, not an accidental inversion from pitch-class ordering: intervals are
        # semitones above the root, so 48 + root + interval stacks the tones over the root.
        midi = [48 + root_idx + iv for iv in _QUALITY_INTERVALS[seg.quality]]
        ch = m21chord.Chord(midi)
        k = m21key.Key(key[0], key[1])
        rn = m21roman.romanNumeralFromChord(ch, k)
        return str(rn.figure)
    except Exception:
        return None
