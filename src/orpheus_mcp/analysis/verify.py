"""``analysis/verify.py`` (Task 4.2, §5): the four verification gates that triangulate
"is this separated stem trustworthy" without a ground-truth stem to compare against.
Gates 1-3 are pure numeric functions on constructed signals (no model dependency beyond
whatever produced their inputs); gate 4 (cross-model agreement) is a Phase-5 stub here —
``run_gates`` always returns it as ``na``/``None``, consistent with §5's decision rule
treating N/A gates as excluded rather than missing evidence.

**Alignment precondition (§5, stated once, applies to every gate below):** callers must
pass LENGTH-ALIGNED arrays — ``analysis/separation.py``'s ``separate_region`` already
resamples/length-aligns every stem to the input before these gates ever see it. An
unaligned ``mix - sum(stems)`` residual is spuriously huge, not a real quality signal.

**The §5 decision rule** (``overall_verdict``): the ladder escalates iff any APPLICABLE
gate is not ``pass``; ``na`` gates are excluded from the decision; if every gate is
``na`` the artifact is ``unverifiable`` (climbing cannot fix unverifiability).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import GateResult, GateSet, Timeline

if TYPE_CHECKING:
    from orpheus_mcp.analysis.timeline import TaggerEngine

__all__ = [
    "gate_mix_consistency",
    "gate_bleed",
    "gate_silence_honesty",
    "overall_verdict",
    "run_gates",
]

# Guards the (denominator == 0) case in every ratio below without perturbing normal
# readings — matches ``analysis/timeline.py``'s ``_EPS`` (same §5.2 formula, reused for
# the §3.1 exposure check there and the bleed gate here; kept as a separate literal so
# this module has no import-time dependency on timeline.py's private constant).
_EPS = 1e-6

# Floors every dB computation below at this value instead of -inf when a residual/tail
# is EXACT digital silence (log10(0) is undefined) — comfortably below any pass
# threshold in CFG, so a perfect reconstruction still reads as an obvious, finite pass
# rather than raising or serializing as -inf/NaN.
_DB_FLOOR = -240.0


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _db_ratio(numerator_rms: float, denominator_rms: float) -> float:
    """``20*log10(numerator_rms / denominator_rms)``, floored at ``_DB_FLOOR`` instead
    of raising/going to -inf when either RMS is exact digital silence."""
    ratio = numerator_rms / denominator_rms if denominator_rms > 0.0 else float("inf")
    if ratio <= 0.0:
        return _DB_FLOOR
    return max(20.0 * math.log10(ratio), _DB_FLOOR)


# --------------------------------------------------------------------------- #
# Gate 1 — mix consistency (§5.1): multi-stem outputs only.
# --------------------------------------------------------------------------- #


def gate_mix_consistency(
    mix: np.ndarray, stems: dict[str, np.ndarray], multi_stem: bool
) -> GateResult:
    """``residual_db = 20*log10(rms(mix - sum(stems)) / rms(mix))``; pass iff
    ``residual_db <= CFG.GATE_MIX_RESIDUAL_PASS_DB``. N/A unless ``multi_stem`` — a
    single-target extraction's "residual" is the rest of the song by construction (§5.1),
    not a meaningful physics check."""
    if not multi_stem:
        return GateResult(verdict="na", value=None)

    summed = np.zeros_like(mix, dtype=np.float64)
    for stem in stems.values():
        summed = summed + np.asarray(stem, dtype=np.float64)
    residual = np.asarray(mix, dtype=np.float64) - summed

    value = _db_ratio(_rms(residual), _rms(np.asarray(mix, dtype=np.float64)))
    verdict: Literal["pass", "fail"] = (
        "pass" if value <= CFG.GATE_MIX_RESIDUAL_PASS_DB else "fail"
    )
    return GateResult(verdict=verdict, value=value)


# --------------------------------------------------------------------------- #
# Gate 2 — bleed (§5.2): the T2 tagger run ON THE ISOLATED STEM.
# --------------------------------------------------------------------------- #


def gate_bleed(
    stem_activations: np.ndarray, target_idx: list[int], mapped_idx: list[int]
) -> GateResult:
    """``bleed = mean over active frames of max(non-target activation) / (target
    activation + eps)``, clipped to ``[0, 1]`` per frame. "Target" = ``target_idx``
    columns; "non-target" = ``mapped_idx`` columns minus ``target_idx`` (columns outside
    ``mapped_idx`` — e.g. AudioSet classes with no musical meaning — never count as
    bleed sources, §9's "mapped-class machinery"). Active frames = target activation >
    ``CFG.TIMELINE_ACTIVITY_THRESHOLD``. Empty ``target_idx`` or zero active frames both
    resolve to N/A (an empty tagger-class mapping, or a target that never registers
    above the activity floor, leave nothing to average — never a division over an empty
    set). Pass strictly < ``CFG.GATE_BLEED_PASS``."""
    if not target_idx:
        return GateResult(verdict="na", value=None)

    arr = np.asarray(stem_activations, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return GateResult(verdict="na", value=None)

    target_set = set(target_idx)
    nontarget_idx = [i for i in mapped_idx if i not in target_set]

    target_vals = arr[:, target_idx].max(axis=1)
    active_mask = target_vals > CFG.TIMELINE_ACTIVITY_THRESHOLD
    if not bool(active_mask.any()):
        return GateResult(verdict="na", value=None)

    if nontarget_idx:
        nontarget_vals = arr[:, nontarget_idx].max(axis=1)
    else:
        nontarget_vals = np.zeros(arr.shape[0], dtype=np.float64)

    ratios = np.clip(nontarget_vals / (target_vals + _EPS), 0.0, 1.0)
    value = float(ratios[active_mask].mean())
    verdict: Literal["pass", "fail"] = "pass" if value < CFG.GATE_BLEED_PASS else "fail"
    return GateResult(verdict=verdict, value=value)


# --------------------------------------------------------------------------- #
# Gate 3 — silence honesty (§5.3): FX-tail-tolerant.
# --------------------------------------------------------------------------- #


def _active_mask(timeline: Timeline, target_idx: Sequence[int]) -> list[bool]:
    mask: list[bool] = []
    for frame in timeline.frames:
        vals = [frame[i] for i in target_idx if i < len(frame)]
        target_val = max(vals) if vals else 0.0
        mask.append(target_val > CFG.TIMELINE_ACTIVITY_THRESHOLD)
    return mask


def _tail_excluded(active: Sequence[bool], tail_frames: int) -> list[bool]:
    """Frames within ``tail_frames`` after each maximal active run are excluded from
    the inactive-window pool (§5.3's reverb/delay-tail carve-out) — not thresholded,
    removed from consideration entirely."""
    n = len(active)
    excluded = [False] * n
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        for k in range(j, min(n, j + tail_frames)):
            excluded[k] = True
        i = j
    return excluded


def _frame_samples(stem: np.ndarray, indices: Iterable[int], hop_s: float, sr: int) -> np.ndarray:
    n = stem.shape[0]
    chunks: list[np.ndarray] = []
    for i in indices:
        start = min(int(round(i * hop_s * sr)), n)
        end = min(int(round((i + 1) * hop_s * sr)), n)
        if end > start:
            chunks.append(stem[start:end])
    if not chunks:
        return np.zeros((0, *stem.shape[1:]), dtype=stem.dtype)
    return np.concatenate(chunks, axis=0)


def gate_silence_honesty(
    stem: np.ndarray, sr: int, timeline: Timeline, target_idx: list[int]
) -> GateResult:
    """``false_energy_db = 20*log10(rms(stem, inactive) / rms(stem, active))``, where
    "inactive" windows are the timeline's non-active frames MINUS a tail allowance
    (``CFG.GATE_SILENCE_TAIL_ALLOWANCE_S``) after each active region — a legitimately
    separated stem's reverb/delay tail is excluded from the check, not thresholded.
    ``timeline`` must already be region-cropped and re-indexed so frame 0 corresponds to
    ``stem``'s sample 0 (the caller's job — this function has no notion of a song-wide
    offset). N/A when there is nothing to score: an empty ``target_idx``, zero active
    frames (no active-window RMS to normalize against), or zero qualifying inactive
    frames (wall-to-wall — §5.3's stated example). Pass iff
    ``false_energy_db <= CFG.GATE_SILENCE_PASS_DB``."""
    if not target_idx:
        return GateResult(verdict="na", value=None)

    active = _active_mask(timeline, target_idx)
    if not any(active):
        return GateResult(verdict="na", value=None)

    hop = timeline.hop_s
    tail_frames = max(0, math.ceil(CFG.GATE_SILENCE_TAIL_ALLOWANCE_S / hop)) if hop > 0 else 0
    excluded = _tail_excluded(active, tail_frames)

    paired = zip(active, excluded, strict=True)
    inactive_idx = [i for i, (a, x) in enumerate(paired) if not a and not x]
    if not inactive_idx:
        return GateResult(verdict="na", value=None)

    active_idx = [i for i, a in enumerate(active) if a]
    stem = stem if stem.ndim == 2 else stem.reshape(-1, 1)

    active_rms = _rms(_frame_samples(stem, active_idx, hop, sr))
    inactive_rms = _rms(_frame_samples(stem, inactive_idx, hop, sr))

    value = _db_ratio(inactive_rms, active_rms)
    verdict: Literal["pass", "fail"] = "pass" if value <= CFG.GATE_SILENCE_PASS_DB else "fail"
    return GateResult(verdict=verdict, value=value)


# --------------------------------------------------------------------------- #
# §5 decision rule + run_gates composition
# --------------------------------------------------------------------------- #


def overall_verdict(gates: Iterable[GateResult]) -> Literal["pass", "fail", "unverifiable"]:
    """fail if ANY applicable gate fails; unverifiable iff EVERY gate is N/A; N/A gates
    are excluded from the decision otherwise (§5)."""
    gate_list = list(gates)
    if any(g.verdict == "fail" for g in gate_list):
        return "fail"
    if gate_list and all(g.verdict == "na" for g in gate_list):
        return "unverifiable"
    return "pass"


def run_gates(
    *,
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    target: str,
    sr: int,
    multi_stem: bool,
    tagger: TaggerEngine,
    target_tagger_classes: list[str],
    mix_timeline: Timeline,
) -> GateSet:
    """Composes gates 1-3 for ``stems[target]`` plus the Phase-5 gate-4 stub (always
    ``na``/``None``) into one :class:`GateSet`, including the §5 ``overall`` verdict.

    - Gate 1 sees the full ``stems`` dict (mix consistency needs every co-produced stem).
    - Gate 2 runs ``tagger`` ON THE STEM itself (§5.2) via ``analysis.timeline.
      build_timeline`` — reusing the same max-pool-to-1s-hop/mapped-class machinery the
      T2 timeline artifact uses, so the bleed judge and the T2 timeline can never
      silently disagree about class columns or frame alignment.
    - Gate 3 reads ``mix_timeline`` — the SONG's (not the stem's own) tagger timeline,
      already region-cropped/re-indexed by the caller so frame 0 == the stem's sample 0
      (§5.3: judging silence against the mix's ground truth of when the target plays,
      not against a possibly bleed-contaminated re-tagging of the stem itself).
    """
    from orpheus_mcp.analysis.cachefs import Pcm
    from orpheus_mcp.analysis.timeline import build_timeline

    g1 = gate_mix_consistency(mix, stems, multi_stem)

    stem = stems[target]
    stem = stem if stem.ndim == 2 else stem.reshape(-1, 1)
    stem_pcm = Pcm(
        samples=np.asarray(stem, dtype=np.float32),
        sample_rate=sr,
        path="<stem>",
        pcm_hash="stem",
        duration_s=(stem.shape[0] / sr) if sr else 0.0,
        decoder="stem",
    )
    stem_timeline = build_timeline(stem_pcm, tagger)
    stem_classes = stem_timeline.classes
    stem_target_idx = [stem_classes.index(c) for c in target_tagger_classes if c in stem_classes]
    stem_mapped_idx = list(range(len(stem_classes)))
    stem_activations = (
        np.asarray(stem_timeline.frames, dtype=np.float64)
        if stem_timeline.frames
        else np.zeros((0, len(stem_classes)), dtype=np.float64)
    )
    g2 = gate_bleed(stem_activations, stem_target_idx, stem_mapped_idx)

    mix_target_idx = [
        mix_timeline.classes.index(c) for c in target_tagger_classes if c in mix_timeline.classes
    ]
    g3 = gate_silence_honesty(stem, sr, mix_timeline, mix_target_idx)

    g4 = GateResult(verdict="na", value=None)  # Phase 5: cross-model agreement

    return GateSet(
        mix_consistency=g1,
        bleed=g2,
        silence_honesty=g3,
        agreement=g4,
        overall=overall_verdict([g1, g2, g3, g4]),
    )
