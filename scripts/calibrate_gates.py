"""``scripts/calibrate_gates.py`` — §15.1 gate-threshold calibration pipeline (Task 5.3).

Runs every working registry separator over the desktop synthetic-truth fixture set (Task 0.4,
``~/.orpheus_fixtures``: 8 genres × 3 densities = 24 mixes whose true stems are retained),
computes **true SDR** (``10*log10(rms(truth)^2 / rms(truth-est)^2)``) and every §5 gate metric
per (fixture, engine, target), then picks each gate constant's threshold at the ROC knee that
separates good runs (``SDR > CFG.CALIBRATION_SDR_GOOD``) from bad runs
(``SDR < CFG.CALIBRATION_SDR_BAD``). The fitted values are written to
``data/calibrated_thresholds.json`` with per-constant provenance so ``config.py`` (D7) loads
them at import and ``cachefs.verify_config_hash`` busts stale gate-artifact caches automatically.

**Domain-shift discipline (Task 2.2 standing escalation).** These fixtures are synthetic sfizz
renders — domain-shifted for PANNs, which barely crosses ``TIMELINE_ACTIVITY_THRESHOLD`` on
them (pin run (d): bass/chords 0.000 F1, drums 0.071). So:

  (a) every fitted value is emitted alongside its ``fit_domain`` and ``fixture_set_hash`` caveat;
  (b) a ``--domain`` tag stamps each constant's provenance, and re-runs MERGE into any existing
      file per-constant, so a later owned-audio pass (``--domain owned-audio``) coexists with the
      synthetic gate fits instead of clobbering them;
  (c) ``TIMELINE_ACTIVITY_THRESHOLD`` is guarded: if the fitted value would make the §3.1c
      absent-target pre-gate read >20% of truth-active fixture frames as inactive, it is NOT
      shipped — it falls back to the §16 initial with an explicit "domain-shift fallback" note.
      ``EXPOSURE_THRESHOLD`` is likewise owned-audio-deferred (no per-frame clean/unclean label
      exists on fixtures), shipping the §16 initial with a caveat. Design goal: never let
      synthetic-domain fitting silently break real-audio behavior.

The 6-song owned-audio ear panel (§15.1) is HUMAN-deferred — this script never fabricates it.

Usage::

    uv run --extra analysis --extra separation python scripts/calibrate_gates.py   # desktop
    uv run --extra analysis python scripts/calibrate_gates.py --ci-smoke           # CI stub run
    uv run ... python scripts/calibrate_gates.py --fixtures 12            # documented >=12 subset
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scripts.pick_sed_head import (
    ROLE_CLASS_MAP,
    activity_frames_from_mono,
    verify_manifest_integrity,
)

from orpheus_mcp.analysis.config import CFG, cal_field_names, calibrated_thresholds_path

if TYPE_CHECKING:
    from orpheus_mcp.analysis.cachefs import Pcm
    from orpheus_mcp.analysis.model_registry import ModelEntry
    from orpheus_mcp.analysis.separation import SeparatorEngine
    from orpheus_mcp.analysis.timeline import TaggerEngine

_LOCKED_HOP_S = 1.0  # matches analysis/timeline.py's locked Timeline hop
_SDR_CEILING_DB = 120.0  # finite stand-in for a perfect (zero-error) reconstruction

# The two working desktop engines (brief): SCNet-XL (msst) + htdemucs-ft (audio-separator).
WORKING_ENGINE_IDS: tuple[str, ...] = ("4stem-b/scnet-xl", "4stem-c/htdemucs-ft")

# Separator target -> the truth role(s) whose stems sum to that target's ground truth. "other"
# is everything that is not drums/bass/vocals (the fixture set's chords + lead melody).
TARGET_TRUTH_ROLES: dict[str, list[str]] = {
    "drums": ["drums"],
    "bass": ["bass"],
    "vocals": ["vocals"],
    "other": ["chords", "lead"],
    "guitar": ["guitar"],
    "piano": ["chords"],
}

# Minimum ROC separation quality (Youden's J) below which a gate fit is not trusted enough to
# ship — the fit is reported but the constant falls back to its §16 initial (never silently
# break real-audio behavior on a weak/overlapping synthetic-domain separation).
MIN_ROC_QUALITY = 0.5
MIN_CLASS_SAMPLES = 3  # need at least this many good AND bad samples to fit a gate at all
ABSENT_FIRE_MAX = 0.20  # §3.1c guard: fitted activity threshold may suppress <= this share

# Constants this script fits, and how. EXPOSURE_THRESHOLD is deliberately absent (owned-audio
# deferred); the remaining # cal fields (VAD_VOCAL_ACTIVE_DBFS, GATE_SILENCE_TAIL_ALLOWANCE_S,
# CHORD_EXTENSION_ENERGY_RATIO) are out of §15.1's SDR-ROC scope entirely.


@dataclass(frozen=True)
class GateMetricSpec:
    """One gate constant fit by SDR-labelled ROC. ``direction`` is ``"le"`` when a LOWER metric
    is better (pass iff ``metric <= threshold``: residual dB, bleed ratio, false-energy dB) or
    ``"ge"`` when higher is better (pass iff ``metric >= threshold``: agreement r)."""

    constant: str
    metric: str
    direction: str


GATE_SPECS: tuple[GateMetricSpec, ...] = (
    GateMetricSpec("GATE_MIX_RESIDUAL_PASS_DB", "mix_residual_db", "le"),
    GateMetricSpec("GATE_BLEED_PASS", "bleed", "le"),
    GateMetricSpec("GATE_SILENCE_PASS_DB", "silence_db", "le"),
    GateMetricSpec("GATE_AGREEMENT_PASS", "agreement_r", "ge"),
)

_PIN_REPORT = Path("docs/pins/2026-07-model-pins.md")
_PIN_SECTION_HEADER = "## Pin run — §15.1 calibration"


# --------------------------------------------------------------------------- #
# Pure math (CI-safe): SDR, ROC knee, absent-fire guard — exercised directly by tests.
# --------------------------------------------------------------------------- #


def _mono(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim > 1 and arr.shape[1] > 1:
        return arr.mean(axis=1)
    return arr.reshape(-1)


def sdr_db(truth: np.ndarray, est: np.ndarray) -> float:
    """True SDR in dB: ``10*log10(rms(truth)^2 / rms(truth-est)^2)`` = ``20*log10(rms(truth) /
    rms(truth-est))``. Both are downmixed to mono and aligned to the shorter length first.
    Returns ``-inf`` when ``truth`` is exact silence (no ground-truth signal to recover — the
    caller skips these) and ``_SDR_CEILING_DB`` when the error is exact zero (perfect, floored
    finite instead of ``+inf``)."""
    t = _mono(truth)
    e = _mono(est)
    n = min(t.shape[0], e.shape[0])
    if n == 0:
        return -math.inf
    t = t[:n]
    e = e[:n]
    rms_t = float(np.sqrt(np.mean(np.square(t))))
    rms_e = float(np.sqrt(np.mean(np.square(t - e))))
    if rms_t <= 0.0:
        return -math.inf
    if rms_e <= 0.0:
        return _SDR_CEILING_DB
    return 20.0 * math.log10(rms_t / rms_e)


def label_by_sdr(sdr: float, *, good: float, bad: float) -> bool | None:
    """``True`` (known-good) if ``sdr > good``, ``False`` (known-bad) if ``sdr < bad``, else
    ``None`` — the ambiguous band ``[bad, good]`` is excluded from the ROC fit entirely (§15.1)."""
    if sdr > good:
        return True
    if sdr < bad:
        return False
    return None


def roc_knee(
    values: Sequence[float], labels: Sequence[bool], *, direction: str
) -> tuple[float | None, float]:
    """The ROC-knee threshold separating known-good (``label True``) from known-bad
    (``label False``) samples, plus its separation quality = Youden's J (``TPR - FPR``) in
    ``[0, 1]`` (1.0 = clean separation, lower = more overlap).

    ``direction="le"`` predicts good when ``value <= threshold`` (lower metric is better);
    ``"ge"`` predicts good when ``value >= threshold``. Candidate thresholds are the midpoints
    between sorted unique values plus a below-min / above-max sentinel, so the picked threshold
    always sits strictly between observed values. Returns ``(None, 0.0)`` for the DEGENERATE
    case — no good samples or no bad samples — where no threshold can separate what isn't there
    (the caller falls back to the §16 initial)."""
    vals = [float(v) for v in values]
    labs = [bool(x) for x in labels]
    if len(vals) != len(labs):
        raise ValueError(f"values/labels length mismatch: {len(vals)} vs {len(labs)}")
    goods = [v for v, x in zip(vals, labs, strict=True) if x]
    bads = [v for v, x in zip(vals, labs, strict=True) if not x]
    if not goods or not bads:
        return None, 0.0

    uniq = sorted(set(vals))
    candidates = [uniq[0] - 1.0]
    candidates += [(uniq[i] + uniq[i + 1]) / 2.0 for i in range(len(uniq) - 1)]
    candidates.append(uniq[-1] + 1.0)

    best_t: float | None = None
    best_j = -1.0
    for t in candidates:
        pred = [(v <= t) if direction == "le" else (v >= t) for v in vals]
        tp = sum(1 for p, x in zip(pred, labs, strict=True) if p and x)
        fn = sum(1 for p, x in zip(pred, labs, strict=True) if not p and x)
        fp = sum(1 for p, x in zip(pred, labs, strict=True) if p and not x)
        tn = sum(1 for p, x in zip(pred, labs, strict=True) if not p and not x)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return best_t, max(0.0, best_j)


def absent_fire_fraction(
    activations: Sequence[float], truth_active: Sequence[bool], threshold: float
) -> float:
    """§3.1c guard metric: the fraction of TRUTH-ACTIVE frames whose target activation is
    ``<= threshold`` — i.e. the share of frames a real target is playing that the fitted
    threshold would read as inactive (and thus could make ``target_absent`` falsely fire).
    ``0.0`` when there are no truth-active frames (nothing the threshold could wrongly suppress)."""
    active = [a for a, t in zip(activations, truth_active, strict=True) if t]
    if not active:
        return 0.0
    below = sum(1 for a in active if a <= threshold)
    return below / len(active)


# --------------------------------------------------------------------------- #
# Fit orchestration (pure, CI-safe): metric samples -> per-constant fit decisions.
# --------------------------------------------------------------------------- #


@dataclass
class MetricSample:
    """One (metric value, its run's SDR) pair feeding a gate's ROC fit."""

    value: float
    sdr: float


@dataclass
class FrameSample:
    """One (target activation, truth-active) frame feeding the activity-threshold fit."""

    activation: float
    truth_active: bool


@dataclass
class ConstantFit:
    """The fit decision for one # cal constant: what was fitted, what ships, and why."""

    constant: str
    initial: float
    fitted: float | None
    shipped: float
    fallback: bool
    quality: float
    n_good: int
    n_bad: int
    note: str = ""


def fit_gate(spec: GateMetricSpec, samples: Sequence[MetricSample], *, good: float, bad: float,
             ) -> ConstantFit:
    """Fit one gate constant from its SDR-labelled metric samples. Falls back to the §16 initial
    (never shipping an untrusted synthetic-domain threshold) when the two classes are too small,
    the ROC is degenerate, or its separation quality is below :data:`MIN_ROC_QUALITY`."""
    initial = float(getattr(CFG, spec.constant))
    labelled = [(s.value, label_by_sdr(s.sdr, good=good, bad=bad)) for s in samples]
    kept = [(v, lab) for v, lab in labelled if lab is not None]
    n_good = sum(1 for _v, lab in kept if lab)
    n_bad = sum(1 for _v, lab in kept if not lab)

    if n_good < MIN_CLASS_SAMPLES or n_bad < MIN_CLASS_SAMPLES:
        return ConstantFit(
            constant=spec.constant, initial=initial, fitted=None, shipped=initial,
            fallback=True, quality=0.0, n_good=n_good, n_bad=n_bad,
            note=(f"domain-shift fallback: only {n_good} good / {n_bad} bad labelled samples "
                  f"(< {MIN_CLASS_SAMPLES} each) — kept §16 initial."),
        )

    thr, quality = roc_knee([v for v, _ in kept], [lab for _, lab in kept],
                            direction=spec.direction)
    if thr is None or quality < MIN_ROC_QUALITY:
        return ConstantFit(
            constant=spec.constant, initial=initial, fitted=thr, shipped=initial,
            fallback=True, quality=quality, n_good=n_good, n_bad=n_bad,
            note=(f"domain-shift fallback: ROC separation J={quality:.2f} "
                  f"< {MIN_ROC_QUALITY} — kept §16 initial over an untrusted fit."),
        )
    return ConstantFit(
        constant=spec.constant, initial=initial, fitted=float(thr), shipped=float(thr),
        fallback=False, quality=quality, n_good=n_good, n_bad=n_bad,
        note=f"fitted at ROC knee (J={quality:.2f}).",
    )


def fit_activity_threshold(frames: Sequence[FrameSample]) -> ConstantFit:
    """Fit ``TIMELINE_ACTIVITY_THRESHOLD`` from per-frame (activation, truth-active) pairs, then
    apply the §3.1c absent-fire guard: if the fitted value would suppress > :data:`ABSENT_FIRE_MAX`
    of truth-active frames it is NOT shipped — fall back to the §16 initial with a domain-shift
    note (the Task 2.2 standing escalation, made mechanical)."""
    initial = float(CFG.TIMELINE_ACTIVITY_THRESHOLD)
    acts = [f.activation for f in frames]
    truth = [f.truth_active for f in frames]
    n_good = sum(1 for t in truth if t)
    n_bad = sum(1 for t in truth if not t)

    if n_good < MIN_CLASS_SAMPLES or n_bad < MIN_CLASS_SAMPLES:
        return ConstantFit(
            constant="TIMELINE_ACTIVITY_THRESHOLD", initial=initial, fitted=None, shipped=initial,
            fallback=True, quality=0.0, n_good=n_good, n_bad=n_bad,
            note=(f"domain-shift fallback: only {n_good} truth-active / {n_bad} inactive frames "
                  f"— kept §16 initial."),
        )

    thr, quality = roc_knee(acts, truth, direction="ge")
    if thr is None or quality < MIN_ROC_QUALITY:
        return ConstantFit(
            constant="TIMELINE_ACTIVITY_THRESHOLD", initial=initial, fitted=thr, shipped=initial,
            fallback=True, quality=quality, n_good=n_good, n_bad=n_bad,
            note=(f"domain-shift fallback: ROC separation J={quality:.2f} < {MIN_ROC_QUALITY} "
                  "(the tagger barely crosses any activity threshold on these synthetic renders) "
                  "— kept §16 initial."),
        )

    fire = absent_fire_fraction(acts, truth, thr)
    if fire > ABSENT_FIRE_MAX:
        return ConstantFit(
            constant="TIMELINE_ACTIVITY_THRESHOLD", initial=initial, fitted=float(thr),
            shipped=initial, fallback=True, quality=quality, n_good=n_good, n_bad=n_bad,
            note=(f"domain-shift fallback: fitted {thr:.3f} would read {fire:.0%} of truth-active "
                  f"frames as inactive (> {ABSENT_FIRE_MAX:.0%}), risking a false §3.1c "
                  "absent-target fire — kept §16 initial (owned-audio calibration deferred)."),
        )
    return ConstantFit(
        constant="TIMELINE_ACTIVITY_THRESHOLD", initial=initial, fitted=float(thr),
        shipped=float(thr), fallback=False, quality=quality, n_good=n_good, n_bad=n_bad,
        note=(f"fitted at ROC knee (J={quality:.2f}); absent-fire {fire:.0%} within "
              f"{ABSENT_FIRE_MAX:.0%} guard."),
    )


def exposure_fallback() -> ConstantFit:
    """``EXPOSURE_THRESHOLD`` has no per-frame clean/unclean ground truth on synthetic fixtures,
    so it is owned-audio-deferred: ship the §16 initial with an explicit caveat rather than fit a
    threshold the domain can't defend."""
    initial = float(CFG.EXPOSURE_THRESHOLD)
    return ConstantFit(
        constant="EXPOSURE_THRESHOLD", initial=initial, fitted=None, shipped=initial,
        fallback=True, quality=0.0, n_good=0, n_bad=0,
        note=("owned-audio deferred: no per-frame clean/unclean label exists on synthetic "
              "fixtures — kept §16 initial pending the ear-panel calibration pass."),
    )


# --------------------------------------------------------------------------- #
# Metric collection over the fixture set (desktop or CI-stub) — the impure layer.
# --------------------------------------------------------------------------- #


@dataclass
class CalibrationSamples:
    """Everything the fit layer consumes, pooled across the whole run."""

    gate_samples: dict[str, list[MetricSample]] = field(default_factory=dict)
    frame_samples: list[FrameSample] = field(default_factory=list)
    n_fixtures: int = 0
    n_separations: int = 0
    engines_used: list[str] = field(default_factory=list)

    def add_metric(self, metric: str, value: float | None, sdr: float) -> None:
        if value is None or not math.isfinite(value) or not math.isfinite(sdr):
            return
        self.gate_samples.setdefault(metric, []).append(MetricSample(value=value, sdr=sdr))


def _truth_stem(fixture_dir: Path, roles: Sequence[str], truth_roles: dict[str, str],
                ) -> np.ndarray | None:
    """Sum the truth stems for ``roles`` that this fixture actually has (``truth_roles`` maps
    role -> relative stem path), returning a mono float64 array — or ``None`` if none exist."""
    import soundfile as sf

    parts: list[np.ndarray] = []
    for role in roles:
        rel = truth_roles.get(role)
        if rel is None:
            continue
        data, _sr = sf.read(str(fixture_dir / rel), always_2d=True, dtype="float32")
        parts.append(_mono(data))
    if not parts:
        return None
    n = max(p.shape[0] for p in parts)
    acc = np.zeros(n, dtype=np.float64)
    for p in parts:
        acc[: p.shape[0]] += p
    return acc


def collect_samples(
    fixture_dirs: Sequence[Path],
    engine_pairs: Sequence[tuple[ModelEntry, SeparatorEngine | None]],
    tagger: TaggerEngine,
) -> CalibrationSamples:
    """Run every engine over every fixture and pool the SDR + gate + activity-frame samples.

    Impure by nature (decode + separation + tagger) but engine-agnostic: the desktop run passes
    real registry entries with ``engine=None`` (the real msst / audio-separator adapters); the
    CI smoke passes stub entries + :class:`~stub_engines.StubEngine` doubles, so the identical
    control flow runs with ZERO checkpoint downloads."""
    import soundfile as sf

    from orpheus_mcp.analysis.cachefs import decode
    from orpheus_mcp.analysis.separation import separate_region as sep_core
    from orpheus_mcp.analysis.timeline import build_timeline, crop_timeline
    from orpheus_mcp.analysis.verify import (
        gate_agreement,
        gate_bleed,
        gate_mix_consistency,
        gate_silence_honesty,
    )
    from orpheus_mcp.models import Window

    out = CalibrationSamples(engines_used=[e.id for e, _ in engine_pairs])

    for fixture_dir in fixture_dirs:
        truth_path = fixture_dir / "truth.json"
        if not truth_path.is_file():
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        truth_roles: dict[str, str] = truth.get("roles", {})

        mix_pcm = decode(str(fixture_dir / "mix.flac"))
        sr = mix_pcm.sample_rate
        n_frames = math.ceil(mix_pcm.duration_s / _LOCKED_HOP_S) if mix_pcm.duration_s > 0 else 0
        if n_frames == 0:
            continue
        out.n_fixtures += 1

        mix_timeline = build_timeline(mix_pcm, tagger)

        # --- activity-threshold frame samples (mix timeline vs per-role truth activity) --- #
        for role, rel in truth_roles.items():
            if role not in ROLE_CLASS_MAP:
                continue
            classes = [c for c in ROLE_CLASS_MAP[role] if c in mix_timeline.classes]
            if not classes:
                continue
            idx = [mix_timeline.classes.index(c) for c in classes]
            stem_pcm = decode(str(fixture_dir / rel))
            truth_active = activity_frames_from_mono(
                _mono(stem_pcm.samples), stem_pcm.sample_rate, n_frames
            )
            for f_i in range(min(n_frames, len(mix_timeline.frames))):
                act = max(mix_timeline.frames[f_i][j] for j in idx)
                out.frame_samples.append(
                    FrameSample(activation=act, truth_active=truth_active[f_i])
                )

        # --- per-engine separation + gate metrics --- #
        per_engine_target_est: dict[str, dict[str, np.ndarray]] = {}
        per_engine_target_sdr: dict[str, dict[str, float]] = {}
        for entry, engine in engine_pairs:
            try:
                results = sep_core(mix_pcm, entry, None, engine=engine)
            except Exception as exc:  # noqa: BLE001 - a failing engine skips, never aborts the run
                print(f"  ! {entry.id} failed on {fixture_dir.name}: {exc}")
                continue
            out.n_separations += 1

            stems_arrays: dict[str, np.ndarray] = {}
            for tgt, res in results.items():
                data, _sr = sf.read(res.stem_path, always_2d=True, dtype="float32")
                stems_arrays[tgt] = np.asarray(data, dtype=np.float32)

            g1 = gate_mix_consistency(_align_mix(mix_pcm, stems_arrays), stems_arrays, True)

            est_by_target: dict[str, np.ndarray] = {}
            sdr_by_target: dict[str, float] = {}
            for tgt, est in stems_arrays.items():
                truth_stem = _truth_stem(fixture_dir, TARGET_TRUTH_ROLES.get(tgt, []), truth_roles)
                if truth_stem is None or float(np.max(np.abs(truth_stem))) <= 0.0:
                    continue  # no ground truth for this target on this fixture — skip
                sdr = sdr_db(truth_stem, est)
                if not math.isfinite(sdr):
                    continue
                est_by_target[tgt] = est
                sdr_by_target[tgt] = sdr

                out.add_metric("mix_residual_db", g1.value, sdr)

                stem_pcm = _pcm_from_array(est, sr)
                stem_tl = build_timeline(stem_pcm, tagger)
                stem_classes = list(TARGET_TRUTH_ROLES.get(tgt, []))
                tgt_idx = _target_class_idx(stem_tl.classes, tgt)
                bleed = gate_bleed(
                    np.asarray(stem_tl.frames, dtype=np.float64)
                    if stem_tl.frames else np.zeros((0, len(stem_tl.classes))),
                    tgt_idx, list(range(len(stem_tl.classes))),
                ) if stem_classes else None
                out.add_metric("bleed", bleed.value if bleed else None, sdr)

                region_tl = crop_timeline(
                    mix_timeline, Window(start_s=0.0, end_s=mix_pcm.duration_s)
                )
                mix_tgt_idx = _target_class_idx(region_tl.classes, tgt)
                sil = gate_silence_honesty(est, sr, region_tl, mix_tgt_idx)
                out.add_metric("silence_db", sil.value, sdr)

            per_engine_target_est[entry.id] = est_by_target
            per_engine_target_sdr[entry.id] = sdr_by_target

        # --- cross-engine agreement (needs both engines' take on the same target) --- #
        ids = [e.id for e, _ in engine_pairs]
        if len(ids) >= 2:
            a_est, b_est = per_engine_target_est.get(ids[0], {}), per_engine_target_est.get(
                ids[1], {})
            a_sdr, b_sdr = per_engine_target_sdr.get(ids[0], {}), per_engine_target_sdr.get(
                ids[1], {})
            for tgt in set(a_est) & set(b_est):
                r = gate_agreement(a_est[tgt], b_est[tgt], sr)
                out.add_metric("agreement_r", r.value, min(a_sdr[tgt], b_sdr[tgt]))

    return out


def _align_mix(mix_pcm: Pcm, stems: dict[str, np.ndarray]) -> np.ndarray:
    """The mix, downmix-preserved to (frames, channels), length-aligned to the stems so
    ``gate_mix_consistency``'s ``mix - Σstems`` residual is a real reconstruction check."""
    mix = mix_pcm.samples if mix_pcm.samples.ndim == 2 else mix_pcm.samples.reshape(-1, 1)
    n = min(mix.shape[0], *(s.shape[0] for s in stems.values())) if stems else mix.shape[0]
    return np.asarray(mix[:n], dtype=np.float32)


def _pcm_from_array(arr: np.ndarray, sr: int) -> Pcm:
    from orpheus_mcp.analysis.cachefs import Pcm as _Pcm

    a2 = arr if arr.ndim == 2 else arr.reshape(-1, 1)
    return _Pcm(
        samples=np.asarray(a2, dtype=np.float32), sample_rate=sr, path="<stem>",
        pcm_hash="stem", duration_s=(a2.shape[0] / sr) if sr else 0.0, decoder="stem",
    )


def _target_class_idx(classes: Sequence[str], target: str) -> list[int]:
    """Column indices in ``classes`` that count as ``target``'s own activity — via the shared
    ``pick_sed_head.ROLE_CLASS_MAP`` where the target maps to a role, else empty (N/A)."""
    role_classes = ROLE_CLASS_MAP.get(target, [])
    return [i for i, c in enumerate(classes) if c in role_classes]


# --------------------------------------------------------------------------- #
# Fit-all + JSON emit (merge-preserving per-constant provenance across domains).
# --------------------------------------------------------------------------- #


def fit_all(samples: CalibrationSamples) -> list[ConstantFit]:
    """Fit every §15.1 constant from the pooled samples, in a stable, reported order."""
    good, bad = CFG.CALIBRATION_SDR_GOOD, CFG.CALIBRATION_SDR_BAD
    fits = [
        fit_gate(spec, samples.gate_samples.get(spec.metric, []), good=good, bad=bad)
        for spec in GATE_SPECS
    ]
    fits.append(fit_activity_threshold(samples.frame_samples))
    fits.append(exposure_fallback())
    return fits


def build_json(fits: Sequence[ConstantFit], fixture_set_hash: str, domain: str,
               existing: dict | None = None) -> dict:
    """The ``calibrated_thresholds.json`` payload. Each constant carries its own ``fit_domain`` +
    ``fixture_set_hash`` provenance; an ``existing`` file's constants are preserved and only the
    ones fit in THIS run are overwritten (so an owned-audio pass coexists with synthetic fits)."""
    cal = set(cal_field_names())
    constants: dict[str, dict] = {}
    if existing and isinstance(existing.get("constants"), dict):
        constants = {k: v for k, v in existing["constants"].items() if k in cal}
    for fit in fits:
        constants[fit.constant] = {
            "value": fit.shipped,
            "fit_domain": domain,
            "fixture_set_hash": fixture_set_hash,
            "fitted": fit.fitted,
            "initial": fit.initial,
            "fallback": fit.fallback,
            "roc_quality": round(fit.quality, 4),
            "n_good": fit.n_good,
            "n_bad": fit.n_bad,
            "note": fit.note,
        }
    return {
        "schema_version": 1,
        "generated_by": "scripts/calibrate_gates.py",
        "fit_domain": domain,
        "fixture_set_hash": fixture_set_hash,
        "calibration_sdr_good": CFG.CALIBRATION_SDR_GOOD,
        "calibration_sdr_bad": CFG.CALIBRATION_SDR_BAD,
        "n_fixtures": None,
        "constants": dict(sorted(constants.items())),
    }


def write_json(payload: dict, path: Path | None = None) -> Path:
    # Repo-root data/ (scripts/ -> repo root), unambiguous regardless of CWD.
    target = path if path is not None else (
        Path(__file__).resolve().parent.parent / "data" / "calibrated_thresholds.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def format_fit_table(fits: Sequence[ConstantFit]) -> str:
    header = ["constant", "fitted", "shipped", "fallback?", "ROC J", "good/bad", "note"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for fit in fits:
        fitted = f"{fit.fitted:.3f}" if fit.fitted is not None else "—"
        lines.append("| " + " | ".join([
            f"`{fit.constant}`", fitted, f"{fit.shipped:.3f}",
            "yes" if fit.fallback else "no", f"{fit.quality:.2f}",
            f"{fit.n_good}/{fit.n_bad}", fit.note,
        ]) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CI-smoke fakes: real 4-stem shape, vram_gb=0.0 so the VRAM guard is a no-op, no downloads.
# --------------------------------------------------------------------------- #


def _ci_smoke_engine_pairs() -> list[tuple[ModelEntry, SeparatorEngine]]:
    from orpheus_mcp.analysis.model_registry import ModelEntry
    from stub_engines import StubEngine

    targets = ["vocals", "drums", "bass", "other"]
    by_target = {
        "vocals": ["Singing"], "drums": ["Drum kit"], "bass": ["Bass guitar"], "other": [],
    }
    pairs: list[tuple[ModelEntry, SeparatorEngine]] = []
    for eid in ("ci-stub/engine-a", "ci-stub/engine-b"):
        entry = ModelEntry(
            id=eid, task="multi-stem", targets=list(targets), engine="msst",
            checkpoint_url=None, sha256=None, vram_gb=0.0, rt_factor=None, benchmark=None,
            license="CC0", rank=1, tagger_classes=["Singing", "Drum kit", "Bass guitar"],
            tagger_classes_by_target=by_target,
        )
        pairs.append((entry, StubEngine(list(targets))))
    return pairs


# --------------------------------------------------------------------------- #
# Pin-report append (idempotent, mirrors pick_sed_head's write-replace pattern).
# --------------------------------------------------------------------------- #


def _strip_section(text: str, header: str) -> str:
    out: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.strip() == header:
            skipping = True
            continue
        if skipping and line.startswith("## ") and line.strip() != header:
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def append_pin_section(
    fits: Sequence[ConstantFit], fixture_set_hash: str, domain: str, n_fixtures: int,
    engines: Sequence[str], *, pin_report_path: Path = _PIN_REPORT,
) -> None:
    base = (
        pin_report_path.read_text(encoding="utf-8")
        if pin_report_path.is_file()
        else "# Model Pins — Reference Analysis Engine\n"
    )
    base = _strip_section(base, _PIN_SECTION_HEADER)
    fitted_n = sum(1 for f in fits if not f.fallback)
    lines = [
        _PIN_SECTION_HEADER,
        "",
        f"Generated by `scripts/calibrate_gates.py` on the reference RTX 3080 over the desktop "
        f"fixture set (`~/.orpheus_fixtures`, {n_fixtures} fixtures scored, `fit_domain` "
        f"`{domain}`, `fixture_set_hash` `{fixture_set_hash}`). Engines: "
        f"{', '.join(f'`{e}`' for e in engines)}. Good/bad SDR bands: "
        f"`> {CFG.CALIBRATION_SDR_GOOD:g} dB` / `< {CFG.CALIBRATION_SDR_BAD:g} dB` (§15.1). "
        f"{fitted_n} of {len(fits)} constants fitted; the rest fall back to their §16 initial "
        "(domain-shift discipline — synthetic sfizz renders are OOD for PANNs/separators, so a "
        "weak or guard-tripping fit must never silently overwrite trusted real-audio behavior).",
        "",
        format_fit_table(fits),
        "",
        "The `TIMELINE_ACTIVITY_THRESHOLD` row carries the §3.1c absent-fire guard (a fitted "
        "value that would read >20% of truth-active frames as inactive falls back); "
        "`EXPOSURE_THRESHOLD` is owned-audio-deferred. **The 6-song owned-audio ear panel (§15.1) "
        "remains HUMAN-deferred** — no synthetic run can stand in for ear-labeled real audio.",
        "",
    ]
    pin_report_path.parent.mkdir(parents=True, exist_ok=True)
    new_text = base.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n"
    pin_report_path.write_text(new_text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _select_working_engines() -> list[tuple[ModelEntry, None]]:
    from orpheus_mcp.analysis.model_registry import load_registry

    entries = {e.id: e for e in load_registry(allow_unpinned=True)}
    pairs: list[tuple[ModelEntry, None]] = []
    for eid in WORKING_ENGINE_IDS:
        if eid in entries and entries[eid].engine_status != "no-engine":
            pairs.append((entries[eid], None))
    return pairs


def _subset(fixture_dirs: list[Path], limit: int | None) -> list[Path]:
    """A documented >=12 subset covering all 8 genres: keep every genre, thinning densities."""
    if limit is None or limit >= len(fixture_dirs):
        return fixture_dirs
    by_genre: dict[str, list[Path]] = {}
    for d in sorted(fixture_dirs, key=lambda p: p.name):
        genre = d.name.split("_")[0]
        by_genre.setdefault(genre, []).append(d)
    picked: list[Path] = []
    round_i = 0
    while len(picked) < limit and any(len(v) > round_i for v in by_genre.values()):
        for genre in sorted(by_genre):
            if len(by_genre[genre]) > round_i and len(picked) < limit:
                picked.append(by_genre[genre][round_i])
        round_i += 1
    return sorted(picked, key=lambda p: p.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path.home() / ".orpheus_fixtures")
    parser.add_argument("--fixtures", type=int, default=None,
                        help="Calibrate on a documented >=12 subset (all 8 genres) if set.")
    parser.add_argument("--domain", default="synthetic-fixtures",
                        help="Provenance tag stamped on every constant fit in this run.")
    parser.add_argument("--ci-smoke", action="store_true",
                        help="Stub engines over the 2 committed CI fixtures — ZERO downloads.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fit and print the table only; do not write JSON or the pin report.")
    args = parser.parse_args(argv)

    if args.ci_smoke:
        ci_root = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "audio"
        manifest = verify_manifest_integrity(ci_root)
        fixture_dirs = sorted(ci_root / n for n in {r.split("/", 1)[0] for r in manifest["files"]})
        from stub_engines import StubTagger
        samples = collect_samples(fixture_dirs, _ci_smoke_engine_pairs(), StubTagger())
        fits = fit_all(samples)
        payload = build_json(fits, manifest["fixture_set_hash"], "ci-smoke")
        print(format_fit_table(fits))
        print(f"\nci-smoke: {samples.n_fixtures} fixtures, {samples.n_separations} separations, "
              f"{len(payload['constants'])} constants — ZERO downloads.")
        return 0

    from orpheus_mcp.analysis.timeline import PannsTagger

    manifest = verify_manifest_integrity(args.fixtures_dir)
    all_dirs = sorted(args.fixtures_dir / n
                      for n in {r.split("/", 1)[0] for r in manifest["files"]})
    fixture_dirs = _subset(all_dirs, args.fixtures)
    engine_pairs = _select_working_engines()
    if not engine_pairs:
        print("BLOCKED: no working desktop separator entries in the registry.")
        return 2

    print(f"Calibrating over {len(fixture_dirs)} fixtures × {len(engine_pairs)} engines "
          f"({', '.join(e.id for e, _ in engine_pairs)}) ...")
    samples = collect_samples(fixture_dirs, engine_pairs, PannsTagger())
    fits = fit_all(samples)
    print(format_fit_table(fits))

    if args.dry_run:
        return 0

    existing = None
    path = calibrated_thresholds_path()
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    payload = build_json(fits, manifest["fixture_set_hash"], args.domain, existing=existing)
    payload["n_fixtures"] = samples.n_fixtures
    out_path = write_json(payload)
    append_pin_section(fits, manifest["fixture_set_hash"], args.domain, samples.n_fixtures,
                       [e.id for e, _ in engine_pairs])
    print(f"\nWrote {out_path} and appended the pin section ({samples.n_separations} separations).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
