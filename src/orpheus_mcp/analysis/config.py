"""ALL analysis tunables (§16). Nothing is hard-coded at a call site.
Rows marked 'cal' are re-fit by scripts/calibrate_gates.py, which records the
fixture_set_hash it fit against (cache invalidation, §9).

**Calibration override loading (D7, Task 5.3):** when ``data/calibrated_thresholds.json``
exists it is loaded AT IMPORT and its fitted values overwrite the matching ``# cal`` fields
on the module-level :data:`CFG` singleton. Each override carries provenance
(``{"value": …, "fit_domain": …, "fixture_set_hash": …}``) so a synthetic-fixture fit and a
later owned-audio fit can coexist and be told apart. Because ``cachefs.verify_config_hash``
reads the LIVE ``# cal`` field values, an override load automatically busts stale gate-artifact
cache entries — no separate cache-invalidation step is needed (verified by test). Only names
that are actually ``# cal``-marked are ever applied: an unknown key in the JSON is ignored, so
the file can never silently mutate a non-calibration tunable."""
from __future__ import annotations

import inspect
import json
import re
import warnings
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class AnalysisConfig:
    GATE_MIX_RESIDUAL_PASS_DB: float = -20.0        # cal
    GATE_BLEED_PASS: float = 0.20                   # cal
    GATE_SILENCE_PASS_DB: float = -35.0             # cal
    GATE_SILENCE_TAIL_ALLOWANCE_S: float = 3.0      # cal
    GATE_AGREEMENT_PASS: float = 0.90               # cal
    TIMELINE_ACTIVITY_THRESHOLD: float = 0.30       # cal
    EXPOSURE_THRESHOLD: float = 0.15                # cal
    VAD_VOCAL_ACTIVE_DBFS: float = -45.0            # cal
    BEAT_GRID_CONFIDENCE_MIN: float = 0.60
    FIXED_GRID_BARS_TO_SECONDS: float = 2.0
    CHORD_STABILITY_RATIO: float = 0.60
    CHORD_EXTENSION_MIN_BEATS: int = 2
    CHORD_EXTENSION_ENERGY_RATIO: float = 0.15      # cal
    KEY_WINDOW_BARS: int = 16
    KEY_HOP_BARS: int = 4
    KEY_HYSTERESIS_WINDOWS: int = 2
    LYRIC_MATCH_MIN_SCORE: float = 0.60
    FX_LOOP_MAX_ITERATIONS: int = 4
    FX_LOOP_PLATEAU_EPSILON: float = 0.05
    WIDEN_MONO_SAFETY_CORR_MIN: float = 0.50
    DEFAULT_MAX_TIER: str = "T4"
    DEFAULT_MAX_SECONDS: float = 300.0
    VRAM_FIT_FRACTION: float = 0.80
    CACHE_SIZE_CAP_GB: float = 20.0
    # --- plan-added (allowed by §16's normative rule; NOT in the spec table) ---
    SEPARATION_CONTEXT_PAD_S: float = 5.0           # §3.1 T3 crop pad
    CHORD_SELF_TRANSITION: float = 0.90             # §13.1 Viterbi stay-probability
    CHORD_EMISSION_SHARPNESS: float = 8.0           # §13.1 emission temperature (harmony.py)
    CHORD_QUIET_ENERGY_RATIO: float = 0.15          # §13.1 no-chord below this x median energy
    EXPOSURE_MIN_WINDOW_S: float = 3.0              # shortest usable exposed window (§3.1b)
    ANALYSIS_SR: int = 44100                        # §8 common analysis rate
    CACHE_DIR_ENV: str = "ORPHEUS_CACHE_DIR"        # test override hook
    CALIBRATION_SDR_GOOD: float = 7.0               # §15.1 ROC "good" floor
    CALIBRATION_SDR_BAD: float = 4.0                # §15.1 ROC "bad" ceiling
    # --- §7 SoundCharacter descriptors (Task 6.1, plan-added) ---
    DUAL_MONO_TOLERANCE: float = 1e-4               # |L-R| ≤ this ⇒ dual-mono is channels=1
    SUSTAIN_MIN_WINDOW_S: float = 0.4               # §7 flavor: onset-free window floor
    HARMONIC_TOLERANCE_HZ: float = 15.0             # ± window for per-harmonic energy tally
    FLAVOR_EVEN_ODD_DB: float = 3.0                 # §7 even−odd warm-even/gritty-odd threshold
    DESCRIPTOR_ENV_HOP_MS: float = 1.0              # amplitude-envelope hop for delay autocorr
    MOD_ENV_HOP_MS: float = 2.0                     # envelope hop for modulation-rate detection
    DELAY_MIN_MS: float = 5.0                       # delay-search lag floor (below = comb/timbre)
    DELAY_MAX_MS: float = 600.0                     # delay-search lag ceiling
    MOD_RATE_MIN_HZ: float = 0.5                    # modulation-rate search band
    MOD_RATE_MAX_HZ: float = 20.0
    MOD_DETECT_MIN_DEPTH: float = 0.05              # below this normalized depth ⇒ no modulation
    BANDWIDTH_BAND_HZ: float = 1000.0               # spectral-cliff band width
    BANDWIDTH_CLIFF_DROP_DB: float = 40.0           # §7 "within 40 dB of peak band" cliff rule
    SPECTRAL_TILT_LO_HZ: float = 80.0               # tilt octave-regression band
    SPECTRAL_TILT_HI_HZ: float = 16000.0
    SPECTRAL_TILT_BANDS: int = 14
    DESCRIPTOR_STFT_N: int = 2048                   # STFT size for per-frame band-dynamics
    ROLLOFF_FRACTION: float = 0.85                  # §7 spectral rolloff energy fraction
    STEREO_BAND_LOW_HZ: float = 250.0               # §7 stereo width low/mid crossover
    STEREO_BAND_HIGH_HZ: float = 4000.0             # §7 stereo width mid/high crossover
    REVERB_EARLY_LATE_SPLIT_S: float = 0.08         # §7 early/late reflection split
    RT60_SHORT_MAX_S: float = 0.4                   # §7 RT60 short/medium bucket boundary
    RT60_MEDIUM_MAX_S: float = 1.2                  # §7 RT60 medium/long bucket boundary
    RT60_SLOPE_GATE_DB_PER_S: float = -1.0          # §7 "no measurable decay" slope gate
    PITCH_FMIN_HZ: float = 50.0                     # §7 yin pitch-track search floor
    PITCH_FMAX_HZ: float = 2000.0                   # §7 yin pitch-track search ceiling
    ATTACK_PEAK_SEARCH_FRAMES: int = 20             # §7 attack-time peak search window (frames)
    ATTACK_HOP_SAMPLES: int = 512                   # §7 attack-time onset-strength hop
    LUFS_SHORT_TERM_HOP_S: float = 1.0              # §7 short-term LUFS sliding-window hop
    # --- §14 lyrics pipeline (Task 8.1, plan-added) ---
    VAD_FRAME_MS: float = 20.0                      # §14.1 VAD frame length for per-frame RMS
    VAD_HOP_MS: float = 10.0                        # §14.1 VAD frame hop
    VAD_MERGE_GAP_S: float = 0.20                   # bridge active frames closer than this
    VAD_MIN_WINDOW_S: float = 0.10                  # drop active windows shorter than this
    LYRIC_LINE_GAP_S: float = 1.0                   # §14.1.4 new line when word gap exceeds this
    LYRIC_SENTENCE_PUNCT: str = ".!?"               # §14.1.4 line-ending punctuation
    LYRIC_CHORUS_SIMILARITY: float = 0.80           # §14.1.4 near-identical line match ratio
    LYRIC_SECTION_SNAP_S: float = 1.0               # §14.1.4 _snap_to_recon edge-snap tolerance


# --------------------------------------------------------------------------- #
# Calibration override loading (D7, §15.1 / Task 5.3)
# --------------------------------------------------------------------------- #

# Matches ``NAME: type = value    # cal`` rows in the AnalysisConfig source above — the same
# regex ``cachefs._cal_field_names`` uses, kept here too so this module never imports cachefs
# (which imports THIS module) just to learn which fields are calibratable.
_CAL_COMMENT = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*:.*#\s*cal\b", re.MULTILINE)


def cal_field_names() -> list[str]:
    """Names of every ``# cal``-marked :class:`AnalysisConfig` field, read from the live
    source via ``inspect.getsource`` so it can never drift from config.py's own comments."""
    return sorted(set(_CAL_COMMENT.findall(inspect.getsource(AnalysisConfig))))


def calibrated_thresholds_path() -> Path:
    """Locate ``data/calibrated_thresholds.json`` in both layouts (installed wheel: inside the
    package; dev checkout: repo root next to ``src/``) — same two-layout probe as
    ``model_registry.registry_path``."""
    package_path = Path(__file__).resolve().parents[1] / "data" / "calibrated_thresholds.json"
    if package_path.is_file():
        return package_path
    return Path(__file__).resolve().parents[3] / "data" / "calibrated_thresholds.json"


# The pristine §16 initial values, captured before any override loads, so ``reset_calibration``
# can restore them (test isolation; the file is loaded once at import into the shared singleton).
_INITIAL_VALUES: dict[str, object] = {}


def reset_calibration(*, cfg: AnalysisConfig | None = None) -> None:
    """Restore every ``# cal`` field on ``cfg`` (default: the :data:`CFG` singleton) to its
    §16 initial value — the inverse of :func:`load_calibration_overrides`, for test teardown."""
    target = cfg if cfg is not None else CFG
    for name, value in _INITIAL_VALUES.items():
        setattr(target, name, value)


def load_calibration_overrides(
    path: str | Path | None = None, *, cfg: AnalysisConfig | None = None
) -> list[str]:
    """Load ``data/calibrated_thresholds.json`` (or an explicit ``path``) and overwrite the
    matching ``# cal`` fields on ``cfg`` (default: :data:`CFG`) with the fitted ``value`` of
    each provenance-tagged entry. Returns the list of field names actually applied.

    Schema (per the D7 provenance contract)::

        {"schema_version": 1, "fit_domain": "synthetic-fixtures", "fixture_set_hash": "…",
         "constants": {"GATE_BLEED_PASS": {"value": 0.18, "fit_domain": "…",
                                           "fixture_set_hash": "…", "fallback": false}, …}}

    A bare scalar is also accepted in place of the ``{"value": …}`` object (forward tolerance).
    Unknown / non-``# cal`` keys warn (naming the key) and are skipped rather than applied; a
    malformed ``value`` for a known constant (e.g. one that can't coerce to the field's runtime
    type) also warns (naming the constant + offending value) and is skipped, WITHOUT aborting the
    rest of the file. A missing, empty, or fully malformed (unparseable JSON) file applies
    nothing (returns ``[]``) rather than raising — a box with no calibration simply runs the
    §16 initials. This function must never raise: it runs at import time on everything's import
    graph."""
    target = cfg if cfg is not None else CFG
    p = Path(path) if path is not None else calibrated_thresholds_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    constants = data.get("constants", {}) if isinstance(data, dict) else {}
    if not isinstance(constants, dict):
        return []

    allowed = {f.name: f.type for f in fields(target)}
    cal = set(cal_field_names())
    applied: list[str] = []
    for name, spec in constants.items():
        if name not in cal or name not in allowed:
            warnings.warn(
                f"load_calibration_overrides: skipping unknown/non-calibration key {name!r} "
                f"in {p}", stacklevel=2,
            )
            continue
        value = spec.get("value") if isinstance(spec, dict) else spec
        if value is None:
            continue
        current = getattr(target, name)
        # Preserve the field's runtime type (every # cal field is a float today; an int-typed
        # one would coerce cleanly rather than smuggle a float in). A malformed value (e.g. a
        # string that doesn't parse as a number) must never take down every importer of this
        # module — warn (naming the constant + offending value) and skip it, keeping the box on
        # its safe §16 initial for that one constant while the rest of the file still applies.
        try:
            setattr(target, name, type(current)(value))
        except (ValueError, TypeError) as exc:
            warnings.warn(
                f"load_calibration_overrides: skipping malformed value {value!r} for "
                f"calibration constant {name!r} in {p}: {exc}", stacklevel=2,
            )
            continue
        applied.append(name)
    return applied


CFG = AnalysisConfig()
_INITIAL_VALUES.update({name: getattr(CFG, name) for name in cal_field_names()})
# D7: fitted overrides (if the file exists) load NOW, at import, into the shared singleton, so
# every downstream reader — gates, verify_config_hash, tools — sees the calibrated values.
load_calibration_overrides()
