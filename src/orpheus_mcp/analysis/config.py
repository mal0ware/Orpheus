"""ALL analysis tunables (§16). Nothing is hard-coded at a call site.
Rows marked 'cal' are re-fit by scripts/calibrate_gates.py, which records the
fixture_set_hash it fit against (cache invalidation, §9)."""
from __future__ import annotations

from dataclasses import dataclass


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
    EXPOSURE_MIN_WINDOW_S: float = 3.0              # shortest usable exposed window (§3.1b)
    ANALYSIS_SR: int = 44100                        # §8 common analysis rate
    CACHE_DIR_ENV: str = "ORPHEUS_CACHE_DIR"        # test override hook
    CALIBRATION_SDR_GOOD: float = 7.0               # §15.1 ROC "good" floor
    CALIBRATION_SDR_BAD: float = 4.0                # §15.1 ROC "bad" ceiling


CFG = AnalysisConfig()
