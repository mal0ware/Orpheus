"""Audio feature extraction — the post-FX 'sonic signature'.

numpy + soundfile + pyloudnorm only; librosa (the [analysis] extra) can sharpen spectral
features later but nothing here requires it. Pure functions over WAV paths; no REAPER.
The render tools that PRODUCE these WAVs are separate — this layer just measures.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from orpheus_mcp.models import AudioCharacter

# Band edges in Hz — the classic mix-speak split: lows (weight), mids (body),
# highs (air). Coarse on purpose: three numbers a musician can argue with beat
# thirty they can't.
_LOW_MAX_HZ = 250.0
_MID_MAX_HZ = 4000.0

_EPS = 1e-12
# "Nothing in this band" as a number (digital silence), so band deltas stay arithmetic.
_BAND_FLOOR_DB = -120.0


def analyze_audio_character(wav_path: str) -> AudioCharacter:
    """Measure the sonic signature of a rendered WAV.

    - lufs_integrated: BS.1770 integrated loudness via pyloudnorm (None when the file is
      shorter than one 400 ms gating block, or silent).
    - low/mid/high_energy_db: per-band RMS in dBFS via one numpy rFFT (band edges 250 Hz
      and 4 kHz). Compare bands RELATIVE to each other across files — the absolute
      numbers move with overall level.
    - spectral_centroid_hz: power-weighted mean frequency ("brightness").
    - true_peak_db: SAMPLE peak dBFS. Honest limitation: no oversampling, so real
      inter-sample true peak can be ~0.5 dB hotter.
    - crest_factor_db: peak minus RMS — punchy material is high, brickwalled is low.
    - stereo_width: side / (mid + side) RMS ratio — 0 = mono, 1 = pure anti-phase.
    """
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"No such audio file: {wav_path}")

    import soundfile as sf

    data, sample_rate = sf.read(str(path), always_2d=True)  # (frames, channels), float64
    mono = data.mean(axis=1)

    return AudioCharacter(
        lufs_integrated=_integrated_lufs(data, sample_rate),
        **_band_energies_db(mono, sample_rate),
        spectral_centroid_hz=_spectral_centroid_hz(mono, sample_rate),
        true_peak_db=_db(float(np.max(np.abs(data)))),
        crest_factor_db=_crest_factor_db(data),
        stereo_width=_stereo_width(data),
    )


def has_librosa() -> bool:
    """Whether the optional high-fidelity spectral path is available."""
    try:
        import librosa  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _db(linear: float) -> float | None:
    return 20.0 * math.log10(linear) if linear > _EPS else None


def _integrated_lufs(data: np.ndarray, sample_rate: int) -> float | None:
    import pyloudnorm

    try:
        lufs = float(pyloudnorm.Meter(sample_rate).integrated_loudness(data))
    except ValueError:
        # File shorter than one 400 ms gating block — measuring would be meaningless.
        return None
    return lufs if math.isfinite(lufs) else None  # silence gates to -inf


def _band_energies_db(mono: np.ndarray, sample_rate: int) -> dict[str, float | None]:
    spectrum = np.abs(np.fft.rfft(mono)) / max(len(mono), 1)
    power = spectrum**2
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)

    def band_rms_db(lo: float, hi: float) -> float:
        band_power = float(power[(freqs >= lo) & (freqs < hi)].sum())
        # x2: rfft folds negative frequencies; RMS^2 == total spectral power (Parseval).
        # Floored, not None: "this band is empty" is a measurement, and the fingerprint
        # diff needs a number to subtract.
        db = _db(math.sqrt(2.0 * band_power))
        return _BAND_FLOOR_DB if db is None else max(db, _BAND_FLOOR_DB)

    return {
        "low_energy_db": band_rms_db(0.0, _LOW_MAX_HZ),
        "mid_energy_db": band_rms_db(_LOW_MAX_HZ, _MID_MAX_HZ),
        "high_energy_db": band_rms_db(_MID_MAX_HZ, sample_rate / 2.0),
    }


def _spectral_centroid_hz(mono: np.ndarray, sample_rate: int) -> float | None:
    spectrum = np.abs(np.fft.rfft(mono))
    power = spectrum**2
    total = float(power.sum())
    if total <= _EPS:
        return None  # silence has no brightness
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    return float((freqs * power).sum() / total)


def _crest_factor_db(data: np.ndarray) -> float | None:
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))
    if peak <= _EPS or rms <= _EPS:
        return None
    return 20.0 * math.log10(peak / rms)


def _stereo_width(data: np.ndarray) -> float:
    if data.shape[1] < 2:
        return 0.0
    left, right = data[:, 0], data[:, 1]
    mid_rms = float(np.sqrt(np.mean(((left + right) / 2.0) ** 2)))
    side_rms = float(np.sqrt(np.mean(((left - right) / 2.0) ** 2)))
    total = mid_rms + side_rms
    return side_rms / total if total > _EPS else 0.0
