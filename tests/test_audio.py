"""M2 audio character: LUFS, 3-band balance, centroid, peak/crest/width.

Fixtures are tiny synthesized WAVs (numpy sine/noise via soundfile) written to tmp_path —
nothing binary is committed, no REAPER, no render tools involved.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from orpheus_mcp.analysis.audio import analyze_audio_character

SR = 48000


def _write_wav(path, data: np.ndarray, sr: int = SR) -> str:
    sf.write(str(path), data, sr)
    return str(path)


def _sine(freq: float, seconds: float = 3.0, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_audio_character(str(tmp_path / "nope.wav"))


def test_1khz_sine_centroid_and_mid_dominance(tmp_path):
    wav = _write_wav(tmp_path / "mid.wav", _sine(1000.0))
    result = analyze_audio_character(wav)
    assert result.spectral_centroid_hz == pytest.approx(1000.0, abs=50.0)
    assert result.mid_energy_db is not None
    assert result.mid_energy_db > result.low_energy_db
    assert result.mid_energy_db > result.high_energy_db


def test_60hz_sine_low_band_dominates(tmp_path):
    wav = _write_wav(tmp_path / "low.wav", _sine(60.0))
    result = analyze_audio_character(wav)
    assert result.low_energy_db > result.mid_energy_db
    assert result.low_energy_db > result.high_energy_db
    assert result.spectral_centroid_hz == pytest.approx(60.0, abs=20.0)


def test_8khz_sine_high_band_dominates(tmp_path):
    wav = _write_wav(tmp_path / "high.wav", _sine(8000.0))
    result = analyze_audio_character(wav)
    assert result.high_energy_db > result.low_energy_db
    assert result.high_energy_db > result.mid_energy_db


def test_lufs_tracks_level(tmp_path):
    loud = analyze_audio_character(_write_wav(tmp_path / "loud.wav", _sine(1000.0, amplitude=0.5)))
    quiet = analyze_audio_character(
        _write_wav(tmp_path / "quiet.wav", _sine(1000.0, amplitude=0.05))
    )
    assert loud.lufs_integrated is not None and quiet.lufs_integrated is not None
    # 10x amplitude ratio = 20 dB; allow slack for gating.
    assert loud.lufs_integrated - quiet.lufs_integrated == pytest.approx(20.0, abs=2.0)
    assert -30.0 < loud.lufs_integrated < 0.0


def test_peak_and_crest_of_sine(tmp_path):
    result = analyze_audio_character(_write_wav(tmp_path / "sine.wav", _sine(1000.0)))
    # amplitude 0.5 -> sample peak -6.02 dBFS; sine crest factor is 3.01 dB.
    assert result.true_peak_db == pytest.approx(-6.02, abs=0.3)
    assert result.crest_factor_db == pytest.approx(3.01, abs=0.5)


def test_stereo_width_mono_content_is_zero(tmp_path):
    mono = _sine(440.0)
    stereo = np.column_stack([mono, mono])
    result = analyze_audio_character(_write_wav(tmp_path / "monoish.wav", stereo))
    assert result.stereo_width == pytest.approx(0.0, abs=0.01)


def test_stereo_width_antiphase_is_full(tmp_path):
    mono = _sine(440.0)
    stereo = np.column_stack([mono, -mono])
    result = analyze_audio_character(_write_wav(tmp_path / "wide.wav", stereo))
    assert result.stereo_width == pytest.approx(1.0, abs=0.01)


def test_mono_file_has_zero_width(tmp_path):
    result = analyze_audio_character(_write_wav(tmp_path / "mono.wav", _sine(440.0)))
    assert result.stereo_width == 0.0


def test_too_short_for_lufs_still_measures_spectrum(tmp_path):
    # pyloudnorm needs >= one 400 ms block; a 100 ms file must degrade, not crash.
    result = analyze_audio_character(
        _write_wav(tmp_path / "short.wav", _sine(1000.0, seconds=0.1))
    )
    assert result.lufs_integrated is None
    assert result.spectral_centroid_hz == pytest.approx(1000.0, abs=100.0)


def test_silence_does_not_crash(tmp_path):
    result = analyze_audio_character(
        _write_wav(tmp_path / "silence.wav", np.zeros(SR, dtype=np.float64))
    )
    assert result.lufs_integrated is None  # -inf gated out
    assert result.spectral_centroid_hz is None
