"""``analysis/descriptors.py`` + ``tools/reference.py::characterize`` (Task 6.1, §7): the
constructed-signal acceptance bars for the ``SoundCharacter`` descriptors, plus the
``resolve_source`` integration duties (§6 R3 complement flag, §7 dual-mono channel honesty).

Every signal is generated in-test with numpy + soundfile — no model weights, no network, no
GPU — so the numeric bars (28 ms echo, −6 dB/oct tilt, 5 Hz tremolo, −20 dBFS LUFS/crest,
even/odd flavor, 16 kHz bandwidth cliff) are self-contained and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
from orpheus_mcp.analysis.descriptors import (
    AudioSource,
    compute_character,
    resolve_source,
)
from orpheus_mcp.models import SourceMeta, StemHandle, Window
from orpheus_mcp.tools.reference import characterize
from stub_engines import StubEngine

SR = 44100


# --------------------------------------------------------------------------- #
# Test source builders
# --------------------------------------------------------------------------- #


def _source(sig: np.ndarray, *, sr: int = SR, stem_type: str = "mix-region") -> AudioSource:
    if sig.ndim < 2:
        sig = sig.reshape(-1, 1)
    sig = np.ascontiguousarray(sig, dtype=np.float32)
    pcm = Pcm(
        samples=sig,
        sample_rate=sr,
        path="",
        pcm_hash="test",
        duration_s=sig.shape[0] / sr,
        decoder="test",
    )
    meta = SourceMeta(
        stem_type=stem_type,
        produced_by="test",
        region=None,
        sample_rate=sr,
        channels=sig.shape[1] if _true_stereo(sig) else 1,
    )
    return AudioSource(pcm=pcm, meta=meta)


def _true_stereo(sig: np.ndarray) -> bool:
    return sig.ndim == 2 and sig.shape[1] == 2 and not np.allclose(sig[:, 0], sig[:, 1])


def mono_source(sig: np.ndarray, *, stem_type: str = "mix-region") -> AudioSource:
    return _source(sig, stem_type=stem_type)


# --------------------------------------------------------------------------- #
# Signal generators
# --------------------------------------------------------------------------- #


def _click_train_with_echo(echo_ms: float = 28.0, fb: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dur = 1.0
    n = int(SR * dur)
    x = np.zeros(n, dtype=np.float32)
    positions = sorted(rng.uniform(0.05, 0.9, 6))
    lag = int(round(echo_ms / 1000.0 * SR))
    for p in positions:
        i = int(p * SR)
        if i < n:
            x[i] += 1.0
        j = i + lag
        if j < n:
            x[j] += fb
    return x


def _tilted_noise(db_per_oct: float = -6.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * 2)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    gain = np.ones_like(freqs)
    # amplitude ∝ f**(db_per_oct/6) — for −6 dB/oct that is 1/f.
    gain[1:] = (100.0 / freqs[1:]) ** (abs(db_per_oct) / 6.0)
    return np.fft.irfft(spectrum * gain, n).astype(np.float32)


def _tremolo(fm: float = 5.0, fc: float = 440.0, depth: float = 0.6) -> np.ndarray:
    t = np.arange(int(SR * 2)) / SR
    carrier = np.sin(2 * np.pi * fc * t)
    return ((1.0 + depth * np.sin(2 * np.pi * fm * t)) * carrier).astype(np.float32)


def _sine_at_dbfs_rms(dbfs: float = -20.0, freq: float = 1000.0, dur: float = 3.0) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    amp = (10.0 ** (dbfs / 20.0)) * np.sqrt(2.0)  # RMS at dbFS ⇒ peak = rms·√2
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _two_tone(f0: float, overtone: float) -> np.ndarray:
    t = np.arange(int(SR * 1.0)) / SR
    return (0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * overtone * t)).astype(
        np.float32
    )


def _brickwall_noise(cutoff_hz: float = 16000.0, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * 1.0)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spectrum[freqs >= cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, n).astype(np.float32)


# --------------------------------------------------------------------------- #
# The locked constructed-signal bars (§10/§12 slice-6)
# --------------------------------------------------------------------------- #


def test_synthetic_28ms_echo_reads_back():
    ch = compute_character(mono_source(_click_train_with_echo(echo_ms=28.0)))
    assert ch.time_fx["delay_ms"] == pytest.approx(28, abs=3)
    assert ch.time_fx["delay_approx"] is True


def test_tilt_6db_per_oct():
    ch = compute_character(mono_source(_tilted_noise(-6.0)))
    tilt = ch.spectral["tilt_db_per_oct"]
    assert tilt is not None
    assert -7.0 <= tilt <= -5.0


def test_tremolo_5hz_detected():
    ch = compute_character(mono_source(_tremolo(fm=5.0)))
    assert ch.time_fx["mod_rate_hz"] == pytest.approx(5.0, abs=1.0)
    assert ch.time_fx["mod_class"] == "tremolo"


def test_mono_source_stereo_descriptors_are_none():
    ch = compute_character(mono_source(_sine_at_dbfs_rms()))
    assert ch.stereo == {
        "width_low": None,
        "width_mid": None,
        "width_high": None,
        "corr": None,
    }


def test_crest_and_lufs_on_reference_tone():
    ch = compute_character(mono_source(_sine_at_dbfs_rms(-20.0)))
    assert ch.dynamics["lufs_i"] == pytest.approx(-20.0, abs=1.5)
    assert ch.dynamics["crest_db"] == pytest.approx(3.01, abs=0.3)


@pytest.mark.analysis
def test_even_vs_odd_flavor():
    warm = compute_character(mono_source(_two_tone(200.0, 400.0)))
    gritty = compute_character(mono_source(_two_tone(200.0, 600.0)))
    assert warm.harmonics["flavor"] == "warm-even"
    assert gritty.harmonics["flavor"] == "gritty-odd"


def test_bandwidth_cliff_detected():
    src = mono_source(_brickwall_noise(16000.0))
    ch = compute_character(src)
    assert ch.source.bandwidth_hz is not None
    assert 15000.0 <= ch.source.bandwidth_hz <= 17000.0


def test_vocal_reverb_uses_dereverb_when_available():
    sig = _sine_at_dbfs_rms(-20.0, dur=1.0)

    vocal_engine = StubEngine()
    vocal = compute_character(mono_source(sig, stem_type="vocals"), dereverb=vocal_engine)
    assert len(vocal_engine.calls) > 0, "dereverb must be called for a vocals stem"
    assert vocal.time_fx["reverb_approx"] is False

    drum_engine = StubEngine()
    drums = compute_character(mono_source(sig, stem_type="drums"), dereverb=drum_engine)
    assert len(drum_engine.calls) == 0, "dereverb must NOT be called for a non-vocals stem"
    assert drums.time_fx["reverb_approx"] is True


def test_vocal_reverb_no_engine_notes_fallback():
    """A vocals source with no dereverb engine injected (``dereverb=None``) still falls back to
    the decay-slope heuristic (``reverb_approx=True``) but, unlike a non-vocals source, MUST
    say why in ``notes`` — mirroring the engine-failure sub-path's wording (review finding)."""
    sig = _sine_at_dbfs_rms(-20.0, dur=1.0)
    ch = compute_character(mono_source(sig, stem_type="vocals"), dereverb=None)
    assert ch.time_fx["reverb_approx"] is True
    assert any("no dereverb engine" in n for n in ch.notes), ch.notes


# --------------------------------------------------------------------------- #
# CFG-hoisted descriptor tunables (review finding, §16) — the DSP call sites must read the
# live CFG singleton, not a frozen default captured at function-definition time.
# --------------------------------------------------------------------------- #


def test_descriptor_cfg_constants_exist():
    from orpheus_mcp.analysis.config import CFG

    for name in (
        "ROLLOFF_FRACTION",
        "STEREO_BAND_LOW_HZ",
        "STEREO_BAND_HIGH_HZ",
        "REVERB_EARLY_LATE_SPLIT_S",
        "RT60_SHORT_MAX_S",
        "RT60_MEDIUM_MAX_S",
        "RT60_SLOPE_GATE_DB_PER_S",
        "PITCH_FMIN_HZ",
        "PITCH_FMAX_HZ",
        "ATTACK_PEAK_SEARCH_FRAMES",
        "ATTACK_HOP_SAMPLES",
        "LUFS_SHORT_TERM_HOP_S",
    ):
        assert hasattr(CFG, name), f"AnalysisConfig missing hoisted tunable {name!r}"


def test_rolloff_reads_live_cfg(monkeypatch):
    from orpheus_mcp.analysis.config import CFG

    sig = _tilted_noise(-6.0)
    baseline = compute_character(mono_source(sig)).spectral["rolloff_hz"]

    monkeypatch.setattr(CFG, "ROLLOFF_FRACTION", 0.10)
    lowered = compute_character(mono_source(sig)).spectral["rolloff_hz"]

    assert baseline is not None and lowered is not None
    assert lowered < baseline, "a lower rolloff fraction must read back a lower rolloff_hz"


# --------------------------------------------------------------------------- #
# resolve_source integration duties (§6 R3 complement, §7 channel honesty)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fs(tmp_path) -> CacheFS:
    return CacheFS(tmp_path / "cache")


def _write_wav(path: Path, sig: np.ndarray, sr: int = SR) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sig.ndim < 2:
        sig = sig.reshape(-1, 1)
    sf.write(str(path), sig.astype(np.float32), sr, subtype="FLOAT")
    return str(path)


def test_resolve_source_literal_path(tmp_path, fs):
    path = _write_wav(tmp_path / "mix.wav", _sine_at_dbfs_rms(-20.0, dur=1.0))
    src = resolve_source(path, fs)
    assert src.meta.stem_type == "mix-region"
    assert src.meta.produced_by == "file"
    assert src.meta.complement is False
    assert src.meta.channels == 1


def test_resolve_source_r3_complement_flag(tmp_path, fs):
    path = _write_wav(tmp_path / "residual.wav", _sine_at_dbfs_rms(-20.0, dur=1.0))
    handle = StemHandle(
        stem_path=path,
        stem_type="vocals",
        produced_by="complement(x − y)",
        region=Window(start_s=0.0, end_s=1.0),
        notes=["complement=True (§6 R3 residual)"],
    )
    src = resolve_source(handle, fs)
    assert src.meta.complement is True
    assert src.meta.stem_type == "vocals"


def test_resolve_source_dual_mono_counts_as_mono(tmp_path, fs):
    mono = _sine_at_dbfs_rms(-20.0, dur=1.0)
    dual = np.stack([mono, mono], axis=1)  # two identical channels
    path = _write_wav(tmp_path / "dual.wav", dual)
    src = resolve_source(path, fs)
    assert src.meta.channels == 1, "dual-mono must report channels=1 (§7 mono policy)"
    ch = compute_character(src)
    assert ch.stereo["corr"] is None


def test_resolve_source_true_stereo(tmp_path, fs):
    left = _sine_at_dbfs_rms(-20.0, freq=440.0, dur=1.0)
    right = _sine_at_dbfs_rms(-20.0, freq=660.0, dur=1.0)  # decorrelated ⇒ genuine stereo
    stereo = np.stack([left, right], axis=1)
    path = _write_wav(tmp_path / "stereo.wav", stereo)
    src = resolve_source(path, fs)
    assert src.meta.channels == 2
    ch = compute_character(src)
    assert ch.stereo["corr"] is not None
    assert ch.stereo["width_mid"] is not None


# --------------------------------------------------------------------------- #
# characterize tool: descriptor cache round-trip (§9)
# --------------------------------------------------------------------------- #


def test_characterize_caches_descriptors(tmp_path, fs, monkeypatch):
    path = _write_wav(tmp_path / "mix.wav", _sine_at_dbfs_rms(-20.0, dur=1.0))

    calls = {"n": 0}
    from orpheus_mcp.analysis import descriptors as desc

    real = desc.compute_character

    def counted(src, *, dereverb=None):
        calls["n"] += 1
        return real(src, dereverb=dereverb)

    # characterize does ``from ...descriptors import compute_character`` at call time, so
    # patching the module attribute is seen by that local import.
    monkeypatch.setattr(desc, "compute_character", counted)

    first = characterize(path, cache=fs)
    second = characterize(path, cache=fs)
    assert calls["n"] == 1, "second characterize must hit the descriptor cache"
    assert first.model_dump() == second.model_dump()
