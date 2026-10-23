"""``analysis/separation.py`` (Task 4.1, §3.1d/§5/§7/§15.5): the region-crop pad/trim math,
EOF clamp, length-alignment, VRAM fallback, and the GPU-semaphore serialization — all driven
by ``StubEngine`` (no real weights, no GPU) in the CI-safe suite. The real-engine end-to-end
smoke (BS-RoFormer-family via audio-separator AND vendored SCNet) is ``@pytest.mark.desktop``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from orpheus_mcp.analysis import model_registry
from orpheus_mcp.analysis.cachefs import Pcm
from orpheus_mcp.analysis.separation import separate_region
from orpheus_mcp.models import StemResult, Window
from stub_engines import StubEngine

SR = 8000  # low rate keeps the constructed-signal tests fast; the math is rate-agnostic
FOUR_STEM = ["vocals", "drums", "bass", "other"]


@pytest.fixture(autouse=True)
def _plenty_of_vram(monkeypatch):
    """Keep the stub-engine tests hermetic: pin ``free_vram_gb`` to a large value so no test
    depends on the box's real GPU state and none imports torch (the real ``free_vram_gb``
    imports torch, which is both slow and — imported first-time across the semaphore test's
    threads — hits torch's TORCH_LIBRARY double-registration bug). The VRAM-fallback test
    overrides this to 1.0 in its own body."""
    monkeypatch.setattr(model_registry, "free_vram_gb", lambda: 24.0)


def _ramp_pcm(duration_s: float, *, sr: int = SR, channels: int = 1) -> Pcm:
    """A PCM whose sample value encodes its index (``x[i] = i / N``), so a trimmed stem's
    values pin down EXACTLY which source samples survived — a sample-accurate crop/trim probe."""
    n = int(round(duration_s * sr))
    ramp = (np.arange(n, dtype=np.float64) / n).astype(np.float32)
    samples = np.repeat(ramp.reshape(-1, 1), channels, axis=1)
    return Pcm(
        samples=samples,
        sample_rate=sr,
        path="<ramp>",
        pcm_hash="ramp",
        duration_s=n / sr,
        decoder="test",
    )


def _entry(model_id: str) -> model_registry.ModelEntry:
    return {e.id: e for e in model_registry.load_registry(allow_unpinned=True)}[model_id]


def _read(stem_path: str) -> np.ndarray:
    import soundfile as sf

    data, _sr = sf.read(stem_path, always_2d=True, dtype="float32")
    return data


# --------------------------------------------------------------------------- #
# 1. pad + trim math — region [10,20] on a 60 s file → engine sees [5,25], out is [10,20]
# --------------------------------------------------------------------------- #


def test_pad_and_trim_is_sample_accurate():
    pcm = _ramp_pcm(60.0)
    engine = StubEngine(FOUR_STEM)
    # htdemucs-ft is VRAM 0.0 (CPU-capable) → never triggers a fallback regardless of the box.
    out = separate_region(pcm, _entry("4stem-c/htdemucs-ft"), Window(start_s=10.0, end_s=20.0),
                          engine=engine)

    # The engine was handed the padded crop [5, 25] = 20 s (±CFG.SEPARATION_CONTEXT_PAD_S=5).
    assert engine.last_call["duration_s"] == pytest.approx(20.0, abs=1e-6)

    assert set(out) == set(FOUR_STEM)
    result = out["vocals"]
    assert isinstance(result, StemResult)
    assert result.region_trimmed == Window(start_s=10.0, end_s=20.0)
    assert result.sample_rate == SR

    stem = _read(result.stem_path)
    expected = pcm.samples[10 * SR : 20 * SR]
    assert stem.shape[0] == 10 * SR
    np.testing.assert_allclose(stem, expected, atol=1e-4)


# --------------------------------------------------------------------------- #
# 2. EOF clamp — region [55,70] on a 60 s file → in [50,60], out [55,60], region_trimmed says so
# --------------------------------------------------------------------------- #


def test_region_past_eof_clamps_and_reports():
    pcm = _ramp_pcm(60.0)
    engine = StubEngine(FOUR_STEM)
    out = separate_region(pcm, _entry("4stem-c/htdemucs-ft"), Window(start_s=55.0, end_s=70.0),
                          engine=engine)

    # Crop is [55-5, 70+5] clamped to [50, 60] = 10 s (not 25 s — the +5 pad + the +10 s
    # overshoot both clamp at EOF).
    assert engine.last_call["duration_s"] == pytest.approx(10.0, abs=1e-6)

    result = out["drums"]
    assert result.region_trimmed == Window(start_s=55.0, end_s=60.0)
    stem = _read(result.stem_path)
    assert stem.shape[0] == 5 * SR
    np.testing.assert_allclose(stem, pcm.samples[55 * SR : 60 * SR], atol=1e-4)
    # The clamp is surfaced, not silent.
    assert any("clamp" in note.lower() for note in result.notes)


# --------------------------------------------------------------------------- #
# 3. length-alignment — a stem padded 137 samples long is aligned back to the mix length
# --------------------------------------------------------------------------- #


def test_overlong_stem_is_length_aligned_to_region():
    pcm = _ramp_pcm(12.0)
    engine = StubEngine(FOUR_STEM, extra_pad_samples=137)
    # region=None ⇒ whole file; the 137-sample tail padding must be trimmed off, not left on.
    out = separate_region(pcm, _entry("4stem-c/htdemucs-ft"), None, engine=engine)

    n = int(round(12.0 * SR))
    for tgt in FOUR_STEM:
        stem = _read(out[tgt].stem_path)
        assert stem.shape[0] == n, f"{tgt}: {stem.shape[0]} != {n}"
    np.testing.assert_allclose(_read(out["bass"].stem_path), pcm.samples, atol=1e-4)


# --------------------------------------------------------------------------- #
# 4. VRAM fallback — 6.5 GB entry with 1 GB free falls to the htdemucs-ft CPU candidate
# --------------------------------------------------------------------------- #


def test_vram_fallback_degrades_to_cpu_candidate(monkeypatch):
    monkeypatch.setattr(model_registry, "free_vram_gb", lambda: 1.0)
    pcm = _ramp_pcm(8.0)
    engine = StubEngine(FOUR_STEM)  # injected stub stands in for whatever engine gets chosen
    # 4stem-a/bsroformer needs 6.5 GB; 0.8 × 1.0 = 0.8 GB free → does NOT fit → resolve() falls
    # to the exact-stem-set CPU candidate (htdemucs-ft, vram 0.0).
    out = separate_region(pcm, _entry("4stem-a/bsroformer"), Window(start_s=2.0, end_s=6.0),
                          engine=engine)

    result = out["vocals"]
    assert result.model_id == "4stem-c/htdemucs-ft"
    assert result.notes, "a VRAM degradation reason string must be present"
    joined = " ".join(result.notes).lower()
    assert "vram" in joined or "fell back" in joined or "fallback" in joined


# --------------------------------------------------------------------------- #
# 5. GPU semaphore — two concurrent separate_region calls never overlap in the engine
# --------------------------------------------------------------------------- #


def test_gpu_semaphore_serializes_engine_calls():
    concurrency = {"now": 0, "max": 0}
    lock = threading.Lock()

    def barrier() -> None:
        with lock:
            concurrency["now"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["now"])
        time.sleep(0.05)  # widen the window an overlapping call would slip through
        with lock:
            concurrency["now"] -= 1

    pcm = _ramp_pcm(4.0)
    entry = _entry("4stem-c/htdemucs-ft")

    def worker() -> None:
        engine = StubEngine(FOUR_STEM, barrier_hook=barrier)
        separate_region(pcm, entry, Window(start_s=0.0, end_s=2.0), engine=engine)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert concurrency["max"] == 1, (
        f"engine calls overlapped (max concurrency {concurrency['max']})"
    )


# --------------------------------------------------------------------------- #
# 6. native_sr resample — entry.native_sr=44100 + 22050 kHz input: stub receives 44100 Hz
#    audio, results return at the original 22050 Hz with a consistent sample count (Task 4.1
#    follow-up: SCNet-XL fed off-rate audio hit torch's slow ComplexHalf istft path).
# --------------------------------------------------------------------------- #


def test_native_sr_resamples_engine_input_and_restores_output_rate():
    entry = _entry("4stem-b/scnet-xl")
    assert entry.native_sr == 44100  # verified from the vendored SCNet config (§ this test)

    sr_in = 22050
    pcm = _ramp_pcm(4.0, sr=sr_in)
    engine = StubEngine(FOUR_STEM)
    out = separate_region(pcm, entry, Window(start_s=1.0, end_s=3.0), engine=engine)

    # The stub engine received audio resampled UP to native_sr, not the original 22050 Hz.
    assert engine.last_call["sr"] == 44100

    assert set(out) == set(FOUR_STEM)
    result = out["vocals"]
    assert result.sample_rate == sr_in  # output rides back down to the input rate
    expected_len = int(round(2.0 * sr_in))  # region [1,3] = 2 s at the input rate
    stem = _read(result.stem_path)
    assert stem.shape[0] == expected_len
    joined = " ".join(result.notes).lower()
    assert "native_sr" in joined or "44100" in joined


def test_native_sr_is_a_noop_when_input_already_matches():
    entry = _entry("4stem-b/scnet-xl")
    pcm = _ramp_pcm(4.0, sr=44100)
    engine = StubEngine(FOUR_STEM)
    out = separate_region(pcm, entry, Window(start_s=1.0, end_s=3.0), engine=engine)

    assert engine.last_call["sr"] == 44100
    result = out["vocals"]
    assert result.sample_rate == 44100
    assert not any("native_sr" in note.lower() for note in result.notes)


# --------------------------------------------------------------------------- #
# real-engine end-to-end smoke (desktop + GPU): audio-separator AND vendored SCNet
# --------------------------------------------------------------------------- #

_CI_SPARSE = Path(__file__).resolve().parent / "fixtures" / "audio" / "ci_sparse" / "mix.flac"


@pytest.mark.desktop
@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_id",
    ["4stem-c/htdemucs-ft", "4stem-b/scnet-xl"],
    ids=["audio-separator/htdemucs-ft", "msst/scnet-xl"],
)
def test_real_engines_produce_four_aligned_stems(model_id):
    """§12 slice-4 engine half: audio-separator (htdemucs-ft — the 4-stem model whose weights
    audio-separator's registry actually knows and our pins are hash-equal to) and the vendored
    SCNet path each produce 4 length-aligned, non-silent stems from ci_sparse."""
    from orpheus_mcp.analysis.cachefs import decode

    torch = pytest.importorskip("torch")
    if model_id == "4stem-b/scnet-xl" and not torch.cuda.is_available():
        pytest.skip("no CUDA for the in-process SCNet path")
    if not _CI_SPARSE.is_file():
        pytest.skip("ci_sparse fixture missing")

    pcm = decode(str(_CI_SPARSE))
    entry = _entry(model_id)
    out = separate_region(pcm, entry, None, engine=None)

    assert set(FOUR_STEM) <= set(out)
    n = pcm.samples.shape[0]
    for tgt in FOUR_STEM:
        stem = _read(out[tgt].stem_path)
        assert stem.shape[0] == n, f"{tgt} not aligned: {stem.shape[0]} != {n}"
        assert float(np.sqrt(np.mean(stem**2))) > 0.0, f"{tgt} is digital silence"
