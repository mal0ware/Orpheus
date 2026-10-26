"""``analysis/recon.py`` + ``recon_reference`` tool (Task 1.3, §T1): tempo/beat grid,
tuning, key, structure sections, band sketch — verbatim from the task 1.3 brief's test list.

Constructed-signal tests generate audio in-test with numpy+soundfile (click track,
detuned chord progression, noise burst); the two remaining brief tests use the committed
CI fixtures (``ci_sparse``) and the desktop fixture set (``~/.orpheus_fixtures``,
deselected in CI via ``-m "not desktop and not gpu"``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orpheus_mcp.analysis import cachefs
from orpheus_mcp.analysis.cachefs import CacheFS, decode
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.recon import run_recon
from orpheus_mcp.models import Recon
from orpheus_mcp.tools.reference import recon_reference

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "audio"
DESKTOP_FIXTURES_ROOT = Path.home() / ".orpheus_fixtures"
DESKTOP_FIXTURES = ("pop_sparse", "jazz_dense", "house_medium")


# --------------------------------------------------------------------------- #
# Synthetic-fixture generators
# --------------------------------------------------------------------------- #


def make_click_track(
    path: Path, *, bpm: float = 120.0, seconds: float = 16.0, sr: int = 22050
) -> Path:
    """A hard click track: short decaying tone bursts on every beat — an unambiguous
    onset pattern for the beat tracker."""
    interval = 60.0 / bpm
    n_samples = int(seconds * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    click_len = int(0.02 * sr)
    envelope = np.exp(-np.linspace(0.0, 15.0, click_len)).astype(np.float32)
    tone = (np.sin(2 * np.pi * 1000.0 * np.arange(click_len) / sr) * envelope).astype(np.float32)

    t = 0.0
    while t < seconds:
        start = int(t * sr)
        end = min(start + click_len, n_samples)
        audio[start:end] += tone[: end - start]
        t += interval

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def make_accented_click_track(
    path: Path,
    *,
    bpm: float = 120.0,
    seconds: float = 16.0,
    sr: int = 22050,
    accent_every: int = 4,
    offbeat_gain: float = 0.04,
) -> Path:
    """A click track where every ``accent_every``-th beat is louder than the rest — unlike
    ``make_click_track``'s uniform clicks, this gives the {3,4}-group correlation an
    unambiguous winner (high ``meter_confidence``, D10)."""
    interval = 60.0 / bpm
    n_samples = int(seconds * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    click_len = int(0.02 * sr)
    envelope = np.exp(-np.linspace(0.0, 15.0, click_len)).astype(np.float32)
    tone = (np.sin(2 * np.pi * 1000.0 * np.arange(click_len) / sr) * envelope).astype(np.float32)

    t = 0.0
    beat_idx = 0
    while t < seconds:
        start = int(t * sr)
        end = min(start + click_len, n_samples)
        gain = 1.0 if beat_idx % accent_every == 0 else offbeat_gain
        audio[start:end] += gain * tone[: end - start]
        t += interval
        beat_idx += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def make_chord_progression(
    path: Path,
    chords: list[list[float]],
    *,
    seconds_per_chord: float = 2.0,
    sr: int = 22050,
    detune_cents: float = 0.0,
) -> Path:
    """A progression of held triads (sum of sine partials), optionally detuned by a fixed
    cents offset applied uniformly to every partial — simulates a sharp/flat tape or a
    432 Hz-style tuning."""
    factor = 2 ** (detune_cents / 1200.0)
    segments = []
    for freqs in chords:
        n = int(seconds_per_chord * sr)
        t = np.arange(n) / sr
        segment = np.zeros(n, dtype=np.float32)
        for freq in freqs:
            segment += (0.2 * np.sin(2 * np.pi * freq * factor * t)).astype(np.float32)
        segments.append(segment)
    audio = np.concatenate(segments)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def make_noise_burst(path: Path, *, seconds: float = 10.0, sr: int = 22050, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    audio = (0.3 * rng.standard_normal(int(seconds * sr))).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


# A C-F-G-C progression (all diatonic major triads, root/third/fifth) — the whole passage
# activates exactly the 7 C-major scale pitch classes, weighted toward C and G (each
# appears in two chords).
_C_MAJOR_PROGRESSION = [
    [261.63, 329.63, 392.00],  # C major
    [349.23, 440.00, 523.25],  # F major
    [392.00, 493.88, 587.33],  # G major
    [261.63, 329.63, 392.00],  # C major
]


# --------------------------------------------------------------------------- #
# Load-bearing tests — verbatim from the Task 1.3 brief
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
def test_click_track_tempo_and_grid_confidence(tmp_path):
    path = make_click_track(tmp_path / "click.wav", bpm=120.0)
    pcm = decode(str(path))
    recon = run_recon(pcm)

    assert 118.0 <= recon.tempo_bpm <= 122.0
    assert recon.grid.confidence > 0.8


@pytest.mark.analysis
def test_accented_click_track_meter_confidence_high(tmp_path):
    """D10: an unambiguous 4/4 accent pattern (every 4th beat louder) must produce a high
    margin ratio between the winning group-{3,4} correlation and the runner-up."""
    path = make_accented_click_track(tmp_path / "accented.wav", bpm=120.0, accent_every=4)
    pcm = decode(str(path))
    recon = run_recon(pcm)

    assert recon.grid.meter == (4, 4)
    assert recon.grid.meter_confidence > 0.3


@pytest.mark.analysis
def test_uniform_click_meter_confidence_low_while_grid_confidence_stays_high(tmp_path):
    """D10: an unaccented, perfectly uniform click train has a near-tied 3-vs-4 grouping
    (every beat equal strength) — ``meter_confidence`` must read low even though the grid
    itself (``confidence``, a different signal) is rock-solid."""
    path = make_click_track(tmp_path / "uniform.wav", bpm=120.0, seconds=16.0)
    pcm = decode(str(path))
    recon = run_recon(pcm)

    assert recon.grid.meter_confidence < 0.15
    assert recon.grid.confidence > 0.8


@pytest.mark.analysis
def test_ci_sparse_key_matches_truth():
    truth = json.loads((FIXTURES / "ci_sparse" / "truth.json").read_text("utf-8"))
    pcm = decode(str(FIXTURES / "ci_sparse" / "mix.flac"))
    recon = run_recon(pcm)

    assert recon.key.root == truth["key"]["root"]
    assert recon.key.mode == truth["key"]["mode"]


@pytest.mark.analysis
def test_detuned_progression_tuning_offset_and_key(tmp_path):
    path = make_chord_progression(
        tmp_path / "detuned.wav", _C_MAJOR_PROGRESSION, detune_cents=40.0
    )
    pcm = decode(str(path))
    recon = run_recon(pcm)

    assert 30.0 <= recon.tuning_offset_cents <= 50.0
    assert recon.key.root == "C"
    assert recon.key.mode == "major"


@pytest.mark.analysis
def test_noise_burst_under_30s_is_single_section(tmp_path):
    path = make_noise_burst(tmp_path / "noise.wav", seconds=10.0)
    pcm = decode(str(path))
    recon = run_recon(pcm)

    assert len(recon.sections) == 1
    assert recon.sections[0].start_s == pytest.approx(0.0)
    assert recon.sections[0].end_s == pytest.approx(recon.duration_s)


@pytest.mark.analysis
def test_recon_reference_second_call_served_from_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path / "cache"))
    audio_path = make_click_track(tmp_path / "src" / "click.wav", seconds=4.0)

    calls = {"n": 0}
    real_decode = cachefs.decode

    def counted(path: str):
        calls["n"] += 1
        return real_decode(path)

    monkeypatch.setattr(cachefs, "decode", counted)

    first = recon_reference(str(audio_path))
    count_after_first = calls["n"]
    assert count_after_first >= 1  # the miss path must have decoded at least once

    second = recon_reference(str(audio_path))
    assert calls["n"] == count_after_first  # cache HIT — no additional decode

    assert first == second


@pytest.mark.analysis
def test_reference_category_in_default_and_full_not_explain():
    from fastmcp import FastMCP

    from orpheus_mcp.registry import register_tools

    m_default = FastMCP(name="t")
    assert "reference" in register_tools(m_default, profile="default")
    m_full = FastMCP(name="t")
    assert "reference" in register_tools(m_full, profile="full")
    m_explain = FastMCP(name="t")
    assert "reference" not in register_tools(m_explain, profile="explain")


@pytest.mark.desktop
@pytest.mark.parametrize("fixture", DESKTOP_FIXTURES)
def test_desktop_fixture_recon_has_every_field(fixture):
    """§12 slice-1 acceptance bar, verbatim: recon fields present incl. tuning/downbeats/
    grid confidence on 3 (desktop) fixture files. Deselected in CI; needs the real
    ~/.orpheus_fixtures set on the desktop box."""
    mix = DESKTOP_FIXTURES_ROOT / fixture / "mix.flac"
    if not mix.is_file():
        pytest.skip(f"desktop fixture not present: {mix}")

    pcm = decode(str(mix))
    recon = run_recon(pcm)

    assert isinstance(recon, Recon)
    assert recon.duration_s > 0.0
    assert recon.sample_rate > 0
    assert recon.tempo_bpm > 0.0
    assert recon.grid.times_s, "no beats detected"
    assert recon.grid.downbeats_s, "no downbeats detected"
    assert recon.grid.meter in {(3, 4), (4, 4)}
    assert 0.0 <= recon.grid.confidence <= 1.0
    assert recon.grid.kind == "beat"
    assert isinstance(recon.tuning_offset_cents, float)
    assert recon.key.root
    assert recon.key.mode
    assert 0.0 <= recon.key.confidence <= 1.0
    assert len(recon.key_alternates) == 2
    assert recon.sections
    assert len(recon.band_energy_db) == 8


# --------------------------------------------------------------------------- #
# Supplementary coverage (not verbatim in the brief, but exercising the same locked
# decisions from other angles).
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
def test_band_energy_sketch_names_are_locked():
    """Weak-lock fix (review finding): pin the EXACT full 8-key set, not just one member —
    these are ``geomspace(60, 16000, 9)`` edges, `f"{int(lo)}-{int(hi)}"` per adjacent pair."""
    pcm = decode(str(FIXTURES / "ci_sparse" / "mix.flac"))
    recon = run_recon(pcm)

    assert set(recon.band_energy_db.keys()) == {
        "60-120",
        "120-242",
        "242-487",
        "487-979",
        "979-1969",
        "1969-3959",
        "3959-7959",
        "7959-16000",
    }
    assert all(isinstance(v, float) for v in recon.band_energy_db.values())


@pytest.mark.analysis
def test_all_chroma_downstream_is_tuning_corrected(tmp_path, monkeypatch):
    """The §T1 lock: every chroma call downstream of tuning estimation must pass the
    measured ``tuning=`` value. Patches ``librosa.feature.chroma_cqt`` and asserts every
    call it recorded a non-None ``tuning`` kwarg."""
    import librosa

    path = make_chord_progression(tmp_path / "prog.wav", _C_MAJOR_PROGRESSION, detune_cents=25.0)
    pcm = decode(str(path))

    seen_tunings: list[float | None] = []
    real_chroma_cqt = librosa.feature.chroma_cqt

    def spy(*args, **kwargs):
        seen_tunings.append(kwargs.get("tuning"))
        return real_chroma_cqt(*args, **kwargs)

    monkeypatch.setattr(librosa.feature, "chroma_cqt", spy)
    run_recon(pcm)

    assert seen_tunings, "chroma_cqt was never called"
    assert all(t is not None for t in seen_tunings)


@pytest.mark.analysis
async def test_recon_reference_tool_registered_read_only():
    from fastmcp import Client, FastMCP

    from orpheus_mcp.registry import register_tools

    mcp = FastMCP(name="t")
    register_tools(mcp, profile="default")
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "recon_reference" in tools
    assert tools["recon_reference"].annotations.readOnlyHint is True


def test_pcm_hash_stability_same_audio_round_trips(tmp_path):
    """§9 PCM-hash stability: recon caches under the SAME pcm-hash for a byte-identical
    re-decode of the same file."""
    path = make_click_track(tmp_path / "click.wav", seconds=4.0)
    pcm_a = decode(str(path))
    pcm_b = decode(str(path))
    assert pcm_a.pcm_hash == pcm_b.pcm_hash


@pytest.mark.analysis
def test_recon_reference_uses_provided_cache(tmp_path):
    fs = CacheFS(tmp_path / "explicit_cache")
    path = make_click_track(tmp_path / "click.wav", seconds=4.0)

    recon_reference(str(path), cache=fs)

    file_hash = cachefs.pcm_hash(str(path))
    cached = fs.read_json(
        file_hash,
        "recon",
        meta={
            "schema_version": 1,
            "registry_version": cachefs.registry_version(),
            "verify_config_hash": cachefs.verify_config_hash(),
            "decoder": cachefs.decoder_for(str(path)),
        },
    )
    assert cached is not None
