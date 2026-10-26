"""``tools/reference.py``'s ``get_chords`` (Task 3.2/9.1, §13/§9): the read-through ``"chords"``
cache over the full chord lane (``harmony.compose_lane`` — recognizer + §13.3 extensions +
[§13 T3 model re-write] + §13.2 slash + §13.4 theory), plus the post-cache region filter (§3.1d
clamping). The §13 model lane and its bass-STEM slash preference are covered in
``test_harmony_models.py``; these tests force the CPU-only baseline path where the lane identity
matters.

Contract tests run through an in-memory FastMCP ``Client`` (``tests/test_m2_tools.py``'s
pattern); the compute-count / cache-sharing assertions call ``get_chords``/``recon_reference``
directly against an explicit ``CacheFS`` (``tests/test_recon.py``'s pattern) since a monkeypatch
on a module-level helper is easiest to see from the same process, not through the wire.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from orpheus_mcp.analysis import cachefs, model_registry
from orpheus_mcp.analysis.cachefs import CacheFS, decode
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.recon import run_recon
from orpheus_mcp.analysis.separation import SeparationError
from orpheus_mcp.models import LocateResult, LyricWord, StemResult, Window
from orpheus_mcp.registry import register_tools
from orpheus_mcp.tools import reference as reference_tools
from orpheus_mcp.tools.reference import (
    ReferenceEnvMissing,
    get_chords,
    locate_lyric,
    recon_reference,
    separate_region,
    transcribe_lyrics,
)
from stub_engines import StubAsr, StubEngine, StubTagger

# A short re-struck two-chord progression (same recipe as test_harmony_baseline.py's
# ``_make_progression``: a decaying sine-triad strike per beat, so the beat tracker finds a
# real grid) — kept to 2 chords x 4 beats so these contract tests stay fast; the exact chord
# labels recognized are NOT asserted here (that bar lives in test_harmony_baseline.py), only
# cache/region-filter behavior over whatever lane comes back.
_TRIADS: dict[str, list[float]] = {
    "C": [261.63, 329.63, 392.00],
    "G": [196.00, 246.94, 293.66],
}


def _make_progression(path: Path, *, bpm: float = 120.0, sr: int = 22050) -> Path:
    beat_len = 60.0 / bpm
    n = int(beat_len * sr)
    t = np.arange(n) / sr
    env = np.exp(-3.0 * t).astype(np.float32)
    out: list[np.ndarray] = []
    for sym in ["C", "G"]:
        for _ in range(4):
            strike = np.zeros(n, dtype=np.float32)
            for freq in _TRIADS[sym]:
                strike += (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            out.append(strike * env)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.concatenate(out), sr, subtype="PCM_16")
    return path


@pytest.fixture
def audio_path(tmp_path) -> Path:
    return _make_progression(tmp_path / "prog.wav")


@pytest.fixture
def fs(tmp_path) -> CacheFS:
    return CacheFS(tmp_path / "cache")


# --------------------------------------------------------------------------- #
# Cache HIT never recomputes the lane
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
def test_full_lane_fetch_computes_once_then_cache_hits(audio_path, fs, monkeypatch):
    from orpheus_mcp.analysis import harmony

    calls = {"n": 0}
    real = harmony.compose_lane

    def counted(pcm, recon, **kwargs):
        calls["n"] += 1
        return real(pcm, recon, **kwargs)

    # get_chords imports compose_lane from harmony at call time, so patch it at the source.
    monkeypatch.setattr(harmony, "compose_lane", counted)

    first = get_chords(str(audio_path), cache=fs)
    assert calls["n"] == 1
    assert first.segments  # a sane lane, not empty

    second = get_chords(str(audio_path), cache=fs)
    assert calls["n"] == 1  # cache HIT — the lane was never recomputed
    assert second == first


@pytest.mark.analysis
def test_cache_hit_never_decodes(audio_path, fs, monkeypatch):
    get_chords(str(audio_path), cache=fs)  # populate the cache

    calls = {"n": 0}
    real_decode = cachefs.decode

    def counted(path: str):
        calls["n"] += 1
        return real_decode(path)

    monkeypatch.setattr(cachefs, "decode", counted)

    get_chords(str(audio_path), cache=fs)
    assert calls["n"] == 0


# --------------------------------------------------------------------------- #
# Region filtering (post-cache, §3.1d clamping)
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
def test_region_filter_returns_only_overlapping_whole_segments(audio_path, fs):
    full = get_chords(str(audio_path), cache=fs)
    assert len(full.segments) >= 1

    mid = full.segments[len(full.segments) // 2]
    region = Window(start_s=mid.start_s, end_s=mid.end_s)
    filtered = get_chords(str(audio_path), region, cache=fs)

    expected = [
        seg for seg in full.segments if seg.start_s < region.end_s and seg.end_s > region.start_s
    ]
    assert filtered.segments == expected
    assert filtered.grid == full.grid
    # segments come back WHOLE — never re-cut to the region's exact edges
    for seg in filtered.segments:
        assert seg in full.segments


@pytest.mark.analysis
def test_region_end_10x_duration_clamps_and_keeps_full_tail(audio_path, fs):
    full = get_chords(str(audio_path), cache=fs)
    recon = run_recon(decode(str(audio_path)))
    duration = recon.duration_s

    region = Window(start_s=0.0, end_s=duration * 10.0)
    filtered = get_chords(str(audio_path), region, cache=fs)

    # clamps to [0, duration] cleanly (no error) and keeps every segment, incl. the tail
    assert filtered.segments == full.segments
    assert filtered.segments[-1].end_s == pytest.approx(full.segments[-1].end_s)


@pytest.mark.analysis
def test_region_fully_outside_duration_returns_empty_segments_grid_preserved(audio_path, fs):
    full = get_chords(str(audio_path), cache=fs)
    recon = run_recon(decode(str(audio_path)))
    duration = recon.duration_s

    region = Window(start_s=duration + 50.0, end_s=duration + 100.0)
    filtered = get_chords(str(audio_path), region, cache=fs)

    assert filtered.segments == []
    assert filtered.grid == full.grid  # grid preserved, not defaulted/dropped


# --------------------------------------------------------------------------- #
# Shared recon read-through
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
def test_get_chords_reuses_recon_already_cached_by_recon_reference(audio_path, fs, monkeypatch):
    calls = {"n": 0}
    import orpheus_mcp.analysis.recon as recon_mod

    real_run_recon = recon_mod.run_recon

    def counted(pcm):
        calls["n"] += 1
        return real_run_recon(pcm)

    monkeypatch.setattr(recon_mod, "run_recon", counted)

    recon_reference(str(audio_path), cache=fs)
    assert calls["n"] == 1

    lane = get_chords(str(audio_path), cache=fs)
    assert calls["n"] == 1  # recon was reused from cache, not recomputed
    assert lane.segments


@pytest.mark.analysis
def test_chords_meta_sidecar_records_produced_by_baseline(audio_path, fs, monkeypatch):
    # Force the CPU-only path (no chord model resolves) so the baseline lane is deterministic
    # regardless of the box: on a CUDA box with the BTC checkpoint fetched, get_chords would
    # otherwise engage the §13 model lane and key the sidecar under the model entry id.
    monkeypatch.setattr(model_registry, "free_vram_gb", lambda: None)
    get_chords(str(audio_path), cache=fs)

    file_hash = cachefs.pcm_hash(str(audio_path))
    meta = {
        "schema_version": 1,
        "registry_version": cachefs.registry_version(),
        "verify_config_hash": cachefs.verify_config_hash(),
        "decoder": cachefs.decoder_for(str(audio_path)),
        "produced_by": "baseline",
    }
    cached = fs.read_json(file_hash, "chords", meta=meta)
    assert cached is not None
    assert all(seg["produced_by"] == "baseline" for seg in cached["segments"])


# --------------------------------------------------------------------------- #
# MCP contract (in-memory Client, tests/test_m2_tools.py's pattern)
# --------------------------------------------------------------------------- #


@pytest.fixture
def mcp_client(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path / "orpheus_cache"))
    mcp = FastMCP(name="OrpheusReferenceTest")
    register_tools(mcp, profile="default")
    return Client(mcp)


@pytest.mark.analysis
async def test_get_chords_tool_registered_read_only(mcp_client):
    async with mcp_client as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert "get_chords" in tools
    assert tools["get_chords"].annotations.readOnlyHint is True


@pytest.mark.analysis
async def test_get_chords_tool_full_lane_via_client(audio_path, mcp_client):
    async with mcp_client as c:
        result = await c.call_tool("get_chords", {"file": str(audio_path)})
    lane = result.data
    assert lane.segments
    assert lane.grid in {"beat", "fixed"}


@pytest.mark.analysis
async def test_get_chords_tool_region_via_client(audio_path, mcp_client):
    async with mcp_client as c:
        full = (await c.call_tool("get_chords", {"file": str(audio_path)})).data
        mid = full.segments[len(full.segments) // 2]
        filtered = (
            await c.call_tool(
                "get_chords",
                {
                    "file": str(audio_path),
                    "region": {"start_s": mid.start_s, "end_s": mid.end_s},
                },
            )
        ).data
    assert all(
        seg.start_s < mid.end_s and seg.end_s > mid.start_s for seg in filtered.segments
    )
    assert filtered.segments  # the midpoint segment itself always overlaps its own span


# --------------------------------------------------------------------------- #
# separate_region (Task 4.2, §5/§9): resolve -> Task 4.1 core -> gates 1-3 -> §9 stem
# cache. Driven end-to-end by StubEngine + StubTagger (tests/stub_engines.py) -- no real
# checkpoints, no GPU. The desktop/gpu real-engine smoke lives in test_separation.py's
# sibling module convention; this file only proves the TOOL wrapper's plumbing.
# --------------------------------------------------------------------------- #

FOUR_STEM = ["vocals", "drums", "bass", "other"]
_HTDEMUCS_FT = "4stem-c/htdemucs-ft"  # vram_gb == 0.0 -- never triggers the VRAM fallback


def _write_sep_wav(path: Path, *, seconds: float = 4.0, sr: int = 4000) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tone, sr, subtype="PCM_16")
    return path


@pytest.fixture
def _plenty_of_vram(monkeypatch):
    """Same hermeticity fixture as test_separation.py -- pin free_vram_gb so the VRAM
    guard's ``model_registry.free_vram_gb()`` call never imports real torch."""
    monkeypatch.setattr(model_registry, "free_vram_gb", lambda: 24.0)


def test_separate_region_returns_stem_result_with_gates(tmp_path, _plenty_of_vram):
    audio = _write_sep_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(FOUR_STEM)
    tagger = StubTagger()

    result = separate_region(
        str(audio), "drums", model_id=_HTDEMUCS_FT, cache=fs, engine=engine, tagger=tagger
    )

    assert isinstance(result, StemResult)
    assert result.target == "drums"
    assert result.model_id == _HTDEMUCS_FT
    assert Path(result.stem_path).is_file()
    assert result.gates is not None
    assert result.gates.overall in {"pass", "fail", "unverifiable"}
    for gate in (
        result.gates.mix_consistency,
        result.gates.bleed,
        result.gates.silence_honesty,
        result.gates.agreement,
    ):
        assert gate.verdict in {"pass", "fail", "na"}
        assert (gate.value is None) == (gate.verdict == "na")
    # Gate 4 is a Phase-5 stub -- always N/A here, never fabricated.
    assert result.gates.agreement.verdict == "na"
    assert result.gates.agreement.value is None


def test_separate_region_caches_stem_and_skips_recompute_on_hit(tmp_path, _plenty_of_vram):
    audio = _write_sep_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(FOUR_STEM)
    tagger = StubTagger()

    first = separate_region(
        str(audio), "bass", model_id=_HTDEMUCS_FT, cache=fs, engine=engine, tagger=tagger
    )
    assert len(engine.calls) == 1

    second = separate_region(
        str(audio), "bass", model_id=_HTDEMUCS_FT, cache=fs, engine=engine, tagger=tagger
    )
    assert len(engine.calls) == 1  # cache HIT -- the engine was never re-invoked
    assert second.stem_path == first.stem_path
    assert second.gates == first.gates


def test_separate_region_caches_under_the_9_stems_layout(tmp_path, _plenty_of_vram):
    audio = _write_sep_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(FOUR_STEM)
    tagger = StubTagger()

    result = separate_region(
        str(audio), "vocals", model_id=_HTDEMUCS_FT, cache=fs, engine=engine, tagger=tagger
    )

    stem_path = Path(result.stem_path)
    assert stem_path.name == "vocals.flac"
    assert stem_path.parent.parent.name == _HTDEMUCS_FT.split("/")[-1]
    assert (stem_path.parent / "verify.json").is_file()
    assert (stem_path.parent / "meta.json").is_file()


def test_separate_region_unknown_target_raises_separation_error(tmp_path, _plenty_of_vram):
    audio = _write_sep_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")

    with pytest.raises(SeparationError):
        separate_region(str(audio), "definitely_not_a_registry_target", cache=fs)


def test_separate_region_explicit_model_id_bypasses_ranking(tmp_path, _plenty_of_vram):
    audio = _write_sep_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(FOUR_STEM)
    tagger = StubTagger()

    result = separate_region(
        str(audio), "other", model_id=_HTDEMUCS_FT, cache=fs, engine=engine, tagger=tagger
    )
    assert result.model_id == _HTDEMUCS_FT


# --------------------------------------------------------------------------- #
# separate_region: MCP registration (gated exactly like instrument_timeline -- §4.2
# brief: torch + panns_inference importable suffices; the separation-stack errors are
# typed and surface at CALL time, not registration time).
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
async def test_separate_region_tool_registered_read_only():
    mcp = FastMCP(name="t")
    register_tools(mcp, profile="default")
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "separate_region" in tools
    assert tools["separate_region"].annotations.readOnlyHint is True


@pytest.mark.analysis
async def test_separate_region_not_registered_without_panns_or_torch(monkeypatch):
    import importlib.util as ilu

    real_find_spec = ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("panns_inference", "torch"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(ilu, "find_spec", fake_find_spec)

    mcp = FastMCP(name="t")
    with pytest.warns(UserWarning, match="instrument_timeline not registered"):
        reference_tools.register(mcp)

    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "separate_region" not in tools
    assert "recon_reference" in tools  # category-level tool unaffected


# --------------------------------------------------------------------------- #
# Desktop/gpu real-engine end-to-end: separate_region tool on ci_sparse, target
# "drums", SCNet rank-1 (§12 slice-4 bar's separate_region half). Real PannsTagger runs
# on both the mix (instrument_timeline) and the freshly separated stem (the bleed
# judge) -- assert MECHANISM (stems cached, gates populated, verdicts in the enum), not
# specific verdicts: a failing gate on real separated audio is DATA, not a test failure.
# --------------------------------------------------------------------------- #

_CI_SPARSE = Path(__file__).resolve().parent / "fixtures" / "audio" / "ci_sparse" / "mix.flac"


@pytest.mark.desktop
@pytest.mark.gpu
def test_separate_region_end_to_end_drums_scnet_rank1(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA for the in-process SCNet path")
    if not _CI_SPARSE.is_file():
        pytest.skip("ci_sparse fixture missing")

    # 4stem-b/scnet-xl is registry rank 1 for target "drums" (Task 4.1's D12 re-rank).
    entry = {e.id: e for e in model_registry.load_registry(allow_unpinned=True)}["4stem-b/scnet-xl"]
    assert entry.rank == 1
    assert "drums" in entry.targets

    fs = CacheFS(tmp_path / "cache")
    result = separate_region(str(_CI_SPARSE), "drums", model_id="4stem-b/scnet-xl", cache=fs)

    assert isinstance(result, StemResult)
    assert result.target == "drums"
    assert result.model_id == "4stem-b/scnet-xl"
    assert Path(result.stem_path).is_file()
    assert (Path(result.stem_path).parent / "verify.json").is_file()
    assert (Path(result.stem_path).parent / "meta.json").is_file()

    assert result.gates is not None
    for gate in (
        result.gates.mix_consistency,
        result.gates.bleed,
        result.gates.silence_honesty,
        result.gates.agreement,
    ):
        assert gate.verdict in {"pass", "fail", "na"}
        assert (gate.value is None) == (gate.verdict == "na")
    assert result.gates.overall in {"pass", "fail", "unverifiable"}
    assert result.gates.agreement.verdict == "na"  # Phase-5 stub, never fabricated

    print(
        "\nDESKTOP separate_region(ci_sparse, drums, 4stem-b/scnet-xl) gate values:\n"
        f"  mix_consistency: {result.gates.mix_consistency}\n"
        f"  bleed:           {result.gates.bleed}\n"
        f"  silence_honesty: {result.gates.silence_honesty}\n"
        f"  agreement:       {result.gates.agreement}\n"
        f"  overall:         {result.gates.overall}\n"
    )


# --------------------------------------------------------------------------- #
# verify_stem (Task 5.3, §9): the lightweight librosa-tier re-verifier. Gates 1/2/4
# are N/A by construction (single stem, no tagger, no second take); gate 3 (silence
# honesty) runs only when a mix timeline is supplied. Numeric cases are hand-built.
# --------------------------------------------------------------------------- #

from orpheus_mcp.models import GateSet, Timeline  # noqa: E402
from orpheus_mcp.tools.reference import verify_stem  # noqa: E402


def _write_stem(path: Path, loud_windows: list[bool], *, sr: int = 8000) -> Path:
    """One 1 s frame per bool: a 220 Hz tone where True, digital silence where False."""
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    frames = [tone if loud else np.zeros(sr, dtype=np.float32) for loud in loud_windows]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.concatenate(frames), sr, subtype="PCM_16")
    return path


def _timeline(active_windows: list[bool]) -> Timeline:
    hot = CFG.TIMELINE_ACTIVITY_THRESHOLD + 0.5
    return Timeline(
        classes=["Drum kit"], hop_s=1.0,
        frames=[[hot if a else 0.0] for a in active_windows],
    )


def test_verify_stem_no_timeline_is_all_na_unverifiable(tmp_path):
    stem = _write_stem(tmp_path / "stem.flac", [True, True, False])
    gates = verify_stem(str(stem), str(stem), None, None)
    assert isinstance(gates, GateSet)
    for g in (gates.mix_consistency, gates.bleed, gates.silence_honesty, gates.agreement):
        assert g.verdict == "na"
    assert gates.overall == "unverifiable"


def test_verify_stem_silent_stem_passes_silence_honesty(tmp_path):
    # Active for 3 s, then silent for 8 s. Timeline marks the same. The stem is genuinely
    # silent in the (post-tail) inactive windows -> silence honesty passes; others stay N/A.
    windows = [True, True, True] + [False] * 8
    stem = _write_stem(tmp_path / "stem.flac", windows)
    gates = verify_stem(str(stem), str(stem), None, _timeline(windows))
    assert gates.silence_honesty.verdict == "pass"
    assert gates.mix_consistency.verdict == "na"
    assert gates.bleed.verdict == "na"
    assert gates.agreement.verdict == "na"
    assert gates.overall == "pass"


def test_verify_stem_energy_in_silent_window_fails(tmp_path):
    # Timeline says inactive after 3 s, but the stem keeps blaring -> false energy -> fail.
    active = [True, True, True] + [False] * 8
    loud = [True] * 11
    stem = _write_stem(tmp_path / "stem.flac", loud)
    gates = verify_stem(str(stem), str(stem), None, _timeline(active))
    assert gates.silence_honesty.verdict == "fail"
    assert gates.overall == "fail"


@pytest.mark.analysis
async def test_verify_stem_tool_registered_read_only():
    mcp = FastMCP(name="t")
    register_tools(mcp, profile="default")
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "verify_stem" in tools
    assert tools["verify_stem"].annotations.readOnlyHint is True


# --------------------------------------------------------------------------- #
# transcribe_lyrics + locate_lyric (Task 8.2, §14.2): the vocal-stem-via-ladder ASR index
# and the tool-layer wrapper over the pure fuzzy resolver. Driven end-to-end by StubEngine +
# StubTagger + StubAsr -- no real separator/tagger/ASR weights, no GPU (the same convention
# separate_region's tests above use). The desktop/gpu-optional real end-to-end smoke sits at
# the bottom of this file.
# --------------------------------------------------------------------------- #

_VOCAL_CLASSES = [
    "Singing", "Speech", "Drum kit", "Snare drum", "Bass guitar", "Guitar",
    "Electric guitar", "Piano",
]


def _write_lyrics_test_wav(path: Path, *, seconds: float = 4.0, sr: int = 22050) -> Path:
    """A higher-sample-rate tone than ``_write_sep_wav`` (4 kHz, too low for recon's CQT-based
    key/chroma passes) -- ``transcribe_lyrics`` runs the FULL §3.1 recon, unlike
    ``separate_region``'s tests above, which never touch it."""
    t = np.arange(int(seconds * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tone, sr, subtype="PCM_16")
    return path


def _mixed_vocal_tagger() -> StubTagger:
    """Vocals present (Singing=0.5) but not exposed (Guitar bleeds equally) -- the T2
    pre-flight falls through to rank-1, matching test_ladder.py's ``_mixed_timeline``."""
    row = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    return StubTagger(_VOCAL_CLASSES, hop_s=1.0, activations=[row] * 5)


def _absent_vocal_tagger() -> StubTagger:
    """Vocal classes never exceed CFG.TIMELINE_ACTIVITY_THRESHOLD anywhere -- T2 confirms
    the target absent, matching test_ladder.py's ``_absent_timeline``."""
    row = [0.1, 0.1, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0]
    return StubTagger(_VOCAL_CLASSES, hop_s=1.0, activations=[row] * 5)


def _lw(text: str, start: float, end: float) -> LyricWord:
    return LyricWord(word=text, start_s=start, end_s=end, conf=0.9)


_SCRIPTED_WORDS = [_lw("hold", 0.5, 0.8), _lw("the", 0.9, 1.0), _lw("line.", 1.1, 1.4)]


@pytest.mark.analysis
def test_transcribe_lyrics_confirmed_absent_skips_separation_and_asr(tmp_path, _plenty_of_vram):
    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(["vocals"])
    asr = StubAsr(language="zz", words=list(_SCRIPTED_WORDS))

    result = transcribe_lyrics(
        str(audio), cache=fs, asr=asr, tagger=_absent_vocal_tagger(), engine=engine
    )

    assert result.status == "no_vocals"
    assert result.words == []
    assert asr.calls == 0  # T2 pre-flight confirmed absent -> ASR never ran
    assert len(engine.calls) == 0  # -> and no separation ever ran either


@pytest.mark.analysis
def test_transcribe_lyrics_runs_asr_on_ladder_stem_then_cache_hits(tmp_path, _plenty_of_vram):
    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(["vocals"])
    asr = StubAsr(language="zz", words=list(_SCRIPTED_WORDS))

    first = transcribe_lyrics(
        str(audio), cache=fs, asr=asr, tagger=_mixed_vocal_tagger(), engine=engine
    )

    assert first.status == "ok"
    assert first.language == "zz"
    assert [w.word for w in first.words] == ["hold", "the", "line."]
    assert asr.calls == 1
    assert len(engine.calls) == 1

    second = transcribe_lyrics(
        str(audio), cache=fs, asr=asr, tagger=_mixed_vocal_tagger(), engine=engine
    )

    assert second == first
    assert asr.calls == 1  # cache HIT -- ASR never re-ran
    assert len(engine.calls) == 1  # -- and separation never re-ran either


@pytest.mark.analysis
def test_transcribe_lyrics_caches_under_lyrics_lines_sections_kinds(tmp_path, _plenty_of_vram):
    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(["vocals"])
    asr = StubAsr(language="zz", words=list(_SCRIPTED_WORDS))

    transcribe_lyrics(str(audio), cache=fs, asr=asr, tagger=_mixed_vocal_tagger(), engine=engine)

    file_hash = cachefs.pcm_hash(str(audio))
    art_dir = fs.artifact_dir(file_hash)
    assert (art_dir / "lyrics.json").is_file()
    assert (art_dir / "lines.json").is_file()
    assert (art_dir / "sections.json").is_file()


@pytest.mark.analysis
def test_transcribe_lyrics_cache_busts_on_asr_model_id_change(
    tmp_path, _plenty_of_vram, monkeypatch
):
    """Fix wave 1 (Important, 2d): a cached transcript written under ``asr_model_id="model-x"``
    must MISS (not silently serve stale content) when a later call's meta carries a DIFFERENT
    id -- the belt-and-suspenders bust ``_lyrics_meta``'s docstring calls out. Call-counted:
    the second call re-invokes the ASR (and re-separates), proving the recompute path was
    genuinely taken, not just that the returned content happens to differ."""
    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(["vocals"])
    asr = StubAsr(language="zz", words=list(_SCRIPTED_WORDS))

    monkeypatch.setattr(reference_tools, "_asr_model_id", lambda: "model-x")
    first = transcribe_lyrics(
        str(audio), cache=fs, asr=asr, tagger=_mixed_vocal_tagger(), engine=engine
    )
    assert asr.calls == 1
    assert first.status == "ok"

    monkeypatch.setattr(reference_tools, "_asr_model_id", lambda: "model-y")
    second = transcribe_lyrics(
        str(audio), cache=fs, asr=asr, tagger=_mixed_vocal_tagger(), engine=engine
    )
    assert asr.calls == 2  # cache MISS on the asr_model_id change -> genuinely recomputed
    assert second.status == "ok"  # same scripted ASR output, but a fresh compute, not a HIT


@pytest.mark.analysis
def test_transcribe_lyrics_propagates_separation_error(tmp_path, _plenty_of_vram):
    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")

    class _BrokenEngine:
        def separate(self, wav_path, entry, out_dir):
            raise SeparationError("no compatible checkpoint for this box")

    with pytest.raises(SeparationError):
        transcribe_lyrics(
            str(audio),
            cache=fs,
            asr=StubAsr(),
            tagger=_mixed_vocal_tagger(),
            engine=_BrokenEngine(),
        )


def test_transcribe_lyrics_raises_reference_env_missing_without_analysis_extra(
    tmp_path, _plenty_of_vram, monkeypatch
):
    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "orpheus_mcp.analysis.timeline":
            raise ImportError("no panns_inference")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ReferenceEnvMissing):
        transcribe_lyrics(str(audio), cache=fs, asr=StubAsr())


# --------------------------------------------------------------------------- #
# locate_lyric (tool layer): fetches via transcribe_lyrics, then the pure scorer.
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
def test_locate_lyric_tool_finds_scripted_quote_and_reuses_the_cache(tmp_path, _plenty_of_vram):
    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")
    engine = StubEngine(["vocals"])
    asr = StubAsr(language="zz", words=list(_SCRIPTED_WORDS))

    result = locate_lyric(
        str(audio), "hold the line", cache=fs, asr=asr, tagger=_mixed_vocal_tagger(), engine=engine
    )

    assert isinstance(result, LocateResult)
    assert result.status == "ok"
    assert result.matches
    assert result.matches[0].score == pytest.approx(1.0)
    assert asr.calls == 1

    # a second locate_lyric call reuses the already-cached lyrics index (no re-ASR).
    locate_lyric(
        str(audio), "hold the line", cache=fs, asr=asr, tagger=_mixed_vocal_tagger(), engine=engine
    )
    assert asr.calls == 1


@pytest.mark.analysis
def test_locate_lyric_tool_on_no_vocals_file_returns_defined_result(tmp_path, _plenty_of_vram):
    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    fs = CacheFS(tmp_path / "cache")

    result = locate_lyric(
        str(audio),
        "anything",
        cache=fs,
        asr=StubAsr(words=[]),
        tagger=_absent_vocal_tagger(),
        engine=StubEngine(["vocals"]),
    )

    assert result.status == "no_vocals"
    assert result.matches == []


# --------------------------------------------------------------------------- #
# MCP registration: locate_lyric always (librosa tier); transcribe_lyrics additionally
# gated on faster_whisper (nested inside the panns/torch probe, matching separate_region's
# already-established pattern for a desktop-tier extra).
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
async def test_locate_lyric_tool_registered_read_only():
    mcp = FastMCP(name="t")
    register_tools(mcp, profile="default")
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "locate_lyric" in tools
    assert tools["locate_lyric"].annotations.readOnlyHint is True


@pytest.mark.analysis
async def test_locate_lyric_registered_even_without_panns_or_torch(monkeypatch):
    import importlib.util as ilu

    real_find_spec = ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("panns_inference", "torch"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(ilu, "find_spec", fake_find_spec)

    mcp = FastMCP(name="t")
    with pytest.warns(UserWarning, match="instrument_timeline not registered"):
        reference_tools.register(mcp)

    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "locate_lyric" in tools  # librosa tier -- unaffected by the panns/torch probe
    assert "transcribe_lyrics" not in tools


@pytest.mark.analysis
async def test_transcribe_lyrics_tool_registered_when_faster_whisper_available():
    # transcribe_lyrics registers only when all of these are importable (reference.py's
    # registration probes); faster-whisper lives in the [separation] extra, not [analysis].
    pytest.importorskip("faster_whisper")
    pytest.importorskip("panns_inference")
    pytest.importorskip("torch")
    mcp = FastMCP(name="t")
    register_tools(mcp, profile="default")
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "transcribe_lyrics" in tools
    assert tools["transcribe_lyrics"].annotations.readOnlyHint is True


async def test_transcribe_lyrics_tool_surfaces_lyrics_asr_env_missing_as_typed_error(
    tmp_path, monkeypatch
):
    """Fix wave 1 (Minor, part 3): end-to-end proof that a genuine faster-whisper absence AT
    ASR-LOAD TIME (not registration time -- the tool IS registered here, since faster_whisper
    really is importable in this env, per the ``importorskip`` below) surfaces through the
    ``transcribe_lyrics`` MCP tool RESULT as a typed ``fastmcp.exceptions.ToolError`` carrying
    ``LyricsAsrEnvMissing``'s own message -- never a raw, unhandled traceback escaping the call
    (the §9 ``capped_by_missing_extra`` pattern). Hides faster_whisper via a ``sys.modules``
    sentinel -- the reliable way to force ``from faster_whisper import WhisperModel`` to raise
    ``ImportError`` regardless of what's actually installed, applied at CALL time rather than
    the registration-time ``find_spec`` probes this file already uses elsewhere."""
    pytest.importorskip("faster_whisper")
    pytest.importorskip("panns_inference")
    pytest.importorskip("torch")

    audio = _write_lyrics_test_wav(tmp_path / "mix.wav")
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path / "orpheus_cache"))

    # Bypass the real separation/tagger ladder (heavy models, desktop-only) -- the subject
    # under test is the ASR-load failure surfacing through the tool, not the ladder itself.
    def _fake_ladder(file, pcm, fs, *, tagger, engine, max_seconds):
        return pcm, False  # "vocal stem" == the loud raw mix; build_lyrics's VAD gate stays live

    monkeypatch.setattr(reference_tools, "_vocal_stem_via_ladder", _fake_ladder)

    # Register FIRST, while faster_whisper is genuinely present (a ``sys.modules`` sentinel
    # also fools the registration-time ``find_spec`` probe -- ``importlib.util.find_spec``
    # returns ``None`` for a name mapped to ``None`` in ``sys.modules`` -- so the tool must be
    # registered before the sentinel goes in, or it never gets registered at all).
    mcp = FastMCP(name="t")
    register_tools(mcp, profile="default")

    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # force ImportError on load

    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="faster-whisper is not installed"):
            await client.call_tool("transcribe_lyrics", {"file": str(audio)})


@pytest.mark.analysis
async def test_transcribe_lyrics_not_registered_without_faster_whisper(monkeypatch):
    import importlib.util as ilu

    real_find_spec = ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "faster_whisper":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(ilu, "find_spec", fake_find_spec)

    mcp = FastMCP(name="t")
    reference_tools.register(mcp)

    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "transcribe_lyrics" not in tools
    assert "separate_region" in tools  # unaffected -- gated independently
    assert "locate_lyric" in tools


# --------------------------------------------------------------------------- #
# Desktop/gpu-optional real end-to-end: transcribe_lyrics on ci_vocal (ladder -> real vocal
# stem via the registry-resolved engine -> real faster-whisper ASR on CPU), asserting
# locate_lyric finds an exact truth-line quote. Run yourself at the desktop (Task 8.2 brief)
# -- CPU is fine here; only the SEPARATE GPU ASR smoke (test_lyrics.py) is GPU-gated.
# --------------------------------------------------------------------------- #

_CI_VOCAL_MIX = Path(__file__).resolve().parent / "fixtures" / "audio" / "ci_vocal" / "mix.flac"


@pytest.mark.desktop
def test_transcribe_lyrics_and_locate_lyric_end_to_end_on_ci_vocal(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper")
    pytest.importorskip("panns_inference")
    torch = pytest.importorskip("torch")
    if not _CI_VOCAL_MIX.is_file():
        pytest.skip("ci_vocal fixture missing")

    import json

    truth = json.loads((_CI_VOCAL_MIX.parent / "truth.json").read_text(encoding="utf-8"))
    truth_line = truth["lyrics"]["lines"][0]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # force CPU per the brief

    fs = CacheFS(tmp_path / "cache")
    result = transcribe_lyrics(str(_CI_VOCAL_MIX), cache=fs)
    assert result.status == "ok"

    located = locate_lyric(str(_CI_VOCAL_MIX), truth_line, cache=fs)
    assert located.status == "ok"
    assert located.matches
    assert located.matches[0].score >= CFG.LYRIC_MATCH_MIN_SCORE
