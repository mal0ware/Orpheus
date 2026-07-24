"""``tools/reference.py``'s ``get_chords`` (Task 3.2, §13/§9): the read-through ``"chords"``
cache over the full baseline lane (``harmony.py``'s recognizer + §13.3 extensions + §13.2
slash + §13.4 theory, composed by ``_full_baseline_lane`` since ``harmony.py`` ships no single
compose entry point), plus the post-cache region filter (§3.1d clamping).

Contract tests run through an in-memory FastMCP ``Client`` (``tests/test_m2_tools.py``'s
pattern); the compute-count / cache-sharing assertions call ``get_chords``/``recon_reference``
directly against an explicit ``CacheFS`` (``tests/test_recon.py``'s pattern) since a monkeypatch
on a module-level helper is easiest to see from the same process, not through the wire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastmcp import Client, FastMCP

from orpheus_mcp.analysis import cachefs
from orpheus_mcp.analysis.cachefs import CacheFS, decode
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.recon import run_recon
from orpheus_mcp.models import Window
from orpheus_mcp.registry import register_tools
from orpheus_mcp.tools import reference as reference_tools
from orpheus_mcp.tools.reference import get_chords, recon_reference

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


def test_full_lane_fetch_computes_once_then_cache_hits(audio_path, fs, monkeypatch):
    calls = {"n": 0}
    real = reference_tools._full_baseline_lane

    def counted(pcm, recon):
        calls["n"] += 1
        return real(pcm, recon)

    monkeypatch.setattr(reference_tools, "_full_baseline_lane", counted)

    first = get_chords(str(audio_path), cache=fs)
    assert calls["n"] == 1
    assert first.segments  # a sane lane, not empty

    second = get_chords(str(audio_path), cache=fs)
    assert calls["n"] == 1  # cache HIT — the lane was never recomputed
    assert second == first


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


def test_region_end_10x_duration_clamps_and_keeps_full_tail(audio_path, fs):
    full = get_chords(str(audio_path), cache=fs)
    recon = run_recon(decode(str(audio_path)))
    duration = recon.duration_s

    region = Window(start_s=0.0, end_s=duration * 10.0)
    filtered = get_chords(str(audio_path), region, cache=fs)

    # clamps to [0, duration] cleanly (no error) and keeps every segment, incl. the tail
    assert filtered.segments == full.segments
    assert filtered.segments[-1].end_s == pytest.approx(full.segments[-1].end_s)


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


def test_chords_meta_sidecar_records_produced_by_baseline(audio_path, fs):
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


async def test_get_chords_tool_registered_read_only(mcp_client):
    async with mcp_client as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert "get_chords" in tools
    assert tools["get_chords"].annotations.readOnlyHint is True


async def test_get_chords_tool_full_lane_via_client(audio_path, mcp_client):
    async with mcp_client as c:
        result = await c.call_tool("get_chords", {"file": str(audio_path)})
    lane = result.data
    assert lane.segments
    assert lane.grid in {"beat", "fixed"}


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
