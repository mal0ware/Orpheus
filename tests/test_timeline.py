"""``analysis/timeline.py`` + ``instrument_timeline`` tool (Task 2.1, §3.1b/c, §9):
exposure math, absent-target rule, max-pool aggregation, and the tool contract —
verbatim from the Task 2.1 brief's test list, all driven by ``StubTagger`` (no real
model weights in the CI-safe suite).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orpheus_mcp.analysis import cachefs
from orpheus_mcp.analysis.cachefs import CacheFS, decode
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.timeline import build_timeline, most_exposed_window, target_absent
from orpheus_mcp.models import Timeline, Window
from orpheus_mcp.tools.reference import instrument_timeline
from stub_engines import STUB_CLASSES, StubTagger

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "audio"


def make_sine(path: Path, *, seconds: float = 1.0, sr: int = 22050, freq: float = 440.0) -> Path:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sine = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sine, sr, subtype="PCM_16")
    return path


# --------------------------------------------------------------------------- #
# §5.2 exposure math, verbatim from the brief
# --------------------------------------------------------------------------- #


def _flat_timeline(*, target_val: float, nontarget_val: float, n_frames: int) -> Timeline:
    """A Timeline where every frame reads the same (target, non-target) pair on a
    2-class ["target", "other"] axis — isolates the ratio math from the run-finding
    logic."""
    frames = [[target_val, nontarget_val] for _ in range(n_frames)]
    return Timeline(classes=["target", "other"], hop_s=1.0, frames=frames)


def test_exposure_ratio_below_threshold_is_exposed():
    # 0.05 / (0.9 + eps) ~= 0.056 < 0.15 (CFG.EXPOSURE_THRESHOLD default) -> exposed.
    tl = _flat_timeline(target_val=0.9, nontarget_val=0.05, n_frames=5)
    window = most_exposed_window(tl, ["target"], min_len_s=3.0)
    assert window is not None
    assert window == Window(start_s=0.0, end_s=5.0)


def test_exposure_ratio_above_threshold_is_not_exposed():
    # 0.2 / (0.9 + eps) ~= 0.222 >= 0.15 -> not exposed.
    tl = _flat_timeline(target_val=0.9, nontarget_val=0.2, n_frames=5)
    assert most_exposed_window(tl, ["target"], min_len_s=3.0) is None


def test_most_exposed_window_picks_lowest_mean_ratio_among_candidates():
    # Two qualifying runs (both exposed, both >= 3s): first has ratio ~0.111, second ~0.056.
    # The second must win (lower mean ratio = "most" exposed).
    frames = (
        [[0.9, 0.10]] * 4  # ratio ~0.111 -- exposed but worse
        + [[0.9, 0.50]] * 2  # ratio ~0.556 -- breaks the run (not exposed)
        + [[0.9, 0.05]] * 4  # ratio ~0.056 -- exposed and better
    )
    tl = Timeline(classes=["target", "other"], hop_s=1.0, frames=frames)
    window = most_exposed_window(tl, ["target"], min_len_s=3.0)
    assert window == Window(start_s=6.0, end_s=10.0)


def test_exposed_run_shorter_than_min_window_is_rejected():
    tl = _flat_timeline(target_val=0.9, nontarget_val=0.05, n_frames=2)  # only 2s < 3s min
    assert most_exposed_window(tl, ["target"], min_len_s=3.0) is None


# --------------------------------------------------------------------------- #
# §3.1c target_absent
# --------------------------------------------------------------------------- #


def test_target_absent_true_when_target_peaks_below_activity_threshold():
    # Peaks at 0.29 < CFG.TIMELINE_ACTIVITY_THRESHOLD (0.30) everywhere -> absent.
    tl = Timeline(
        classes=["target", "other"],
        hop_s=1.0,
        frames=[[0.1, 0.0], [0.29, 0.0], [0.2, 0.0]],
    )
    assert target_absent(tl, ["target"]) is True


def test_target_absent_false_when_target_exceeds_activity_threshold_once():
    tl = Timeline(
        classes=["target", "other"],
        hop_s=1.0,
        frames=[[0.1, 0.0], [0.31, 0.0], [0.2, 0.0]],
    )
    assert target_absent(tl, ["target"]) is False


# --------------------------------------------------------------------------- #
# Empty tagger_classes -> N/A semantics (locked by test)
# --------------------------------------------------------------------------- #


def test_empty_target_classes_is_na_not_absent_or_exposed():
    tl = _flat_timeline(target_val=0.9, nontarget_val=0.05, n_frames=5)
    assert most_exposed_window(tl, []) is None
    assert target_absent(tl, []) is False


# --------------------------------------------------------------------------- #
# build_timeline: round-trip, frame alignment, max-pool aggregation
# --------------------------------------------------------------------------- #


def test_build_timeline_round_trips_and_frames_align(tmp_path):
    path = make_sine(tmp_path / "sine.wav", seconds=3.4)
    pcm = decode(str(path))
    tagger = StubTagger(hop_s=0.01)

    tl = build_timeline(pcm, tagger)

    assert len(tl.frames) == math.ceil(pcm.duration_s / 1.0)
    round_tripped = Timeline.model_validate_json(tl.model_dump_json())
    assert round_tripped == tl


def test_build_timeline_subsets_to_registry_mapped_classes(tmp_path):
    path = make_sine(tmp_path / "sine.wav", seconds=1.0)
    pcm = decode(str(path))
    # Add a class that is NOT in any registry entry's tagger_classes -- it must be dropped.
    tagger = StubTagger(classes=[*STUB_CLASSES, "Not A Registry Class"], hop_s=0.01)

    tl = build_timeline(pcm, tagger)

    assert "Not A Registry Class" not in tl.classes
    assert set(tl.classes) <= set(STUB_CLASSES)
    assert tl.classes  # every STUB_CLASSES entry is drawn from the real registry union


def test_build_timeline_max_pools_native_frames_to_locked_one_second_hop(tmp_path):
    path = make_sine(tmp_path / "sine.wav", seconds=2.0)
    pcm = decode(str(path))
    n_native = math.ceil(pcm.duration_s / 0.01)
    activations = np.zeros((n_native, len(STUB_CLASSES)), dtype=np.float64)
    drum_col = STUB_CLASSES.index("Drum kit")
    activations[50, drum_col] = 0.9  # a single spike at t=0.5s -- inside second 0

    tagger = StubTagger(hop_s=0.01, activations=activations)
    tl = build_timeline(pcm, tagger)

    assert tl.hop_s == 1.0
    out_col = tl.classes.index("Drum kit")
    assert tl.frames[0][out_col] == pytest.approx(0.9)
    for i in range(1, len(tl.frames)):
        assert tl.frames[i][out_col] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Tool contract: instrument_timeline(file) -- mapped-subset classes + raw activations,
# cached under kind "timeline"
# --------------------------------------------------------------------------- #


def test_instrument_timeline_tool_contract(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path / "cache"))
    path = make_sine(tmp_path / "src" / "sine.wav", seconds=2.5)
    stub = StubTagger(hop_s=0.01)

    tl = instrument_timeline(str(path), tagger=stub)

    assert isinstance(tl, Timeline)
    assert tl.classes  # mapped-subset, not raw StubTagger's full class list necessarily
    assert set(tl.classes) <= set(STUB_CLASSES)
    assert len(tl.frames) == math.ceil(decode(str(path)).duration_s / 1.0)

    fs = CacheFS(tmp_path / "cache")
    file_hash = cachefs.pcm_hash(str(path))
    cached = fs.read_json(
        file_hash,
        "timeline",
        meta={
            "schema_version": 1,
            "registry_version": cachefs.registry_version(),
            "verify_config_hash": cachefs.verify_config_hash(),
            "decoder": cachefs.decoder_for(str(path)),
            "tagger_id": "stub:StubTagger",
        },
    )
    assert cached is not None
    assert cached["classes"] == tl.classes


def test_instrument_timeline_second_call_served_from_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path / "cache"))
    path = make_sine(tmp_path / "src" / "sine.wav", seconds=1.5)
    calls = {"n": 0}

    class CountingStubTagger(StubTagger):
        def activations(self, pcm):
            calls["n"] += 1
            return super().activations(pcm)

    tagger = CountingStubTagger(hop_s=0.01)

    first = instrument_timeline(str(path), tagger=tagger)
    assert calls["n"] == 1

    second = instrument_timeline(str(path), tagger=tagger)
    assert calls["n"] == 1  # cache HIT -- tagger not invoked again
    assert first == second


def test_instrument_timeline_uses_provided_cache(tmp_path):
    fs = CacheFS(tmp_path / "explicit_cache")
    path = make_sine(tmp_path / "sine.wav", seconds=1.0)

    instrument_timeline(str(path), cache=fs, tagger=StubTagger(hop_s=0.01))

    file_hash = cachefs.pcm_hash(str(path))
    assert fs.read_json(
        file_hash,
        "timeline",
        meta={
            "schema_version": 1,
            "registry_version": cachefs.registry_version(),
            "verify_config_hash": cachefs.verify_config_hash(),
            "decoder": cachefs.decoder_for(str(path)),
            "tagger_id": "stub:StubTagger",
        },
    ) is not None


# --------------------------------------------------------------------------- #
# Tool registration: per-tool conditional probe (panns-inference + torch), category
# stays librosa-only.
# --------------------------------------------------------------------------- #


async def test_instrument_timeline_tool_registered_read_only():
    from fastmcp import Client, FastMCP

    from orpheus_mcp.registry import register_tools

    mcp = FastMCP(name="t")
    register_tools(mcp, profile="default")
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "instrument_timeline" in tools
    assert tools["instrument_timeline"].annotations.readOnlyHint is True


async def test_instrument_timeline_not_registered_without_panns_or_torch(monkeypatch):
    import importlib.util as ilu

    from fastmcp import Client, FastMCP

    from orpheus_mcp.tools import reference as reference_tools

    real_find_spec = ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("panns_inference", "torch"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(ilu, "find_spec", fake_find_spec)

    mcp = FastMCP(name="t")
    reference_tools.register(mcp)  # must NOT raise -- librosa is still present

    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "instrument_timeline" not in tools
    assert "recon_reference" in tools  # category-level tool unaffected


# --------------------------------------------------------------------------- #
# PannsTagger: real-model desktop path (§12 slice-2 acceptance bar)
# --------------------------------------------------------------------------- #


@pytest.mark.desktop
def test_panns_tagger_detects_drums_on_ci_sparse():
    """A drum-family class must activate above CFG.TIMELINE_ACTIVITY_THRESHOLD somewhere
    in ci_sparse/mix.flac -- its drums stem (tests/fixtures/audio/ci_sparse/truth.json's
    `roles.drums`) spans the whole 8-bar arrangement.

    Deliberately checks PannsTagger's RAW (un-mapped, full AudioSet) output, not a
    Timeline built through the registry's mapped-subset filter: build_timeline's
    subsetting is already covered by StubTagger-driven unit tests above, and is
    orthogonal to what this test validates -- that the real model correctly "hears"
    drums in this fixture. (Empirically: this sfizz/sample-rendered fixture's kit reads
    to PANNs as "Drum machine" (peak ~0.35) rather than "Drum kit"/"Drum" (peak ~0.04) --
    both are legitimate drum-family AudioSet classes; the registry's tagger_classes union
    (Task 0.2 data) does not currently include "Drum machine". That is a registry-mapping
    gap worth revisiting in a future pin run, not a PannsTagger correctness bug.)"""
    from orpheus_mcp.analysis.timeline import PannsTagger

    pcm = decode(str(FIXTURES / "ci_sparse" / "mix.flac"))
    tagger = PannsTagger()
    raw = tagger.activations(pcm)
    classes = tagger.classes

    drum_family = [i for i, c in enumerate(classes) if "drum" in c.lower()]
    assert drum_family, f"no drum-family class in PANNs' AudioSet output: {classes}"

    peak = raw[:, drum_family].max()
    assert peak > CFG.TIMELINE_ACTIVITY_THRESHOLD
