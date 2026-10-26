"""``analysis/router.py::analyze_reference`` (Task 12.1, §9): the orchestrator + answer
envelope. One golden per question class against stub engines + a tiny CI wav (no weights, no
GPU, no network) — every heavy stage is an injection seam (stub tagger/engine/asr, a scripted
``verify``/``resolver``/``timeline``, a stubbed basic-pitch ``transcribe``, a fake ``clock``).

Envelope goldens assert KEY PRESENCE/ABSENCE + the critical values (the §9 packaging rule:
"only the artifact groups the question class needed are populated"), never a full JSON snapshot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orpheus_mcp.analysis import model_registry
from orpheus_mcp.analysis.cachefs import CacheFS
from orpheus_mcp.analysis.router import analyze_reference
from orpheus_mcp.analysis.transcribe_cli import NoteEvent
from orpheus_mcp.models import (
    BeatGrid,
    KeyEstimate,
    LocateMatch,
    LocateResult,
    Recon,
    Timeline,
    Window,
)
from orpheus_mcp.tools import reference as reference_tools
from stub_engines import ScriptedGates, StubEngine, StubTagger

SR = 22050
RANK1 = "4stem-c/htdemucs-ft"  # multi-stem, vram 0, targets vocals/drums/bass/other
_REG = {e.id: e for e in model_registry.load_registry(allow_unpinned=True)}


@pytest.fixture(autouse=True)
def _plenty_of_vram(monkeypatch):
    monkeypatch.setattr(model_registry, "free_vram_gb", lambda: 24.0)


@pytest.fixture
def fs(tmp_path) -> CacheFS:
    return CacheFS(tmp_path / "cache")


def _resolver(mapping):
    return lambda target: [_REG[i] for i in mapping.get(target, [])]


_VOCALS_MAP = {"vocals": [RANK1], "bass": [RANK1], "drums": [RANK1]}
# "strings" is checkpoint-less (rank-1 empty) but bass/drums still yield the R3 complement model.
_STRINGS_MAP = {"strings": [], "bass": [RANK1], "drums": [RANK1], "vocals": [RANK1]}


def _wav(tmp_path: Path, *, dur_s: float = 6.0, name: str = "mix.wav") -> str:
    """A short decaying-triad tone — enough for a real recon/chord lane and to be identity-
    separated by ``StubEngine``."""
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    env = np.exp(-2.0 * (t % 0.5)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    for f in (261.63, 329.63, 392.00):
        y += (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32)
    y *= env
    path = tmp_path / name
    sf.write(str(path), y, SR, subtype="PCM_16")
    return str(path)


def _recon(*, dur=30.0, downbeats=(0.0, 2.0, 4.0, 6.0, 8.0), kind="beat", conf=0.9) -> Recon:
    return Recon(
        duration_s=dur,
        sample_rate=44100,
        lufs_integrated=-14.0,
        tempo_bpm=120.0,
        grid=BeatGrid(
            times_s=list(downbeats),
            downbeats_s=list(downbeats),
            meter=(4, 4),
            confidence=conf,
            meter_confidence=0.5,
            kind=kind,
        ),
        tuning_offset_cents=0.0,
        key=KeyEstimate(root="C", mode="major", confidence=0.8),
        key_alternates=[],
        sections=[],
        band_energy_db={},
    )


_TL_CLASSES = ["Singing", "Speech", "Guitar"]


def _mixed_timeline(n: int = 8) -> Timeline:
    """Vocals present (Singing 0.5 > activity) but NOT exposed (Guitar bleeds equally)."""
    return Timeline(classes=_TL_CLASSES, hop_s=1.0, frames=[[0.5, 0.0, 0.5]] * n)


def _absent_timeline(n: int = 8) -> Timeline:
    """Singing/Speech never exceed the activity threshold → §3.1c absent."""
    return Timeline(classes=_TL_CLASSES, hop_s=1.0, frames=[[0.1, 0.1, 0.5]] * n)


def _patch_recon(monkeypatch, recon: Recon) -> None:
    monkeypatch.setattr(reference_tools, "recon_reference", lambda file, *, cache=None: recon)


# --------------------------------------------------------------------------- #
# recon / timeline / chords — the artifact-group packaging rule (§9)
# --------------------------------------------------------------------------- #


@pytest.mark.analysis
def test_recon_envelope_has_no_timeline_key_and_populated_summary(tmp_path, fs):
    env = analyze_reference(_wav(tmp_path), question_focus=None, cache=fs)
    assert env.question_class == "recon"
    assert env.tier_reached == "T1"
    assert env.recon_summary["tempo_bpm"] > 0
    assert "timeline" not in env.artifacts
    assert "chords" not in env.artifacts
    assert "stems" not in env.artifacts
    assert env.artifacts == {}


@pytest.mark.analysis
def test_timeline_envelope_has_only_timeline_group(tmp_path, fs):
    env = analyze_reference(
        _wav(tmp_path), question_focus="timeline", cache=fs, tagger=StubTagger()
    )
    assert env.question_class == "timeline"
    assert env.tier_reached == "T2"
    assert "timeline" in env.artifacts
    assert "chords" not in env.artifacts
    assert "stems" not in env.artifacts


@pytest.mark.analysis
def test_chords_envelope_has_only_chords_group(tmp_path, fs):
    env = analyze_reference(_wav(tmp_path), question_focus="chords", cache=fs)
    assert env.question_class == "chords"
    assert env.tier_reached == "T2"
    assert "chords" in env.artifacts
    assert "timeline" not in env.artifacts


def test_unknown_question_focus_raises(tmp_path, fs):
    with pytest.raises(ValueError, match="question_focus"):
        analyze_reference(_wav(tmp_path), question_focus="bogus", cache=fs)


@pytest.mark.analysis
def test_cache_hits_ride_the_envelope(tmp_path, fs):
    path = _wav(tmp_path)
    analyze_reference(path, question_focus="chords", cache=fs)  # populate
    env = analyze_reference(path, question_focus="chords", cache=fs)  # now a HIT
    assert any(n.startswith("cache:") for n in env.notes)
    assert any("HIT" in n for n in env.notes)


# --------------------------------------------------------------------------- #
# sound_at — stems + gates + descriptors (StubEngine path)
# --------------------------------------------------------------------------- #


def test_sound_at_envelope_has_stems_gates_descriptors(tmp_path, fs, monkeypatch):
    _patch_recon(monkeypatch, _recon())
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="sound_at",
        target="vocals",
        region=Window(start_s=1.0, end_s=4.0),
        cache=fs,
        engine=StubEngine(),
        verify=ScriptedGates(["pass"]),
        resolver=_resolver(_VOCALS_MAP),
        timeline=_mixed_timeline(),
    )
    assert env.question_class == "sound_at"
    assert env.tier_reached == "T3"  # rank-1 passed, no escalation
    assert "stems" in env.artifacts
    assert "descriptors" in env.artifacts
    assert "vocals" in env.gates
    assert env.gates["vocals"].overall == "pass"
    assert "timeline" not in env.artifacts


@pytest.mark.analysis
def test_sound_at_requires_target(tmp_path, fs):
    with pytest.raises(ValueError, match="requires a `target`"):
        analyze_reference(_wav(tmp_path), question_focus="sound_at", cache=fs)


def _exposed_timeline(n: int = 5) -> Timeline:
    """Singing loud, everything else near-silent → §3.1b exposed window."""
    return Timeline(classes=_TL_CLASSES, hop_s=1.0, frames=[[0.9, 0.0, 0.01]] * n)


def test_sound_at_exposed_target_characterizes_mix_region_at_t2(tmp_path, fs, monkeypatch):
    # §3.1b: when the target is exposed in the raw mix the ladder skips separation; the envelope
    # still delivers descriptors — from the mix window — at T2 cost, with NO per-stem gates.
    _patch_recon(monkeypatch, _recon())
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="sound_at",
        target="vocals",
        cache=fs,
        engine=StubEngine(),
        verify=ScriptedGates([]),  # never reached — exposure is a pre-flight verdict
        resolver=_resolver(_VOCALS_MAP),
        timeline=_exposed_timeline(),
    )
    assert env.tier_reached == "T2"
    assert "descriptors" in env.artifacts
    assert env.artifacts["descriptors"][0]["source"]["stem_type"] == "mix-region"
    assert "exposed_window" in env.artifacts
    assert env.gates == {}
    assert "stems" not in env.artifacts
    assert any("exposed in the raw mix" in n for n in env.notes)


# --------------------------------------------------------------------------- #
# absent-target (§3.1c) — T2-cost envelope, "target not detected", NO stems
# --------------------------------------------------------------------------- #


def test_absent_target_is_t2_with_note_and_no_stems(tmp_path, fs, monkeypatch):
    _patch_recon(monkeypatch, _recon())
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="sound_at",
        target="vocals",
        cache=fs,
        engine=StubEngine(),
        verify=ScriptedGates(["pass"]),
        resolver=_resolver(_VOCALS_MAP),
        timeline=_absent_timeline(),
    )
    assert env.tier_reached == "T2"
    assert any("target not detected" in n for n in env.notes)
    assert "stems" not in env.artifacts
    assert "descriptors" not in env.artifacts
    assert env.gates == {}


# --------------------------------------------------------------------------- #
# missing [separation] extra — capped_by_missing_extra, answered at highest tier
# --------------------------------------------------------------------------- #


class _RaisingEngine:
    requires_stereo = False

    def separate(self, wav_path, entry, out_dir):  # noqa: ARG002
        from orpheus_mcp.analysis.separation import SeparatorEnvMissing

        raise SeparatorEnvMissing("isolated separator env is not installed")


def test_missing_separation_caps_by_extra_at_highest_available(tmp_path, fs, monkeypatch):
    _patch_recon(monkeypatch, _recon())
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="sound_at",
        target="vocals",
        cache=fs,
        engine=_RaisingEngine(),
        verify=ScriptedGates(["pass"]),
        resolver=_resolver(_VOCALS_MAP),
        timeline=_mixed_timeline(),
    )
    assert env.capped_by_missing_extra == "separation"
    assert env.tier_reached == "T2"  # timeline (T2) is the highest we reached; no stem
    assert "stems" not in env.artifacts


# --------------------------------------------------------------------------- #
# budget_capped — fake clock trips the §6 budget guard mid-ladder
# --------------------------------------------------------------------------- #


def test_budget_capped_envelope(tmp_path, fs, monkeypatch):
    _patch_recon(monkeypatch, _recon())
    ticks = iter([0.0] + [1000.0] * 50)
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="sound_at",
        target="vocals",
        max_seconds=1.0,
        cache=fs,
        engine=StubEngine(),
        verify=ScriptedGates(["pass"]),
        resolver=_resolver(_VOCALS_MAP),
        timeline=_mixed_timeline(),
        clock=lambda: next(ticks),
    )
    assert env.budget_capped is True


# --------------------------------------------------------------------------- #
# R3 complement (carried duty) — complement=True rides end-to-end to the descriptor
# --------------------------------------------------------------------------- #


def test_complement_stem_yields_complement_descriptor(tmp_path, fs, monkeypatch):
    _patch_recon(monkeypatch, _recon())
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="sound_at",
        target="strings",  # checkpoint-less → ladder skips rank1/R1/R2, runs R3 complement
        region=Window(start_s=1.0, end_s=4.0),
        max_tier="T4",
        cache=fs,
        engine=StubEngine(),
        verify=ScriptedGates([]),  # every gate passes (preconditions + residual)
        resolver=_resolver(_STRINGS_MAP),
        timeline=_mixed_timeline(),
    )
    assert env.tier_reached == "T4"  # escalated into the ladder's T4 R3 rung
    assert "descriptors" in env.artifacts
    assert env.artifacts["descriptors"][0]["source"]["complement"] is True
    # The R3 residual stem carries the complement flag in its notes, which is what
    # resolve_source turned into the descriptor's complement=True above.
    assert any("complement=True" in n for n in env.artifacts["stems"]["strings"]["notes"])


# --------------------------------------------------------------------------- #
# lyric_navigate (§14.2) — [match.end_s, +8 bars] window, timeline diff, grid math
# --------------------------------------------------------------------------- #


def _lyric_timeline() -> Timeline:
    """Singing present in [0,10); Guitar rises in [10,30) — so the +8-bars window after a
    match ending at 10 s shows Guitar RISING vs the preceding window."""
    before = [[0.8, 0.0]] * 10  # 0..10 s: Singing loud, Guitar silent
    after = [[0.1, 0.9]] * 20  # 10..30 s: Guitar loud
    return Timeline(classes=["Singing", "Guitar"], hop_s=1.0, frames=before + after)


def _patch_locate(monkeypatch, end_s: float) -> None:
    result = LocateResult(
        status="ok",
        matches=[
            LocateMatch(
                start_s=end_s - 2.0,
                end_s=end_s,
                matched_text="the line",
                score=1.0,
                section=None,
                occurrence=1,
            )
        ],
    )
    monkeypatch.setattr(
        reference_tools,
        "locate_lyric",
        lambda file, quote, occurrence=None, **kw: result,
    )


def test_lyric_navigate_beat_grid_eight_bar_window(tmp_path, fs, monkeypatch):
    # Downbeat spacing 1.0 s → 8 bars = +8 s.
    _patch_recon(monkeypatch, _recon(dur=30.0, downbeats=(0.0, 1.0, 2.0, 3.0, 4.0), kind="beat"))
    _patch_locate(monkeypatch, end_s=10.0)
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="lyric_navigate",
        quote="the line",
        cache=fs,
        timeline=_lyric_timeline(),
    )
    assert env.tier_reached == "T2"
    assert env.artifacts["grid_mode"] == "beat"
    assert env.artifacts["bar_seconds"] == pytest.approx(1.0)
    reg = env.artifacts["region"]
    assert reg["start_s"] == pytest.approx(10.0)
    assert reg["end_s"] == pytest.approx(18.0)  # +8 bars * 1.0 s
    assert "Guitar" in env.artifacts["rising"]
    assert "Singing" not in env.artifacts["rising"]


def test_lyric_navigate_fixed_grid_plus_16s_fallback(tmp_path, fs, monkeypatch):
    from orpheus_mcp.analysis.config import CFG

    _patch_recon(monkeypatch, _recon(dur=30.0, kind="fixed"))
    _patch_locate(monkeypatch, end_s=10.0)
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="lyric_navigate",
        quote="the line",
        cache=fs,
        timeline=_lyric_timeline(),
    )
    assert env.artifacts["grid_mode"] == "fixed"
    reg = env.artifacts["region"]
    width = reg["end_s"] - reg["start_s"]
    assert width == pytest.approx(8 * CFG.FIXED_GRID_BARS_TO_SECONDS)  # +16 s
    assert width == pytest.approx(16.0)


# --------------------------------------------------------------------------- #
# compare_track — reference side only + the follow-up-call plan named in notes
# --------------------------------------------------------------------------- #


def test_compare_track_reference_side_and_next_calls(tmp_path, fs, monkeypatch):
    _patch_recon(monkeypatch, _recon())
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="compare_track",
        target="vocals",
        region=Window(start_s=1.0, end_s=4.0),
        cache=fs,
        engine=StubEngine(),
        verify=ScriptedGates(["pass"]),
        resolver=_resolver(_VOCALS_MAP),
        timeline=_mixed_timeline(),
    )
    assert "stems" in env.artifacts
    assert "descriptors" in env.artifacts
    joined = " ".join(env.notes)
    assert "render_track_region" in joined
    assert "characterize" in joined
    assert "compare_character" in joined


# --------------------------------------------------------------------------- #
# notes (T5) — basic-pitch source selection per §6 R6 (subprocess stubbed)
# --------------------------------------------------------------------------- #


class _StubTranscribe:
    def __init__(self, events):
        self.events = events
        self.calls: list[Path] = []

    def __call__(self, wav_path, out_dir):  # noqa: ARG002
        self.calls.append(Path(wav_path))
        return self.events


def test_notes_transcribes_the_passed_stem_at_t5(tmp_path, fs, monkeypatch):
    _patch_recon(monkeypatch, _recon())
    stub = _StubTranscribe([NoteEvent(0.0, 0.5, 60, 100, None)])
    env = analyze_reference(
        _wav(tmp_path),
        question_focus="notes",
        target="vocals",
        max_tier="T5",
        region=Window(start_s=1.0, end_s=4.0),
        cache=fs,
        engine=StubEngine(),
        verify=ScriptedGates(["pass"]),
        resolver=_resolver(_VOCALS_MAP),
        timeline=_mixed_timeline(),
        transcribe=stub,
    )
    assert env.tier_reached == "T5"
    assert "notes" in env.artifacts
    assert env.artifacts["notes"][0]["pitch_midi"] == 60
    # §6 R6: the ladder produced a passing vocals stem → THAT is the transcribe source.
    assert len(stub.calls) == 1
    assert stub.calls[0].name == "vocals.flac"
    assert any("transcribe source: rank1 stem" in n for n in env.notes)
