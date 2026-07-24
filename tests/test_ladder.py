"""``analysis/router.py`` (Task 5.2, §6): the deterministic escalation ladder. Every §6
semantic is a named test here, all CI-safe — the separator engines are identity ``StubEngine``s
(no weights/GPU), the §5 gates are driven by ``ScriptedGates`` (predetermined pass/fail per
rung), and the budget clock is faked. ``free_vram_gb`` is pinned so nothing imports torch (the
same hermetic convention as ``test_separation.py``).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from orpheus_mcp.analysis import model_registry, router
from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
from orpheus_mcp.analysis.router import climb
from orpheus_mcp.models import Timeline, Window
from stub_engines import EngineHub, ScriptedGates

SR = 8000
RANK1 = "4stem-c/htdemucs-ft"  # vram 0, native_sr None → no resample, no VRAM fallback
RANK2 = "6stem-fallback/htdemucs-6s"  # vram 0, native_sr None; different stem set = "diff arch"

_REG = {e.id: e for e in model_registry.load_registry(allow_unpinned=True)}


@pytest.fixture(autouse=True)
def _plenty_of_vram(monkeypatch):
    monkeypatch.setattr(model_registry, "free_vram_gb", lambda: 24.0)


def _pcm(duration_s: float = 8.0) -> Pcm:
    n = int(round(duration_s * SR))
    ramp = (np.arange(n, dtype=np.float64) / n).astype(np.float32).reshape(-1, 1)
    return Pcm(
        samples=ramp,
        sample_rate=SR,
        path="<ramp>",
        pcm_hash="ladderhash",
        duration_s=n / SR,
        decoder="test",
    )


def _resolver(mapping):
    return lambda target: [_REG[i] for i in mapping.get(target, [])]


# Vocals is a checkpointed target; "bass" is included so R3's complement model resolves.
_VOCALS_MAP = {"vocals": [RANK1, RANK2], "bass": [RANK1], "drums": [RANK1]}
# "strings" has no rank-1 (checkpoint-less) but "bass" still yields a complement model.
_STRINGS_MAP = {"strings": [], "bass": [RANK1], "drums": [RANK1]}


def _timeline(frames, classes=("Singing", "Speech", "Guitar")):
    return Timeline(schema_version=1, classes=list(classes), hop_s=1.0, frames=frames)


def _mixed_timeline():
    """Vocals present but not exposed (Guitar bleeds equally) → pre-flight falls through."""
    return _timeline([[0.5, 0.0, 0.5]] * 5)


def _absent_timeline():
    """Vocal classes never exceed the activity threshold → §3.1c absent."""
    return _timeline([[0.1, 0.1, 0.5]] * 5)


def _exposed_timeline():
    """A 4 s window where Singing is loud and everything else is near-silent → §3.1b exposed."""
    return _timeline([[0.9, 0.0, 0.01]] * 4)


def _climb(pcm, target, timeline, hub, verify, cache, **kw):
    return climb(
        pcm,
        target,
        kw.pop("region", Window(start_s=1.0, end_s=4.0)),
        registry=_resolver(kw.pop("mapping", _VOCALS_MAP)),
        engines=hub,
        tagger=None,  # production verifier unused (verify is injected)
        timeline=timeline,
        max_tier=kw.pop("max_tier", "T4"),
        max_seconds=kw.pop("max_seconds", 1e9),
        cache=cache,
        resume_from=kw.pop("resume_from", None),
        verify=verify,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Pre-flight order: absent → exposure → rank-1 (§3.1a/b/c)
# --------------------------------------------------------------------------- #


def test_preflight_absent_target_returns_absent_without_separation(tmp_path):
    hub = ScriptedGates([])  # unused
    eng = EngineHub()
    out = _climb(_pcm(), "vocals", _absent_timeline(), eng, hub, CacheFS(tmp_path))
    assert out.rung == "absent"
    assert out.stems is None
    assert eng.total_calls == 0  # never separated


def test_preflight_exposure_returns_exposed_without_separation(tmp_path):
    eng = EngineHub()
    out = _climb(_pcm(), "vocals", _exposed_timeline(), eng, ScriptedGates([]), CacheFS(tmp_path))
    assert out.rung == "exposed"
    assert out.stems is None
    assert "exposed" in out.source
    assert eng.total_calls == 0


# --------------------------------------------------------------------------- #
# rank-1 pass, then the R1→R5 rung sequence
# --------------------------------------------------------------------------- #


def test_rank1_pass_stops_at_rank1(tmp_path):
    eng = EngineHub()
    out = _climb(
        _pcm(), "vocals", _mixed_timeline(), eng, ScriptedGates(["pass"]), CacheFS(tmp_path)
    )
    assert out.rung == "rank1"
    assert out.stems is not None and "vocals" in out.stems
    assert out.stems["vocals"].gates.overall == "pass"
    assert eng.calls_for(RANK1) == 1
    assert eng.calls_for(RANK2) == 0


def test_R1_runs_rank2_on_rank1_gate_fail(tmp_path):
    eng = EngineHub()
    out = _climb(
        _pcm(), "vocals", _mixed_timeline(), eng, ScriptedGates(["fail", "pass"]), CacheFS(tmp_path)
    )
    assert out.rung == "R1"
    assert eng.calls_for(RANK1) == 1 and eng.calls_for(RANK2) == 1


def test_R2_ensembles_without_re_separating(tmp_path):
    eng = EngineHub()
    out = _climb(
        _pcm(),
        "vocals",
        _mixed_timeline(),
        eng,
        ScriptedGates(["fail", "fail", "pass"]),
        CacheFS(tmp_path),
    )
    assert out.rung == "R2"
    assert "avg_wave" in out.stems["vocals"].model_id
    # R2 reuses rank-1 + R1 stems → only the two upstream separations ran.
    assert eng.total_calls == 2


def test_every_transition_logs_failing_gate(tmp_path):
    eng = EngineHub()
    out = _climb(
        _pcm(),
        "vocals",
        _mixed_timeline(),
        eng,
        ScriptedGates(["fail", "fail", "pass"]),
        CacheFS(tmp_path),
    )
    assert out.log[0] == "rank1: bleed=0.42"
    assert out.log[1] == "R1: bleed=0.42"
    for line in out.log:
        assert re.match(r"^(rank1|R\d): ", line)


# --------------------------------------------------------------------------- #
# R3 complement + precondition (§6)
# --------------------------------------------------------------------------- #


def test_R3_complement_flags_residual_when_precondition_passes(tmp_path):
    eng = EngineHub()
    # checkpoint-less "strings" enters at R3; strong stems = vocals/drums/bass (3 pass) + residual.
    out = _climb(
        _pcm(),
        "strings",
        _mixed_timeline(),
        eng,
        ScriptedGates(["pass", "pass", "pass", "pass"]),
        CacheFS(tmp_path),
        mapping=_STRINGS_MAP,
    )
    assert out.rung == "R3"
    res = out.stems["strings"]
    assert any("complement=True" in n for n in res.notes)
    assert res.gates.overall == "pass"


def test_R3_skipped_when_subtracted_stems_fail(tmp_path):
    eng = EngineHub()
    # First strong stem fails its precondition → R3 skipped, ladder falls through to R5.
    out = _climb(
        _pcm(),
        "strings",
        _mixed_timeline(),
        eng,
        ScriptedGates(["fail"]),
        CacheFS(tmp_path),
        mapping=_STRINGS_MAP,
    )
    assert out.rung == "R5"
    assert any("precondition" in line and "complement skipped" in line for line in out.log)


def test_checkpointless_target_skips_R1_R2(tmp_path):
    eng = EngineHub()
    out = _climb(
        _pcm(),
        "strings",
        _mixed_timeline(),
        eng,
        ScriptedGates(["pass", "pass", "pass", "pass"]),
        CacheFS(tmp_path),
        mapping=_STRINGS_MAP,
    )
    # No rank-1/rank-2 model for "strings": only the R3 complement model ever separated.
    assert out.rung == "R3"
    assert eng.calls_for(RANK1) == 1  # the complement model (bass→htdemucs-ft), once
    assert eng.calls_for(RANK2) == 0


# --------------------------------------------------------------------------- #
# R4 widen, R5 exposure fallback, R6 gate
# --------------------------------------------------------------------------- #


def test_R4_widens_to_full_song(tmp_path):
    eng = EngineHub()
    # rank1 fail, R1 fail, R2 fail, R3 precondition fail (skip), R4 pass.
    out = _climb(
        _pcm(),
        "vocals",
        _mixed_timeline(),
        eng,
        ScriptedGates(["fail", "fail", "fail", "fail", "pass"]),
        CacheFS(tmp_path),
    )
    assert out.rung == "R4"
    assert "full-song" in out.source
    assert out.stems is not None and "vocals" in out.stems


def test_R5_exposure_fallback_at_default_tier(tmp_path):
    eng = EngineHub()
    out = _climb(
        _pcm(),
        "vocals",
        _mixed_timeline(),
        eng,
        ScriptedGates(["fail", "fail", "fail", "fail", "fail"]),
        CacheFS(tmp_path),
    )
    assert out.rung == "R5"
    assert out.stems is None
    assert "mix-context" in out.source
    assert out.budget_capped is False


def test_R6_only_fires_when_max_tier_is_T5(tmp_path):
    script = ["fail", "fail", "fail", "fail", "fail"]
    at_t4 = _climb(
        _pcm(), "vocals", _mixed_timeline(), EngineHub(), ScriptedGates(list(script)),
        CacheFS(tmp_path / "a"),
    )
    assert at_t4.rung == "R5"

    at_t5 = _climb(
        _pcm(), "vocals", _mixed_timeline(), EngineHub(), ScriptedGates(list(script)),
        CacheFS(tmp_path / "b"), max_tier="T5",
    )
    assert at_t5.rung == "R6"
    assert "transcribe-source" in at_t5.source


# --------------------------------------------------------------------------- #
# Unverifiable → deliver flagged, do NOT climb (§5 decision rule)
# --------------------------------------------------------------------------- #


def test_unverifiable_delivers_without_climbing(tmp_path):
    eng = EngineHub()
    out = _climb(
        _pcm(), "vocals", _mixed_timeline(), eng, ScriptedGates(["unverifiable"]), CacheFS(tmp_path)
    )
    assert out.rung == "rank1"
    assert out.stems["vocals"].gates.overall == "unverifiable"
    assert out.log == []  # no transition — climbing cannot fix unverifiability
    assert eng.total_calls == 1  # rank-1 only; R1 never ran


# --------------------------------------------------------------------------- #
# Budget: cap mid-ladder, resume from the recorded rung, never re-climb a completed ladder
# --------------------------------------------------------------------------- #


class _FakeClock:
    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def __call__(self):
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


def test_budget_cap_records_rung_and_resume_continues(tmp_path):
    cache = CacheFS(tmp_path)
    pcm = _pcm()
    # t0=0; rank1@10, R1@20, R2@30 all under 100; R3 check @200 → cap.
    clock = _FakeClock([0, 10, 20, 30, 200])
    eng1 = EngineHub()
    first = _climb(
        pcm, "vocals", _mixed_timeline(), eng1,
        ScriptedGates(["fail", "fail", "fail"]), cache, max_seconds=100, clock=clock,
    )
    assert first.budget_capped is True
    assert first.rung == "R2"  # best-so-far

    kind = router._ladder_kind("vocals", Window(start_s=1.0, end_s=4.0), "T4")
    cached = cache.read_json(pcm.pcm_hash, kind, router._ladder_meta())
    assert cached["completed"] is False
    assert cached["resume_from"] == "R3"

    # Resume with a bigger budget: starts at R3, never re-runs rank-1/R1/R2.
    eng2 = EngineHub()
    second = _climb(
        pcm, "vocals", _mixed_timeline(), eng2,
        ScriptedGates(["pass", "pass", "pass"]), cache, resume_from="R3",
    )
    assert second.rung == "R3"
    assert eng2.calls_for(RANK2) == 0  # R1 never re-ran
    assert eng2.total_calls == 1  # only the R3 complement separation


def test_completed_ladder_never_reclimbs(tmp_path):
    cache = CacheFS(tmp_path)
    pcm = _pcm()
    eng1 = EngineHub()
    first = _climb(pcm, "vocals", _mixed_timeline(), eng1, ScriptedGates(["pass"]), cache)
    assert first.rung == "rank1" and eng1.total_calls == 1

    eng2 = EngineHub()
    second = _climb(pcm, "vocals", _mixed_timeline(), eng2, ScriptedGates(["pass"]), cache)
    assert second.rung == "rank1"
    assert eng2.total_calls == 0  # served from cache, zero engine calls
