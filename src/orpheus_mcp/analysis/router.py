"""``analysis/router.py`` (Task 5.2, §6): the deterministic escalation ladder — the
project's centerpiece state machine. Given a target, a region, the T2 mix timeline, and the
verification gates, it decides *how far to climb* and returns the best artifact plus a full
transition log, driven only by gate results and the registry (§6: "how does it know?").

The ladder is **pure w.r.t. transcription**: R6 (the T5 last resort) only SELECTS and RECORDS
the source basic-pitch should run on — it never invokes transcription itself. The orchestrator
reads ``LadderOutcome.source`` + ``rung`` and drives T5.

Everything the ladder needs from the outside is injected, so the whole §6 semantics run in CI
with no GPU / no model weights (tests script the gates and count engine calls):

- ``registry``  — ``Callable[[str], list[ModelEntry]]``: the rank-ordered checkpoint candidates
  for a target (the orchestrator builds it from ``model_registry.resolve(task, target,
  vram_free_gb=…)``; an empty list means a **checkpoint-less** target → enter at R3).
- ``engines``   — ``Callable[[ModelEntry], SeparatorEngine]``: the engine to run an entry with.
- ``tagger``    — the T2 ``TaggerEngine`` (only the production verifier / re-tagging touches it).
- ``timeline``  — the song's T2 :class:`Timeline` (the absent/exposure pre-gates and the R5
  exposure fallback read it; no separation needed to answer those).
- ``verify``    — ``Callable[[stems, target, other_stem_path], GateSet]``: the §5 gate seam.
  ``None`` builds the production verifier (loads stem audio, runs ``verify.run_gates``); ladder
  tests inject a scripted double.
- ``clock``     — ``Callable[[], float]`` wall-clock for the §6 budget guard (tests fake it).

**§6 exhaustion semantics (locked):** if ``max_seconds`` expires mid-ladder the ladder returns
the best artifact so far + ``budget_capped=True`` and caches the rung it stopped at; a later
call with ``resume_from`` set continues *from that rung* (no re-run of the rungs below it). A
ladder that ran to COMPLETION caches ``completed=True`` and a re-ask returns the cached outcome
with zero engine calls — "re-asking never re-climbs" applies only to completed ladders, so a
capped first run can never freeze a low-quality answer forever.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from orpheus_mcp.analysis import cachefs as _cachefs
from orpheus_mcp.analysis.model_registry import classes_for_target
from orpheus_mcp.analysis.separation import separate_region
from orpheus_mcp.analysis.timeline import most_exposed_window, target_absent
from orpheus_mcp.analysis.verify import ensemble_avg_wave
from orpheus_mcp.models import GateSet, StemResult, Window

if TYPE_CHECKING:
    from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
    from orpheus_mcp.analysis.model_registry import ModelEntry
    from orpheus_mcp.analysis.separation import SeparatorEngine
    from orpheus_mcp.analysis.timeline import TaggerEngine
    from orpheus_mcp.models import Timeline

__all__ = ["TIER_ORDER", "QUESTION_TIERS", "LadderOutcome", "climb"]

# §3.1 cost tiers, in ascending order — the budget guard compares a rung's tier against the
# caller's ``max_tier`` by index into this list.
TIER_ORDER = ["T0", "T1", "T2", "T3", "T4", "T5"]

# §3.1 routing table: question class → the highest tier it is allowed to reach. The
# orchestrator uses it to pick ``max_tier``; the ladder consumes ``max_tier`` directly.
QUESTION_TIERS = {
    "recon": "T1",
    "timeline": "T2",
    "chords": "T2",
    "sound_at": "T3",
    "compare_track": "T3",
    "lyric_navigate": "T3",
    "notes": "T5",
}

# Every escalation rung is a T4-stage operation by definition (§6) except R6, the opt-in T5
# last resort. Pre-flight (absent/exposure) and rank-1 sit at/below T3 and are never
# tier-gated away.
_RUNG_TIER = {"R1": "T4", "R2": "T4", "R3": "T4", "R4": "T4", "R5": "T4", "R6": "T5"}

# The full ordered stage list the ladder walks. ``resume_from`` names the stage a capped
# ladder restarts at; pre-flight + rank1 sit ahead of the escalation rungs so a resume never
# repeats them.
_ORDER = ["absent", "exposure", "rank1", "R1", "R2", "R3", "R4", "R5", "R6"]

_GATE_FIELDS = ("mix_consistency", "bleed", "silence_honesty", "agreement")

Verifier = Callable[["dict[str, StemResult]", str, "str | None"], GateSet]
Resolver = Callable[[str], "list[ModelEntry]"]
EngineFactory = Callable[["ModelEntry"], "SeparatorEngine"]


@dataclass
class LadderOutcome:
    """The §6 ladder result. ``stems`` is the delivered artifact (``None`` for the pre-flight
    ``absent``/``exposed`` verdicts and the R5 mix-context fallback, which characterize the raw
    mix rather than a separated stem); ``rung`` records where the ladder stopped
    (``"absent"``|``"exposed"``|``"rank1"``|``"R1"``..``"R6"``); ``log`` is the ordered
    transition record (one ``"<rung>: <failing gate>=<value>"`` line per escalation);
    ``budget_capped`` is True when ``max_seconds`` expired mid-climb; ``source`` records what
    R5/R6/exposure characterized (the raw window or the transcription source)."""

    stems: dict[str, StemResult] | None
    rung: str
    log: list[str]
    budget_capped: bool
    source: str


@dataclass
class _Artifact:
    """A produced-and-gated artifact carried as best-so-far / R6-input candidate."""

    stems: dict[str, StemResult] | None
    rung: str
    source: str
    any_gate_passed: bool = False


@dataclass
class _State:
    """Mutable climb bookkeeping threaded through the stage walk."""

    log: list[str] = field(default_factory=list)
    separated: dict[str, dict[str, StemResult]] = field(default_factory=dict)
    best: _Artifact | None = None
    best_pass: _Artifact | None = None


# --------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------- #


def _passed_any(gs: GateSet) -> bool:
    return any(getattr(gs, f).verdict == "pass" for f in _GATE_FIELDS)


def _fail_desc(gs: GateSet) -> str:
    for f in _GATE_FIELDS:
        g = getattr(gs, f)
        if g.verdict == "fail":
            return f"{f}={g.value}"
    return f"overall={gs.overall}"


def _log_fail(rung: str, gs: GateSet) -> str:
    return f"{rung}: {_fail_desc(gs)}"


def _stamp(stems: dict[str, StemResult], target: str, gs: GateSet) -> dict[str, StemResult]:
    """Return ``stems`` with the target stem's ``gates`` field stamped (pydantic copy)."""
    out = dict(stems)
    out[target] = stems[target].model_copy(update={"gates": gs})
    return out


def _read_audio(path: str) -> np.ndarray:
    data, _sr = sf.read(path, always_2d=True, dtype="float32")
    return np.asarray(data, dtype=np.float32)


def _crop_audio(pcm: Pcm, window: Window) -> np.ndarray:
    sr = pcm.sample_rate
    samples = pcm.samples if pcm.samples.ndim == 2 else pcm.samples.reshape(-1, 1)
    start = max(0, int(round(window.start_s * sr)))
    end = min(samples.shape[0], int(round(window.end_s * sr)))
    return np.ascontiguousarray(samples[start:end], dtype=np.float32)


def _window_desc(win: Window | None, region: Window | None) -> str:
    if win is not None:
        return f"[{win.start_s:.3f},{win.end_s:.3f}] s"
    if region is not None:
        return f"[{region.start_s:.3f},{region.end_s:.3f}] s"
    return "full mix"


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def _ladder_kind(target: str, region: Window | None, max_tier: str) -> str:
    if region is None:
        span = "full"
    else:
        span = f"{round(region.start_s * 1000)}-{round(region.end_s * 1000)}"
    return f"ladder__{_safe(target)}__{span}__{max_tier}"


def _ladder_meta() -> dict[str, str]:
    return {
        "registry_version": _cachefs.registry_version(),
        "verify_config_hash": _cachefs.verify_config_hash(),
    }


def _stems_to_json(stems: dict[str, StemResult] | None) -> dict | None:
    if stems is None:
        return None
    return {t: sr.model_dump(mode="json") for t, sr in stems.items()}


def _stems_from_json(obj: object) -> dict[str, StemResult] | None:
    if not isinstance(obj, dict):
        return None
    return {t: StemResult.model_validate(v) for t, v in obj.items()}


def _outcome_from_dict(d: dict) -> LadderOutcome:
    return LadderOutcome(
        stems=_stems_from_json(d.get("stems")),
        rung=str(d.get("rung", "")),
        log=list(d.get("log", [])),
        budget_capped=bool(d.get("budget_capped", False)),
        source=str(d.get("source", "")),
    )


# --------------------------------------------------------------------------- #
# separation + ensemble + complement building blocks
# --------------------------------------------------------------------------- #


def _separate(
    pcm: Pcm, entry: ModelEntry, region: Window | None, engines: EngineFactory
) -> dict[str, StemResult]:
    return separate_region(pcm, entry, region, engine=engines(entry))


def _ensemble_target(
    pcm: Pcm,
    a: StemResult,
    b: StemResult,
    target: str,
    model_id: str,
) -> StemResult:
    """§6 R2: the ``avg_wave`` ensemble of two architectures' takes on ``target``, written to
    a fresh FLAC next to ``a``'s stem so the result is a first-class :class:`StemResult`."""
    avg = ensemble_avg_wave([_read_audio(a.stem_path), _read_audio(b.stem_path)])
    from pathlib import Path

    dest = Path(a.stem_path).with_name(f"{_safe(target)}__ensemble.flac")
    sf.write(str(dest), avg, a.sample_rate, subtype="PCM_24")
    return StemResult(
        stem_path=str(dest),
        model_id=model_id,
        target=target,
        region_trimmed=a.region_trimmed,
        sample_rate=a.sample_rate,
        gates=None,
        notes=["§5.4 avg_wave ensemble of two architectures"],
    )


def _complement_model(registry: Resolver) -> ModelEntry | None:
    """A multi-stem model whose strong stems (vocals/drums/bass) can be subtracted for the R3
    complement. ``bass``/``drums`` are only ever produced by multi-stem entries, so resolving
    either yields such a model without the ladder needing to know the task."""
    for probe in ("bass", "drums", "vocals"):
        cands = list(registry(probe))
        for entry in cands:
            if len(entry.targets) > 1:
                return entry
    return None


def _r3_complement(
    pcm: Pcm,
    target: str,
    region: Window | None,
    registry: Resolver,
    engines: EngineFactory,
    verify: Verifier,
    st: _State,
) -> tuple[str, dict[str, StemResult] | None, GateSet | None]:
    """§6 R3. Returns ``(status, stems, gates)`` where status is ``"skip"`` (precondition
    failed or no complement model — logged), ``"pass"``/``"fail"`` (residual gated)."""
    comp = _complement_model(registry)
    if comp is None:
        st.log.append("R3: skipped (no multi-stem complement model available)")
        return "skip", None, None

    strong = [t for t in comp.targets if t not in (target, "other")]
    if not strong:
        st.log.append("R3: skipped (complement model exposes no strong stems to subtract)")
        return "skip", None, None

    stems = _separate(pcm, comp, region, engines)

    # Precondition (§6): every stem being subtracted must individually pass gates 2-4. If the
    # ladder reached R3 because those stems are bad, subtracting them yields a worse residual.
    for t in strong:
        gs = verify(stems, t, None)
        if gs.overall != "pass":
            st.log.append(f"R3: precondition {t} {_fail_desc(gs)} — complement skipped")
            return "skip", None, None

    region_win = stems[strong[0]].region_trimmed
    mix_region = _crop_audio(pcm, region_win)
    residual = np.asarray(mix_region, dtype=np.float64)
    for t in strong:
        stem = _read_audio(stems[t].stem_path).astype(np.float64)
        n = min(residual.shape[0], stem.shape[0])
        residual = residual[:n] - stem[:n]

    from pathlib import Path

    dest = Path(stems[strong[0]].stem_path).with_name(f"{_safe(target)}__complement.flac")
    sf.write(str(dest), residual.astype(np.float32), pcm.sample_rate, subtype="PCM_24")
    res = StemResult(
        stem_path=str(dest),
        model_id=f"complement({comp.id} − {'+'.join(strong)})",
        target=target,
        region_trimmed=region_win,
        sample_rate=pcm.sample_rate,
        gates=None,
        # §6: the residual inherits the tails of everything subtracted — flag it so downstream
        # SourceMeta.complement is set and Claude discounts tail-sensitive descriptors.
        notes=["complement=True (§6 R3 residual)"],
    )
    out = {target: res}
    gs = verify(out, target, None)
    out = _stamp(out, target, gs)
    return ("pass" if gs.overall == "pass" else "fail"), out, gs


# --------------------------------------------------------------------------- #
# production verifier (default when verify is None) — loads stem audio, runs §5 gates
# --------------------------------------------------------------------------- #


def _production_verifier(
    pcm: Pcm,
    tagger: TaggerEngine,
    timeline: Timeline,
    candidates: Sequence[ModelEntry],
    target: str,
) -> Verifier:
    from orpheus_mcp.analysis.verify import run_gates

    entry = candidates[0] if candidates else None
    tclasses = classes_for_target(entry, target) if entry is not None else []

    def _verify(
        stems: dict[str, StemResult], tgt: str, other_stem_path: str | None
    ) -> GateSet:
        stem_audio = {t: _read_audio(sr.stem_path) for t, sr in stems.items()}
        region = stems[tgt].region_trimmed
        mix = _crop_audio(pcm, region)
        other = _read_audio(other_stem_path) if other_stem_path else None
        return run_gates(
            mix=mix,
            stems=stem_audio,
            target=tgt,
            sr=pcm.sample_rate,
            multi_stem=len(stems) > 1,
            tagger=tagger,
            target_tagger_classes=tclasses,
            mix_timeline=timeline,
            other_stem=other,
        )

    return _verify


# --------------------------------------------------------------------------- #
# climb — the ladder
# --------------------------------------------------------------------------- #


def climb(
    pcm: Pcm,
    target: str,
    region: Window | None,
    *,
    registry: Resolver,
    engines: EngineFactory,
    tagger: TaggerEngine,
    timeline: Timeline,
    max_tier: str,
    max_seconds: float,
    cache: CacheFS,
    resume_from: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    verify: Verifier | None = None,
) -> LadderOutcome:
    """Walk the §6 escalation ladder for ``target`` over ``region`` and return the best
    artifact + full transition log. See the module docstring for the injected seams and the
    exhaustion/resume semantics."""
    kind = _ladder_kind(target, region, max_tier)
    meta = _ladder_meta()

    # Completed ladders never re-climb (§6): a cached completion short-circuits every engine.
    if resume_from is None:
        cached = cache.read_json(pcm.pcm_hash, kind, meta)
        if cached is not None and cached.get("completed"):
            return _outcome_from_dict(cached)

    candidates = list(registry(target))
    if verify is None:
        verify = _production_verifier(pcm, tagger, timeline, candidates, target)

    target_classes = classes_for_target(candidates[0], target) if candidates else []

    st = _State()
    t0 = clock()
    max_i = TIER_ORDER.index(max_tier)

    def _complete(
        stems: dict[str, StemResult] | None, rung: str, source: str
    ) -> LadderOutcome:
        payload = {
            "completed": True,
            "budget_capped": False,
            "rung": rung,
            "log": st.log,
            "source": source,
            "stems": _stems_to_json(stems),
        }
        cache.write_json(pcm.pcm_hash, kind, payload, meta)
        return LadderOutcome(stems, rung, list(st.log), False, source)

    def _capped(blocked_stage: str) -> LadderOutcome:
        best = st.best
        rung = best.rung if best is not None else (resume_from or "rank1")
        source = best.source if best is not None else ""
        stems = best.stems if best is not None else None
        payload = {
            "completed": False,
            "budget_capped": True,
            "rung": rung,
            "resume_from": blocked_stage,
            "log": st.log,
            "source": source,
            "stems": _stems_to_json(stems),
        }
        cache.write_json(pcm.pcm_hash, kind, payload, meta)
        return LadderOutcome(stems, rung, list(st.log), True, source)

    def _record(stems: dict[str, StemResult], rung: str, source: str, gs: GateSet) -> None:
        art = _Artifact(stems=stems, rung=rung, source=source, any_gate_passed=_passed_any(gs))
        st.best = art
        if art.any_gate_passed:
            st.best_pass = art

    start_idx = _ORDER.index(resume_from) if resume_from in _ORDER else 0

    for stage in _ORDER[start_idx:]:
        # Budget guard (§6): the GPU-bearing rungs check wall-clock before spending. R5/R6 are
        # T1-T2-cost bookkeeping and never trip the cap.
        if stage in ("rank1", "R1", "R2", "R3", "R4") and (clock() - t0) >= max_seconds:
            return _capped(stage)
        # Tier cap: an escalation rung above ``max_tier`` is never run (R6 needs T5).
        if stage in _RUNG_TIER and TIER_ORDER.index(_RUNG_TIER[stage]) > max_i:
            break

        if stage == "absent":
            if target_classes and target_absent(timeline, target_classes):
                return _complete(None, "absent", f"target '{target}' not detected in this song")

        elif stage == "exposure":
            win = most_exposed_window(timeline, target_classes)
            if win is not None:
                return _complete(None, "exposed", f"exposed mix window {_window_desc(win, region)}")

        elif stage == "rank1":
            if not candidates:
                continue
            entry = candidates[0]
            stems = _separate(pcm, entry, region, engines)
            gs = verify(stems, target, None)
            stems = _stamp(stems, target, gs)
            st.separated[entry.id] = stems
            _record(stems, "rank1", entry.id, gs)
            if gs.overall in ("pass", "unverifiable"):
                return _complete(stems, "rank1", entry.id)
            st.log.append(_log_fail("rank1", gs))

        elif stage == "R1":
            if len(candidates) < 2:
                continue
            entry = candidates[1]
            stems = _separate(pcm, entry, region, engines)
            gs = verify(stems, target, None)
            stems = _stamp(stems, target, gs)
            st.separated[entry.id] = stems
            _record(stems, "R1", entry.id, gs)
            if gs.overall in ("pass", "unverifiable"):
                return _complete(stems, "R1", entry.id)
            st.log.append(_log_fail("R1", gs))

        elif stage == "R2":
            if len(candidates) < 2:
                continue
            e0, e1 = candidates[0], candidates[1]
            a = st.separated.get(e0.id, {}).get(target)
            b = st.separated.get(e1.id, {}).get(target)
            if a is None or b is None:
                continue
            model_id = f"{e0.id}+{e1.id} avg_wave"
            ens = {target: _ensemble_target(pcm, a, b, target, model_id)}
            gs = verify(ens, target, b.stem_path)
            ens = _stamp(ens, target, gs)
            _record(ens, "R2", model_id, gs)
            if gs.overall in ("pass", "unverifiable"):
                return _complete(ens, "R2", model_id)
            st.log.append(_log_fail("R2", gs))

        elif stage == "R3":
            status, r_stems, r_gs = _r3_complement(
                pcm, target, region, registry, engines, verify, st
            )
            if status == "skip":
                continue
            assert r_stems is not None and r_gs is not None
            _record(r_stems, "R3", r_stems[target].model_id, r_gs)
            if r_gs.overall in ("pass", "unverifiable"):
                return _complete(r_stems, "R3", r_stems[target].model_id)
            st.log.append(_log_fail("R3", r_gs))

        elif stage == "R4":
            if not candidates:
                continue
            entry = candidates[0]
            stems = _separate(pcm, entry, None, engines)  # widen → full song (§6 R4)
            gs = verify(stems, target, None)
            stems = _stamp(stems, target, gs)
            _record(stems, "R4", f"{entry.id} (full-song)", gs)
            if gs.overall in ("pass", "unverifiable"):
                return _complete(stems, "R4", f"{entry.id} (full-song)")
            st.log.append(_log_fail("R4", gs))

        elif stage == "R5":
            win = most_exposed_window(timeline, target_classes)
            source = f"mix-context: best exposed raw window {_window_desc(win, region)}"
            st.best = _Artifact(stems=None, rung="R5", source=source)
            st.log.append(f"R5: exposure fallback → {source}")

        elif stage == "R6":
            if st.best_pass is not None:
                src = (
                    f"transcribe-source: {st.best_pass.rung} artifact "
                    f"({st.best_pass.source}) — a gate passed on it"
                )
                return _complete(st.best_pass.stems, "R6", src)
            win = most_exposed_window(timeline, target_classes)
            src = f"transcribe-source: most-exposed raw window {_window_desc(win, region)}"
            return _complete(None, "R6", src)

    # Loop fell through (tier cap reached, or R5 set best and R6 was gated away): deliver the
    # best artifact the climb produced.
    if st.best is not None:
        return _complete(st.best.stems, st.best.rung, st.best.source)
    return _complete(None, resume_from or "rank1", "")
