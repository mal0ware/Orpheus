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
- ``verify``    — ``verify(stems, target, other_stem_path, *, entry)``: the §5 gate seam. The
  keyword ``entry`` is the ModelEntry that PRODUCED ``stems`` (D13: classes are resolved
  per-call from *that* entry, so a gate never judges one target's stems against another
  target's tagger classes). ``None`` builds the production verifier (loads stem audio, crops a
  per-call region timeline, runs ``verify.run_gates``); ladder tests inject a scripted double.
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
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import soundfile as sf

from orpheus_mcp.analysis import cachefs as _cachefs
from orpheus_mcp.analysis import model_registry
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.model_registry import classes_for_target
from orpheus_mcp.analysis.separation import SeparationError, SeparatorEnvMissing, separate_region
from orpheus_mcp.analysis.timeline import crop_timeline, most_exposed_window, target_absent
from orpheus_mcp.analysis.verify import ensemble_avg_wave
from orpheus_mcp.models import AnswerEnvelope, GateSet, Recon, StemHandle, StemResult, Window

if TYPE_CHECKING:
    from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
    from orpheus_mcp.analysis.model_registry import ModelEntry
    from orpheus_mcp.analysis.separation import SeparatorEngine
    from orpheus_mcp.analysis.timeline import TaggerEngine
    from orpheus_mcp.models import Timeline

__all__ = [
    "TIER_ORDER",
    "QUESTION_TIERS",
    "LadderOutcome",
    "climb",
    "analyze_reference",
    "QUESTION_CLASSES",
]

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

Resolver = Callable[[str], "list[ModelEntry]"]
EngineFactory = Callable[["ModelEntry"], "SeparatorEngine"]


class Verifier(Protocol):
    """The §5 gate seam. ``entry`` is the registry entry that PRODUCED ``stems`` — each call
    resolves ``classes_for_target(entry, target)`` itself (D13: never reuse one target's
    tagger classes to judge another target's stems), or ``[]`` when ``entry`` is ``None`` (a
    synthetic R3 residual / checkpoint-less target with no tagger mapping). ``None`` at the
    ``climb`` boundary builds the production verifier; ladder tests inject a scripted double."""

    def __call__(
        self,
        stems: dict[str, StemResult],
        target: str,
        other_stem_path: str | None = None,
        *,
        entry: ModelEntry | None,
    ) -> GateSet: ...


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


def _ladder_meta() -> dict[str, object]:
    return {
        "registry_version": _cachefs.registry_version(),
        "verify_config_hash": _cachefs.verify_config_hash(),
        "schema_version": 1,
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
    target_entry: ModelEntry | None,
    region: Window | None,
    registry: Resolver,
    engines: EngineFactory,
    verify: Verifier,
    st: _State,
) -> tuple[str, dict[str, StemResult] | None, GateSet | None]:
    """§6 R3. Returns ``(status, stems, gates)`` where status is ``"skip"`` (precondition
    failed or no complement model — logged), ``"pass"``/``"fail"`` (residual gated).

    ``target_entry`` is the target's own rank-1 registry entry (or ``None`` for a
    checkpoint-less target) — the residual has no single producer, so its bleed gate resolves
    the *target's* tagger classes from that entry, never the complement model's."""
    comp = _complement_model(registry)
    if comp is None:
        st.log.append("R3: skipped (no multi-stem complement model available)")
        return "skip", None, None

    strong = [t for t in comp.targets if t not in (target, "other")]
    if not strong:
        st.log.append("R3: skipped (complement model exposes no strong stems to subtract)")
        return "skip", None, None

    stems = _separate(pcm, comp, region, engines)

    # Precondition (§6, locked): every subtracted stem must individually clear gates 2-4
    # (bleed / silence_honesty / agreement) — mix_consistency (gate 1) is NOT part of this
    # gate: it judges the whole co-produced set, not one stem's subtraction-safety. Each stem
    # is verified against ITS OWN tagger classes (D13), resolved from the complement entry.
    _gates_2_4 = ("bleed", "silence_honesty", "agreement")
    for t in strong:
        gs = verify(stems, t, None, entry=comp)
        failing = next((f for f in _gates_2_4 if getattr(gs, f).verdict == "fail"), None)
        if failing is not None:
            val = getattr(gs, failing).value
            st.log.append(f"R3: precondition {t} {failing}={val} — complement skipped")
            return "skip", None, None

    region_win = stems[strong[0]].region_trimmed
    mix_region = _crop_audio(pcm, region_win)
    residual = np.asarray(mix_region, dtype=np.float64)
    for t in strong:
        stem = _read_audio(stems[t].stem_path).astype(np.float64)
        n = min(residual.shape[0], stem.shape[0])
        residual = residual[:n] - stem[:n]

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
    gs = verify(out, target, None, entry=target_entry)
    out = _stamp(out, target, gs)
    return ("pass" if gs.overall == "pass" else "fail"), out, gs


# --------------------------------------------------------------------------- #
# production verifier (default when verify is None) — loads stem audio, runs §5 gates
# --------------------------------------------------------------------------- #


def _production_verifier(
    pcm: Pcm,
    tagger: TaggerEngine,
    timeline: Timeline,
) -> Verifier:
    """Build the default ``verify`` seam. Two wirings the caller must get right per call, not
    once (each was a cross-target/whole-song blindness bug otherwise):

    - **classes** are resolved from the PRODUCING ``entry`` for *this* call's ``tgt`` (D13) —
      R3 verifies drums/bass against the complement model's classes, not the original
      target's; a synthetic residual (``entry is None``) has no tagger mapping → ``[]``.
    - **timeline** is cropped to *this* stem's region and re-indexed (frame 0 == stem sample
      0) for gate 3, on a fresh COPY — the full-song ``timeline`` is untouched so the ladder's
      absent/exposure pre-gates keep the song-wide view."""
    from orpheus_mcp.analysis.verify import run_gates

    def _verify(
        stems: dict[str, StemResult],
        target: str,
        other_stem_path: str | None = None,
        *,
        entry: ModelEntry | None,
    ) -> GateSet:
        stem_audio = {t: _read_audio(sr.stem_path) for t, sr in stems.items()}
        region = stems[target].region_trimmed
        mix = _crop_audio(pcm, region)
        other = _read_audio(other_stem_path) if other_stem_path else None
        tclasses = classes_for_target(entry, target) if entry is not None else []
        return run_gates(
            mix=mix,
            stems=stem_audio,
            target=target,
            sr=pcm.sample_rate,
            multi_stem=len(stems) > 1,
            tagger=tagger,
            target_tagger_classes=tclasses,
            mix_timeline=crop_timeline(timeline, region),
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
    tagger: TaggerEngine | None,
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
        if tagger is None:
            raise ValueError("climb needs a `tagger` when no `verify` seam is injected.")
        verify = _production_verifier(pcm, tagger, timeline)

    # The target's own rank-1 entry (``None`` for a checkpoint-less target) — the seam through
    # which every verify call that judges the *target* (rank1/R1/R2/R3-residual/R4) resolves
    # the target's tagger classes. ``target_classes`` (pre-flight absent/exposure) reuses it.
    target_entry = candidates[0] if candidates else None
    target_classes = classes_for_target(target_entry, target) if target_entry else []

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
            gs = verify(stems, target, None, entry=entry)
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
            gs = verify(stems, target, None, entry=entry)
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
            # Both parents (e0, e1) produce the SAME target stem class set, so the ensemble's
            # gate classes come from e0 (rank-1) — either parent's entry resolves identically.
            gs = verify(ens, target, b.stem_path, entry=e0)
            ens = _stamp(ens, target, gs)
            _record(ens, "R2", model_id, gs)
            if gs.overall in ("pass", "unverifiable"):
                return _complete(ens, "R2", model_id)
            st.log.append(_log_fail("R2", gs))

        elif stage == "R3":
            status, r_stems, r_gs = _r3_complement(
                pcm, target, target_entry, region, registry, engines, verify, st
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
            gs = verify(stems, target, None, entry=entry)
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


# --------------------------------------------------------------------------- #
# analyze_reference — the §9 orchestrator (Task 12.1)
#
# The convenience entry that applies the §3.1 routing table and packages the §9 answer
# envelope. It composes the *cache-plumbing tool functions* (``tools.reference``: recon /
# timeline / chords / characterize / lyrics) plus this module's own :func:`climb` — imported
# LAZILY inside the function so ``analysis/router.py`` carries no import-time dependency on the
# tools layer and, crucially, NONE on the REAPER bridge (§3 module boundary: ``compare_track``
# delivers the reference side only and merely NAMES the render/characterize/compare follow-up
# calls; it never reaches for the bridge). ``analyze_reference`` is therefore the single place
# where a routing decision is made, every stage's cache hit/miss and the §6 ladder log are
# collected, and every degradation is named (``capped_by_missing_extra`` / ``budget_capped``).
# --------------------------------------------------------------------------- #

# The locked §9 question-class vocabulary (router + envelope + tests share it). ``None`` focus
# ⇒ ``"recon"``; anything outside this set is a validation error.
QUESTION_CLASSES = tuple(QUESTION_TIERS.keys())

# Which §3.1 cost tier a finished ladder rung actually COST — so ``tier_reached`` reports the
# honest tier, not the question's ceiling. Pre-flight verdicts (absent/exposed) are T2
# bookkeeping; rank-1 is the single T3 separation; every escalation rung is T4; R6 is T5.
_RUNG_TO_TIER = {
    "absent": "T2",
    "exposed": "T2",
    "rank1": "T3",
    "R1": "T4",
    "R2": "T4",
    "R3": "T4",
    "R4": "T4",
    "R5": "T4",
    "R6": "T5",
}


class _RecordingCache:
    """A transparent :class:`CacheFS` proxy that records every ``read_json`` as a
    ``(kind, hit)`` pair, so the orchestrator can put each stage's cache hit/miss on the
    envelope (§9: "every stage's cache hits/misses ride the envelope") without any tool
    function needing to report it. Every other method/attribute delegates to the wrapped
    instance, so the real cache root, stem layout, and pinning are untouched."""

    def __init__(self, inner: CacheFS) -> None:
        self._inner = inner
        self.reads: list[tuple[str, bool]] = []

    def read_json(self, pcm_hash: str, kind: str, meta: dict) -> dict | None:
        obj = self._inner.read_json(pcm_hash, kind, meta)
        self.reads.append((kind, obj is not None))
        return obj

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _tier_i(tier: str) -> int:
    return TIER_ORDER.index(tier)


def _class_ceiling(question_class: str) -> str:
    """The tier ceiling the routing table permits a class to REACH. This is
    ``QUESTION_TIERS[class]`` for every class except ``sound_at``/``compare_track``, which the
    spec (§3.1: "T3 (+T4 only on gate failure)") allows to escalate one tier into the §6 ladder's
    T4 rungs — ``QUESTION_TIERS`` records their T3 *nominal/entry* tier, and this adds the
    gate-failure escalation allowance so the ladder can actually climb. ``tier_reached`` still
    reports the HONEST tier the ladder stopped at (T3 when rank-1 passed, T4 only if it climbed)."""
    if question_class in ("sound_at", "compare_track"):
        return "T4"
    return QUESTION_TIERS[question_class]


def _cap_tier(question_class: str, max_tier: str | None) -> str:
    """min(class ceiling, max_tier) by TIER_ORDER index (§3.1 routing table)."""
    ceiling = _class_ceiling(question_class)
    if max_tier is None:
        return ceiling
    if max_tier not in TIER_ORDER:
        raise ValueError(f"max_tier must be one of {TIER_ORDER}, got {max_tier!r}")
    return ceiling if _tier_i(ceiling) <= _tier_i(max_tier) else max_tier


def _recon_summary(recon: Recon) -> dict[str, Any]:
    """The compact T1 fact bundle present on every envelope (§9 ``recon_summary``)."""
    g = recon.grid
    return {
        "duration_s": recon.duration_s,
        "tempo_bpm": recon.tempo_bpm,
        "key": recon.key.root,
        "mode": recon.key.mode,
        "key_confidence": recon.key.confidence,
        "lufs_integrated": recon.lufs_integrated,
        "tuning_offset_cents": recon.tuning_offset_cents,
        "meter": list(g.meter),
        "grid_kind": g.kind,
        "grid_confidence": g.confidence,
        "n_sections": len(recon.sections),
    }


def _bar_seconds(recon: Recon) -> tuple[float, str]:
    """The §14.2/§13.4 bar length for the ``lyric_navigate`` "+8 bars" window: the measured
    recon downbeat spacing on a confident beat grid, or ``CFG.FIXED_GRID_BARS_TO_SECONDS`` per
    bar when the grid is fixed (``grid.kind == "fixed"`` — or a low-confidence / too-few-
    downbeats beat grid that downstream consumers treat as fixed, §13.4 rubato fallback).
    Returns ``(seconds_per_bar, mode)``."""
    g = recon.grid
    fixed = (
        g.kind == "fixed"
        or g.confidence < CFG.BEAT_GRID_CONFIDENCE_MIN
        or len(g.downbeats_s) < 2
    )
    if not fixed:
        diffs = [b - a for a, b in zip(g.downbeats_s, g.downbeats_s[1:], strict=False) if b > a]
        if diffs:
            return statistics.median(diffs), "beat"
    return CFG.FIXED_GRID_BARS_TO_SECONDS, "fixed"


def _mean_activation(tl: Timeline, win: Window) -> dict[str, float]:
    """Per-class mean activation of ``tl`` over ``win`` (§3.1 timeline diff for lyric_navigate)."""
    cropped = crop_timeline(tl, win)
    if not cropped.frames:
        return {cls: 0.0 for cls in cropped.classes}
    arr = np.asarray(cropped.frames, dtype=np.float64)
    means = arr.mean(axis=0)
    return {cls: float(m) for cls, m in zip(cropped.classes, means, strict=True)}


def _stem_handle(sr: StemResult) -> StemHandle:
    """A :class:`StemHandle` for ``characterize`` built from a ladder-produced stem — the R3
    ``complement=True`` note rides through ``notes`` so ``resolve_source`` stamps
    ``SourceMeta.complement`` and the envelope's descriptor carries it end-to-end (§6/§7)."""
    return StemHandle(
        stem_path=sr.stem_path,
        stem_type=sr.target,
        produced_by=sr.model_id,
        region=sr.region_trimmed,
        notes=list(sr.notes),
    )


def analyze_reference(
    file: str,
    question_focus: str | None = None,
    region: Window | None = None,
    max_tier: str | None = None,
    max_seconds: float | None = None,
    target: str | None = None,
    quote: str | None = None,
    occurrence: int | None = None,
    *,
    cache: CacheFS | None = None,
    tagger: TaggerEngine | None = None,
    engine: SeparatorEngine | None = None,
    asr: Any | None = None,
    verify: Verifier | None = None,
    resolver: Resolver | None = None,
    timeline: Timeline | None = None,
    transcribe: Callable[[Path, Path], list[Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AnswerEnvelope:
    """§9 orchestrator. Route ``question_focus`` (``None`` ⇒ ``"recon"``) through the §3.1
    tier table, run stages up to ``min(QUESTION_TIERS[class], max_tier)``, populate ONLY the
    artifact groups that class needs, and return the §9 :class:`AnswerEnvelope`.

    Locked behavior: an unknown ``question_focus`` raises ``ValueError``; a missing extra at the
    needed tier degrades to the highest available tier with ``capped_by_missing_extra`` named;
    ``max_seconds`` expiring mid-ladder rides as ``budget_capped``; every stage's cache hit/miss
    and the §6 ladder log ride the envelope. Two locked closures (see the module note):
    ``lyric_navigate`` answers with the timeline diff of ``[match.end_s, +8 bars]`` vs the
    preceding window; ``compare_track`` delivers the REFERENCE side only and NAMES the
    render/characterize/compare follow-up calls (``analysis/`` stays bridge-free).

    All keyword-only params after ``cache`` are hermetic-test injection seams (stub tagger/
    engine/asr, a scripted ``verify``/``resolver``/``timeline``, a stubbed ``transcribe``
    subprocess, a fake ``clock``); production callers leave them ``None``."""
    from orpheus_mcp.tools import reference as _ref

    question_class = question_focus if question_focus is not None else "recon"
    if question_class not in QUESTION_TIERS:
        raise ValueError(
            f"question_focus must be one of {list(QUESTION_TIERS)} (or None ⇒ 'recon'), "
            f"got {question_focus!r}"
        )
    tier_cap = _cap_tier(question_class, max_tier)
    budget = max_seconds if max_seconds is not None else CFG.DEFAULT_MAX_SECONDS

    inner = cache if cache is not None else _ref.CacheFS()
    rec = _RecordingCache(inner)
    notes: list[str] = []
    artifacts: dict[str, Any] = {}
    gates: dict[str, GateSet] = {}
    ladder_log: list[str] = []
    capped_by_missing_extra: str | None = None
    budget_capped = False

    # T1 recon — present on every envelope. Cast the recording proxy to CacheFS for the tool
    # signatures (it delegates transparently); this is the one intentional structural-duck cast.
    fs: CacheFS = rec  # type: ignore[assignment]
    recon = _ref.recon_reference(file, cache=fs)
    tier_reached = "T1"

    def _bump(tier: str) -> None:
        nonlocal tier_reached
        if _tier_i(tier) > _tier_i(tier_reached):
            tier_reached = tier

    def _get_timeline() -> Timeline:
        if timeline is not None:
            return timeline
        return _ref.instrument_timeline(file, cache=fs, tagger=tagger)

    def _finish() -> AnswerEnvelope:
        for kind, hit in rec.reads:
            notes.append(f"cache: {kind} {'HIT' if hit else 'MISS'}")
        return AnswerEnvelope(
            question_class=question_class,
            tier_reached=tier_reached,
            recon_summary=_recon_summary(recon),
            artifacts=artifacts,
            gates=gates,
            ladder_log=ladder_log,
            budget_capped=budget_capped,
            capped_by_missing_extra=capped_by_missing_extra,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # T1: recon — the answer IS the recon summary; no artifact groups.
    # ------------------------------------------------------------------ #
    if question_class == "recon":
        return _finish()

    # ------------------------------------------------------------------ #
    # T2: timeline / chords — populate their one artifact group only.
    # ------------------------------------------------------------------ #
    if question_class == "timeline":
        try:
            tl = _get_timeline()
        except (_ref.ReferenceEnvMissing, ImportError) as exc:
            capped_by_missing_extra = "analysis"
            notes.append(f"instrument timeline unavailable ([analysis] extra): {exc}")
            return _finish()
        artifacts["timeline"] = tl.model_dump(mode="json")
        _bump("T2")
        return _finish()

    if question_class == "chords":
        lane = _ref.get_chords(file, region, cache=fs)
        artifacts["chords"] = lane.model_dump(mode="json")
        _bump("T2")
        return _finish()

    # ------------------------------------------------------------------ #
    # lyric_navigate: cached lyric index → [match.end_s, +8 bars] window →
    # timeline diff vs the preceding window (§14.2). T1–T2, no separation.
    # ------------------------------------------------------------------ #
    if question_class == "lyric_navigate":
        if not quote:
            raise ValueError("question_focus='lyric_navigate' requires a lyric `quote`.")
        located = _ref.locate_lyric(
            file, quote, occurrence, cache=fs, asr=asr, tagger=tagger, engine=engine
        )
        if located.status == "no_vocals" or not located.matches:
            notes.append("no lyric match for the quote — nothing to navigate from.")
            return _finish()
        match = located.matches[0]
        bar_s, grid_mode = _bar_seconds(recon)
        width = 8.0 * bar_s
        dur = recon.duration_s
        r_start = max(0.0, min(match.end_s, dur))
        r_end = max(r_start, min(match.end_s + width, dur))
        p_start = max(0.0, r_start - width)
        p_end = r_start
        region_win = Window(start_s=r_start, end_s=r_end)
        preceding_win = Window(start_s=p_start, end_s=p_end)
        tl = _get_timeline()
        after = _mean_activation(tl, region_win)
        before = _mean_activation(tl, preceding_win)
        diff = {cls: round(after[cls] - before.get(cls, 0.0), 6) for cls in after}
        rising = sorted((c for c, d in diff.items() if d > 1e-6), key=lambda c: -diff[c])
        artifacts["timeline"] = tl.model_dump(mode="json")
        artifacts["region"] = region_win.model_dump(mode="json")
        artifacts["preceding_region"] = preceding_win.model_dump(mode="json")
        artifacts["lyric_match"] = match.model_dump(mode="json")
        artifacts["bar_seconds"] = bar_s
        artifacts["grid_mode"] = grid_mode
        artifacts["activation_diff"] = diff
        artifacts["rising"] = rising
        _bump("T2")
        notes.append(
            f"lyric window [{r_start:.3f},{r_end:.3f}] s = match.end_s + 8 bars "
            f"({grid_mode} grid, {bar_s:.3f} s/bar)"
        )
        return _finish()

    # ------------------------------------------------------------------ #
    # sound_at / compare_track / notes: the §6 separation ladder path.
    # ------------------------------------------------------------------ #
    if target is None:
        if question_class in ("sound_at", "compare_track"):
            raise ValueError(f"question_focus={question_class!r} requires a `target`.")
        target = ""  # notes: whole-mix transcription (no per-instrument attribution, §9 T5)

    pcm = _ref.cachefs.decode(file)

    # Resolve the T2 tagger up front (a missing [analysis] extra degrades to the tier we have).
    # It builds the timeline AND — when no ``verify`` seam is injected — the §6 production
    # verifier inside ``climb``, so both share ONE tagger instance rather than loading two. Only
    # resolved when actually needed (a hermetic test that injects both a ``timeline`` and a
    # ``verify`` seam never constructs a real tagger).
    tagger_engine: TaggerEngine | None = tagger
    if tagger_engine is None and (verify is None or timeline is None):
        try:
            tagger_engine = _ref._resolve_tagger(None)
        except (_ref.ReferenceEnvMissing, ImportError) as exc:
            capped_by_missing_extra = "analysis"
            notes.append(f"tagger unavailable ([analysis] extra): {exc}")
            return _finish()

    # Build the ladder timeline (T2) — needed for absent/exposure pre-gates and the verifier.
    try:
        tl = timeline if timeline is not None else _ref.instrument_timeline(
            file, cache=fs, tagger=tagger_engine
        )
        _bump("T2")
    except (_ref.ReferenceEnvMissing, ImportError) as exc:
        capped_by_missing_extra = "analysis"
        notes.append(f"instrument timeline unavailable ([analysis] extra): {exc}")
        return _finish()

    free = model_registry.free_vram_gb()
    the_resolver: Resolver = (
        resolver if resolver is not None else _ref._separator_registry_resolver(free)
    )
    engines = _ref._engine_factory(engine)

    # The ladder climbs up to ``tier_cap`` (already min(class ceiling, max_tier)); rank-1 is T3,
    # every escalation rung T4, R6 T5. For "notes" the ladder may reach R6's source-selection
    # (T5); the actual basic-pitch transcription is the T5 step run AFTER the ladder, below.
    outcome: LadderOutcome | None = None
    if target:
        try:
            outcome = climb(
                pcm,
                target,
                region,
                registry=the_resolver,
                engines=engines,
                tagger=tagger_engine,
                timeline=tl,
                max_tier=tier_cap,
                max_seconds=budget,
                cache=fs,
                clock=clock,
                verify=verify,
            )
        except (SeparatorEnvMissing, SeparationError) as exc:
            capped_by_missing_extra = "separation"
            notes.append(f"separation unavailable ([separation] extra): {exc}")
            return _finish()

        ladder_log = list(outcome.log)
        budget_capped = outcome.budget_capped
        _bump(_RUNG_TO_TIER.get(outcome.rung, tier_reached))

        if outcome.rung == "absent":
            notes.append(f"target not detected in this song ({target}). {outcome.source}")
            return _finish()

    # notes (T5): transcribe the best available source (§6 R6 selection). Runs only when the
    # tier ceiling actually allows T5 — otherwise we deliver the stems we have and say so.
    if question_class == "notes":
        if _tier_i(tier_cap) < _tier_i("T5"):
            notes.append(f"notes require T5; capped at {tier_cap} — delivering stems only.")
            _package_stem(outcome, artifacts, gates, notes)
            return _finish()
        events, src_desc = _run_transcription(
            file, region, pcm, tl, target, outcome, the_resolver, fs, transcribe
        )
        artifacts["notes"] = [
            {
                "start_time_s": e.start_time_s,
                "end_time_s": e.end_time_s,
                "pitch_midi": e.pitch_midi,
                "velocity": e.velocity,
                "pitch_bend": e.pitch_bend,
            }
            for e in events
        ]
        notes.append(f"transcribe source: {src_desc}")
        _bump("T5")
        return _finish()

    # sound_at / compare_track. §3.1b: when the ladder's pre-flight found the target EXPOSED in
    # the raw mix (rung "exposed"), skip separation and characterize that mix window directly —
    # a T2 answer that still delivers descriptors (the spec's "characterize the mix region
    # directly"). Otherwise package the separated reference stem + gates + descriptors.
    if outcome is not None and outcome.stems is None and outcome.rung == "exposed":
        cands = list(the_resolver(target))
        tclasses = classes_for_target(cands[0], target) if cands else []
        win = most_exposed_window(tl, tclasses) if tclasses else None
        if win is not None:
            character = _ref.characterize(file, region=win, cache=fs)
            artifacts["descriptors"] = [character.model_dump(mode="json")]
            artifacts["exposed_window"] = win.model_dump(mode="json")
            notes.append(
                f"§3.1b: '{target}' is exposed in the raw mix over "
                f"[{win.start_s:.3f},{win.end_s:.3f}] s — characterized the mix window "
                f"directly, no separation needed (no per-stem gates apply)."
            )
        else:
            notes.append(f"answered at rung exposed: {outcome.source} (no stem to describe).")
    else:
        _package_stem_and_descriptors(outcome, target, notes, artifacts, gates, fs, _ref)

    if question_class == "compare_track":
        notes.append(
            "compare_track delivers the REFERENCE side only. Next, Claude composes the "
            "project side: render_track_region(<your track>, region) → characterize(<render>) "
            "→ compare_character(reference_descriptor, render_descriptor). analysis/ makes no "
            "REAPER-bridge calls by design (§3 module boundary)."
        )

    return _finish()


def _package_stem(
    outcome: LadderOutcome | None,
    artifacts: dict[str, Any],
    gates: dict[str, GateSet],
    notes: list[str],
) -> None:
    """Put the ladder's delivered stem + its gates on the envelope (no descriptors)."""
    if outcome is None or outcome.stems is None:
        notes.append("no separated stem available at the reached tier.")
        return
    artifacts["stems"] = {t: sr.model_dump(mode="json") for t, sr in outcome.stems.items()}
    for t, sr in outcome.stems.items():
        if sr.gates is not None:
            gates[t] = sr.gates


def _package_stem_and_descriptors(
    outcome: LadderOutcome | None,
    target: str,
    notes: list[str],
    artifacts: dict[str, Any],
    gates: dict[str, GateSet],
    fs: CacheFS,
    _ref: Any,
) -> None:
    """§9 ``sound_at``/``compare_track`` packaging: the target stem, its §5 gates, and its §7
    descriptors (via ``characterize`` — carrying the R3 ``complement=True`` flag end-to-end)."""
    _package_stem(outcome, artifacts, gates, notes)
    if outcome is None or outcome.stems is None or target not in outcome.stems:
        if outcome is not None and outcome.stems is None:
            notes.append(
                f"answered at rung {outcome.rung}: {outcome.source} (no stem to describe)."
            )
        return
    handle = _stem_handle(outcome.stems[target])
    character = _ref.characterize(handle, cache=fs)
    artifacts["descriptors"] = [character.model_dump(mode="json")]


def _run_transcription(
    file: str,
    region: Window | None,
    pcm: Pcm,
    tl: Timeline,
    target: str,
    outcome: LadderOutcome | None,
    the_resolver: Resolver,
    fs: CacheFS,
    transcribe: Callable[[Path, Path], list[Any]] | None,
) -> tuple[list[Any], str]:
    """§6 R6 / §9 T5: pick the transcription source (a gate-passed stem if the ladder produced
    one, else the target's most-exposed raw window, else the whole/region mix) and run
    basic-pitch (subprocess, isolated env) over it. ``transcribe`` is the injection seam; the
    production default is the isolated ``transcribe_cli.transcribe``."""
    if transcribe is None:
        from orpheus_mcp.analysis.transcribe_cli import transcribe as _transcribe

        transcribe = _transcribe

    out_dir = Path(fs.artifact_dir(pcm.pcm_hash)) / "transcribe"
    out_dir.mkdir(parents=True, exist_ok=True)

    if outcome is not None and outcome.stems is not None and target in outcome.stems:
        src_stem = outcome.stems[target]
        return transcribe(Path(src_stem.stem_path), out_dir), (
            f"{outcome.rung} stem ({src_stem.model_id})"
        )

    # No usable stem — transcribe the target's most-exposed raw window (or the requested
    # region, or the whole mix), written out as a temp wav.
    win: Window | None = None
    if target:
        cands = list(the_resolver(target))
        tclasses = classes_for_target(cands[0], target) if cands else []
        win = most_exposed_window(tl, tclasses) if tclasses else None
    if win is None:
        win = region if region is not None else Window(start_s=0.0, end_s=pcm.duration_s)
    crop = _crop_audio(pcm, win)
    src_wav = out_dir / "source.wav"
    sf.write(str(src_wav), crop, pcm.sample_rate, subtype="PCM_16")
    return transcribe(src_wav, out_dir), f"most-exposed raw window {_window_desc(win, region)}"
