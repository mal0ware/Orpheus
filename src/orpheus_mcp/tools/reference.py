"""The ``reference`` category (§9): local reference-track analysis tools. Tools:
``recon_reference`` (T1), ``instrument_timeline`` (T2), ``get_chords`` (§13),
``separate_region`` (Task 4.2, §5/§9 — engine adapters + verification gates). Logic lives
in ``orpheus_mcp.analysis`` (pure functions, no REAPER); this layer only handles
decode -> cache -> tool-result plumbing.
"""

from __future__ import annotations

import contextlib
import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from fastmcp import FastMCP

from orpheus_mcp.analysis import cachefs, model_registry
from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.separation import SeparationError
from orpheus_mcp.analysis.separation import separate_region as _run_separation_core
from orpheus_mcp.models import (
    BeatGrid,
    ChordsLane,
    GateResult,
    GateSet,
    LocateResult,
    LyricsResult,
    Recon,
    SoundCharacter,
    StemHandle,
    StemResult,
    Timeline,
    Window,
)

if TYPE_CHECKING:
    from orpheus_mcp.analysis.lyrics import AsrEngine
    from orpheus_mcp.analysis.model_registry import ModelEntry
    from orpheus_mcp.analysis.separation import SeparatorEngine
    from orpheus_mcp.analysis.timeline import TaggerEngine

__all__ = [
    "recon_reference",
    "instrument_timeline",
    "get_chords",
    "separate_region",
    "verify_stem",
    "characterize",
    "transcribe_lyrics",
    "locate_lyric",
    "register",
]


class ReferenceEnvMissing(Exception):
    """A tool call needs the ``[analysis]`` extra (panns-inference + torch) that isn't
    installed. Raised instead of letting a raw ``ImportError`` escape a tool call — mirrors
    ``analysis.separation.SeparatorEnvMissing`` / ``analysis.lyrics.LyricsAsrEnvMissing`` for
    the base analysis stack ``instrument_timeline``/the §6 ladder needs. A future orchestrator
    (§9 ``analyze_reference``) catches this and records ``capped_by_missing_extra="analysis"``
    in the answer envelope."""

_RO = {"readOnlyHint": True}

_RECON_KIND = "recon"
_RECON_SCHEMA_VERSION = 1

_TIMELINE_KIND = "timeline"
_TIMELINE_SCHEMA_VERSION = 1

_CHORDS_KIND = "chords"
_CHORDS_SCHEMA_VERSION = 1


def _recon_meta(file: str) -> dict:
    """The cache sidecar shape for kind ``"recon"`` — no model_id/checkpoint fields (T1 has
    no model dependency). Built entirely from cheap, pre-decode information
    (``decoder_for`` never decodes) so a cache HIT never pays for a decode."""
    return {
        "schema_version": _RECON_SCHEMA_VERSION,
        "registry_version": cachefs.registry_version(),
        "verify_config_hash": cachefs.verify_config_hash(),
        "decoder": cachefs.decoder_for(file),
    }


def _recon_read_through(
    file: str, fs: CacheFS, *, pcm: Pcm | None = None
) -> tuple[Recon, Pcm | None]:
    """The shared recon cache read-through (§9 T0): a cache HIT never decodes. ``recon_reference``
    and ``get_chords`` both go through this one implementation rather than duplicating the
    decode -> cache -> tool-result plumbing.

    ``pcm``, if the caller already holds a decoded ``Pcm`` (``get_chords`` does, on a chords
    cache MISS — it needs the audio for the chord lane regardless of whether recon is
    cached), is reused instead of decoding a second time on a recon MISS; returned back to
    the caller (still ``None`` on a recon HIT, since no decode happened here).

    Every call to ``cachefs.decode``/``cachefs.pcm_hash`` below is via the module (not a
    bare import) so a test can monkeypatch ``cachefs.decode`` and see every decode this
    function triggers, directly or through ``pcm_hash``'s own memoized decode."""
    from orpheus_mcp.analysis.recon import run_recon

    file_hash = cachefs.pcm_hash(file)  # memoized by file-bytes-hash — free on repeat calls
    meta = _recon_meta(file)

    hit = fs.read_json(file_hash, _RECON_KIND, meta=meta)
    if hit is not None:
        return Recon.model_validate(hit), pcm

    if pcm is None:
        pcm = cachefs.decode(file)
    recon = run_recon(pcm)
    fs.write_json(file_hash, _RECON_KIND, recon.model_dump(mode="json"), meta=meta)
    return recon, pcm


def recon_reference(file: str, *, cache: CacheFS | None = None) -> Recon:
    """Decode ``file``, run the T1 recon pass, and cache the result under kind
    ``"recon"`` (keyed by the decoded-PCM hash + this artifact's meta shape). A cache HIT
    never re-decodes or re-runs recon (§9 T0)."""
    fs = cache if cache is not None else CacheFS()
    recon, _ = _recon_read_through(file, fs)
    return recon


def _timeline_meta(file: str, tagger: TaggerEngine) -> dict:
    """The cache sidecar shape for kind ``"timeline"``. Unlike T1 recon, T2 has a real
    model dependency — the tagger that produced the artifact must be part of the cache
    key, or swapping ``PannsTagger`` for a test double (or a future SED head) against the
    same audio could return a stale/wrong-engine HIT. Real ``PannsTagger`` carries its own
    ``model_id`` (registry id + head); anything else (a test double) falls back to its
    class name so it never collides with the real tagger's cache entries."""
    return {
        "schema_version": _TIMELINE_SCHEMA_VERSION,
        "registry_version": cachefs.registry_version(),
        "verify_config_hash": cachefs.verify_config_hash(),
        "decoder": cachefs.decoder_for(file),
        "tagger_id": getattr(tagger, "model_id", f"stub:{type(tagger).__name__}"),
    }


def instrument_timeline(
    file: str, *, cache: CacheFS | None = None, tagger: TaggerEngine | None = None
) -> Timeline:
    """Decode ``file``, run the T2 tagger pass, and cache the result under kind
    ``"timeline"`` (keyed by the decoded-PCM hash + this artifact's meta shape, which
    includes the tagger's identity). A cache HIT never re-decodes or re-runs the tagger
    (§9 T0).

    ``tagger`` defaults to :class:`~orpheus_mcp.analysis.timeline.PannsTagger` — imported
    lazily here (not at module import time) so this module never requires torch/
    panns-inference just to be imported; tests inject a stub via the ``tagger`` param the
    same way ``recon_reference`` accepts an explicit ``cache``."""
    from orpheus_mcp.analysis.timeline import PannsTagger, build_timeline

    fs = cache if cache is not None else CacheFS()
    engine: TaggerEngine = tagger if tagger is not None else PannsTagger()
    file_hash = cachefs.pcm_hash(file)
    meta = _timeline_meta(file, engine)

    hit = fs.read_json(file_hash, _TIMELINE_KIND, meta=meta)
    if hit is not None:
        return Timeline.model_validate(hit)

    pcm = cachefs.decode(file)
    timeline = build_timeline(pcm, engine)
    fs.write_json(file_hash, _TIMELINE_KIND, timeline.model_dump(mode="json"), meta=meta)
    return timeline


def _chords_meta(file: str) -> dict:
    """The cache sidecar shape for kind ``"chords"``. ``produced_by`` is pinned to
    ``"baseline"`` here (not derived per-segment) — this task only ships the §13 baseline
    lane; a future model-lane rewrite (Phase 9) would need its own cache-key ingredient the
    way ``instrument_timeline``'s ``tagger_id`` does, not a change to this constant."""
    return {
        "schema_version": _CHORDS_SCHEMA_VERSION,
        "registry_version": cachefs.registry_version(),
        "verify_config_hash": cachefs.verify_config_hash(),
        "decoder": cachefs.decoder_for(file),
        "produced_by": "baseline",
    }


def _fixed_frame_grid(pcm: Pcm) -> BeatGrid:
    """A synthetic one-window-per-second ``BeatGrid`` reproducing ``harmony.py``'s private
    ``_fixed_window_chroma`` window boundaries (whole seconds, ``ceil(duration)`` windows) —
    ``beat_chroma``/``bass_pc_lane`` only ever read ``grid.times_s`` as interval boundaries,
    so this lets the §13.4 rubato fallback reuse the public beat-mode functions instead of
    reaching into harmony.py's internals for a second, fixed-window chroma path."""
    n_windows = max(1, math.ceil(pcm.duration_s))
    return BeatGrid(
        times_s=[float(i) for i in range(n_windows)],
        downbeats_s=[],
        meter=(4, 4),
        confidence=0.0,
        meter_confidence=0.0,
        kind="fixed",
    )


def _full_baseline_lane(pcm: Pcm, recon: Recon) -> ChordsLane:
    """The §13 fully-refined baseline lane. ``harmony.py`` (Task 3.1) ships the recognizer
    and each refinement layer as standalone functions with no compose entry point — this is
    the one place that chains them in the documented order (§13 output schema) so
    ``bass``/``extensions``/``roman``/``key_context`` come back populated instead of at the
    raw recognizer's defaults: templates+Viterbi (``chord_lane_baseline``) -> §13.3
    extensions -> §13.2 bass-informed slash -> §13.4 theory (local key + Roman numerals)."""
    from orpheus_mcp.analysis.harmony import (
        bass_pc_lane,
        beat_chroma,
        chord_lane_baseline,
        detect_slash,
        refine_extensions,
        theory_layer,
    )

    lane = chord_lane_baseline(pcm, recon)
    frame_grid = recon.grid if lane.grid == "beat" else _fixed_frame_grid(pcm)

    chroma = beat_chroma(pcm, frame_grid, recon.tuning_offset_cents)
    lane = refine_extensions(lane, chroma, recon.grid)

    bass_pcs = bass_pc_lane(pcm, frame_grid)
    lane = detect_slash(lane, bass_pcs)

    return theory_layer(lane, recon.grid)


def _clamp_region(region: Window, duration_s: float) -> tuple[float, float]:
    """§3.1d: every region clamps to ``[0, duration]`` — a region reaching past EOF (or
    starting before 0) clamps rather than errors."""
    start = max(0.0, region.start_s)
    end = min(duration_s, region.end_s)
    return start, end


def _filter_lane_to_region(lane: ChordsLane, region: Window, duration_s: float) -> ChordsLane:
    """Region filtering happens AFTER the cache — ``get_chords`` always caches the
    whole-song lane, then filters this copy on the way out. Segments overlapping the
    clamped region are returned WHOLE (never re-cut to the region's exact edges); a region
    that clamps to an empty span (fully outside ``[0, duration]``) returns an empty-segments
    lane with ``grid`` preserved, not an error."""
    start, end = _clamp_region(region, duration_s)
    segments = [seg for seg in lane.segments if seg.start_s < end and seg.end_s > start]
    return ChordsLane(grid=lane.grid, segments=segments)


def get_chords(
    file: str, region: Window | None = None, *, cache: CacheFS | None = None
) -> ChordsLane:
    """Decode ``file``, compute the full §13 baseline chord lane (recognizer + extensions +
    slash + theory), and cache the WHOLE-SONG lane under kind ``"chords"`` (keyed by the
    decoded-PCM hash + this artifact's meta shape). A cache HIT never re-decodes or
    re-computes the lane (§9 T0); recon is reused via the shared ``_recon_read_through``
    read-through so a recon already cached by ``recon_reference`` is never recomputed.

    ``region``, if given, filters the (cached or freshly computed) lane to segments
    overlapping ``[region.start_s, min(region.end_s, duration)]`` — clamped per §3.1d —
    AFTER the cache read/write, so the cache always holds the whole song."""
    fs = cache if cache is not None else CacheFS()
    file_hash = cachefs.pcm_hash(file)
    meta = _chords_meta(file)

    duration_s: float | None
    hit = fs.read_json(file_hash, _CHORDS_KIND, meta=meta)
    if hit is not None:
        lane = ChordsLane.model_validate(hit)
        duration_s = None
    else:
        pcm = cachefs.decode(file)
        recon, _ = _recon_read_through(file, fs, pcm=pcm)
        lane = _full_baseline_lane(pcm, recon)
        fs.write_json(file_hash, _CHORDS_KIND, lane.model_dump(mode="json"), meta=meta)
        duration_s = recon.duration_s

    if region is None:
        return lane
    if duration_s is None:  # cache HIT path never needed a decode — fetch duration cheaply
        recon, _ = _recon_read_through(file, fs)
        duration_s = recon.duration_s
    return _filter_lane_to_region(lane, region, duration_s)


# --------------------------------------------------------------------------- #
# separate_region (Task 4.2, §5/§9)
# --------------------------------------------------------------------------- #

_STEM_SCHEMA_VERSION = 1

# Every §4 task that can actually produce a stem — excludes tagger/transcription/chord/
# lyrics-asr, which have no ``targets`` a separated stem could come from.
_SEPARATOR_TASKS = ("multi-stem", "single-target", "dereverb", "drumkit-split")


def _resolve_separator_entry(target: str, model_id: str | None) -> ModelEntry:
    """The rank-1 registry entry producing ``target`` (or the explicit ``model_id``,
    looked up directly, any task, bypassing ranking). Candidates are gathered across
    EVERY separator-capable task and merged by ``rank`` — the registry has no single
    "task" a target belongs to (a target may be a multi-stem output AND a dedicated
    single-target model), so this can't just call ``model_registry.resolve`` once.
    Each per-task list already applies ``resolve``'s VRAM/license/engine-status
    filtering (§15.5) — only the final merge-and-sort is done here."""
    entries = model_registry.load_registry(allow_unpinned=True)
    if model_id is not None:
        for entry in entries:
            if entry.id == model_id:
                return entry
        raise SeparationError(f"Unknown model_id: {model_id!r}")

    free = model_registry.free_vram_gb()
    candidates: list[ModelEntry] = []
    for task in _SEPARATOR_TASKS:
        candidates.extend(model_registry.resolve(task, target, vram_free_gb=free))
    if not candidates:
        raise SeparationError(
            f"No registry entry (multi-stem/single-target/dereverb/drumkit-split) "
            f"produces target {target!r}."
        )
    return sorted(candidates, key=lambda e: e.rank)[0]


def _stem_verify_meta(entry: ModelEntry) -> dict:
    """The §9 per-stem cache sidecar shape: same ingredients as every other artifact's
    meta (schema/registry/verify-config) plus the producing model's identity — a
    checkpoint swap or re-pin must bust a cached stem the same way it busts recon/
    timeline/chords."""
    checkpoint = entry.sha256
    return {
        "schema_version": _STEM_SCHEMA_VERSION,
        "registry_version": cachefs.registry_version(),
        "verify_config_hash": cachefs.verify_config_hash(),
        "model_id": entry.id,
        "checkpoint_sha256": checkpoint,
    }


def _align_len(arr: np.ndarray, n: int) -> np.ndarray:
    frames = arr.shape[0]
    if frames == n:
        return arr
    if frames > n:
        return arr[:n]
    pad = np.zeros((n - frames, *arr.shape[1:]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def separate_region(
    file: str,
    target: str,
    region: Window | None = None,
    model_id: str | None = None,
    *,
    cache: CacheFS | None = None,
    engine: SeparatorEngine | None = None,
    tagger: TaggerEngine | None = None,
) -> StemResult:
    """§9 tool contract: resolves the rank-1 (or explicit ``model_id``) registry entry
    producing ``target``, runs the Task 4.1 separation core (region-cropped, VRAM-
    guarded, length-aligned — ``analysis.separation.separate_region``), runs
    verification gates 1-3 (§5: mix consistency across every co-produced stem, bleed via
    the T2 tagger run ON the stem, silence honesty against the cached/region-cropped mix
    timeline), caches the requested target's stem FLAC + a ``verify.json`` sidecar under
    the §9 ``stems/<model-id>/<start_ms>-<end_ms>/`` layout, and returns that target's
    ``StemResult`` with ``gates`` populated. Gate 4 (agreement) is not implemented yet
    (Phase 5) — ``gates.agreement`` always reads ``na``/``None`` and is excluded from the
    §5 overall decision rule (fail if any applicable gate fails; unverifiable iff every
    gate is N/A).

    **Desktop-tier tool** (documented, not silently hidden): needs the ``[separation]``
    extra (audio-separator / vendored MSST) AND ``[analysis]`` (panns-inference/torch,
    for the bleed judge) at CALL time. This module only imports numpy/soundfile eagerly;
    the heavier deps are imported lazily inside ``analysis.separation``/
    ``analysis.timeline`` and raise a typed error (``SeparatorEnvMissing``, an honest
    "no compatible model for this engine" ``SeparationError``, ...) if genuinely absent
    — the ``register()`` gate below only probes for torch+panns-inference (the same
    probe ``instrument_timeline`` uses), matching the plan's ``capped_by_missing_extra``
    pattern for a ``[separation]``-tier tool.
    """
    from orpheus_mcp.analysis.timeline import PannsTagger, crop_timeline
    from orpheus_mcp.analysis.verify import run_gates

    fs = cache if cache is not None else CacheFS()
    pcm = cachefs.decode(file)
    entry = _resolve_separator_entry(target, model_id)
    tagger_engine: TaggerEngine = tagger if tagger is not None else PannsTagger()

    req_start = 0.0 if region is None else region.start_s
    req_end = pcm.duration_s if region is None else region.end_s
    clamp_start = max(0.0, min(req_start, pcm.duration_s))
    clamp_end = max(clamp_start, min(req_end, pcm.duration_s))
    cache_region = Window(start_s=clamp_start, end_s=clamp_end)

    meta = _stem_verify_meta(entry)
    stem_dir = fs.stem_dir(pcm.pcm_hash, entry.id, cache_region)
    stem_path = stem_dir / f"{target}.flac"
    verify_path = stem_dir / "verify.json"
    meta_path = stem_dir / "meta.json"

    if stem_path.is_file() and verify_path.is_file() and meta_path.is_file():
        try:
            stored_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored_meta = None
        if stored_meta == meta:
            gate_data = json.loads(verify_path.read_text(encoding="utf-8"))
            info = sf.info(str(stem_path))
            return StemResult(
                stem_path=str(stem_path),
                model_id=entry.id,
                target=target,
                region_trimmed=cache_region,
                sample_rate=info.samplerate,
                gates=GateSet.model_validate(gate_data),
                notes=[],
            )

    results = _run_separation_core(pcm, entry, region, engine=engine)
    if target not in results:
        raise SeparationError(
            f"{entry.id} did not produce a {target!r} stem (produced {sorted(results)})."
        )
    stem_result = results[target]
    rt = stem_result.region_trimmed
    sr = stem_result.sample_rate

    stems_arrays: dict[str, np.ndarray] = {}
    for tgt, res in results.items():
        data, _sr = sf.read(res.stem_path, always_2d=True, dtype="float32")
        stems_arrays[tgt] = data

    mix_samples = pcm.samples if pcm.samples.ndim == 2 else pcm.samples.reshape(-1, 1)
    start_i = min(int(round(rt.start_s * pcm.sample_rate)), mix_samples.shape[0])
    end_i = min(int(round(rt.end_s * pcm.sample_rate)), mix_samples.shape[0])
    start_i = min(start_i, end_i)
    mix_crop = _align_len(mix_samples[start_i:end_i], stems_arrays[target].shape[0])

    mix_tl = instrument_timeline(file, cache=fs, tagger=tagger_engine)
    region_tl = crop_timeline(mix_tl, rt)

    gates = run_gates(
        mix=mix_crop,
        stems=stems_arrays,
        target=target,
        sr=sr,
        multi_stem=entry.task == "multi-stem",
        tagger=tagger_engine,
        target_tagger_classes=list(model_registry.classes_for_target(entry, target)),
        mix_timeline=region_tl,
    )

    stem_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stem_result.stem_path, stem_path)
    verify_path.write_text(json.dumps(gates.model_dump(mode="json")), encoding="utf-8")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with contextlib.suppress(OSError):
        shutil.rmtree(Path(stem_result.stem_path).parent, ignore_errors=True)

    return StemResult(
        stem_path=str(stem_path),
        model_id=entry.id,
        target=target,
        region_trimmed=rt,
        sample_rate=sr,
        gates=gates,
        notes=list(stem_result.notes),
    )


# --------------------------------------------------------------------------- #
# verify_stem (Task 5.3, §9) — re-verify an already-separated stem, librosa tier
# --------------------------------------------------------------------------- #


def verify_stem(
    stem_path: str,
    mix_path: str,  # noqa: ARG001 - reserved for the (N/A here) multi-stem mix-consistency gate
    region: Window | None = None,
    timeline: Timeline | None = None,
    *,
    cache: CacheFS | None = None,  # noqa: ARG001 - signature parity with the other reference tools
) -> GateSet:
    """§9 ``verify_stem`` contract: re-verify a single already-separated stem on disk against
    its source mix over ``region``, returning the §5 :class:`GateSet`. This is the LIBROSA-tier
    verifier (no PANNs/torch at call time) — deliberately lighter than ``separate_region``,
    which re-runs the tagger for the bleed gate. Its gate applicability is therefore narrower,
    and stated plainly rather than silently degraded:

    - **Gate 1 (mix consistency)** — always ``na`` here: a single stem cannot be checked against
      ``mix == Σ stems`` (that needs every co-produced stem; ``separate_region`` owns it). The
      ``mix_path`` argument is accepted for signature/contract parity and a future multi-stem
      variant, but this single-stem tool does not decode it.
    - **Gate 2 (bleed)** — always ``na`` here: bleed requires a per-target PANNs run ON the stem,
      which this librosa-tier tool intentionally does not do (use ``separate_region`` for a
      bleed-judged stem).
    - **Gate 3 (silence honesty)** — computed IFF ``timeline`` is supplied. ``timeline`` is the
      source mix's :class:`~orpheus_mcp.models.Timeline` (from ``instrument_timeline``); it is
      cropped to ``region`` and re-indexed so frame 0 == the stem's sample 0, and EVERY mapped
      class in it is treated as the activity reference — i.e. the stem is expected quiet in the
      windows where the mix shows no mapped-instrument activity at all. This is a deliberately
      conservative honesty check (it flags energy in genuinely silent passages, not bleed under
      other instruments); the caller may pre-filter ``timeline.classes`` to a single target's
      classes to sharpen it. With ``timeline=None`` there is no activity reference, so gate 3 is
      ``na`` too.
    - **Gate 4 (agreement)** — always ``na`` here: no second engine's take is supplied.

    With ``timeline=None`` every gate is ``na`` and ``overall`` is ``unverifiable`` (§5: climbing
    cannot fix unverifiability) — an honest "this tier had nothing to check", not a false pass.
    Does not modify the input files or any REAPER project (``readOnlyHint``).
    """
    from orpheus_mcp.analysis.timeline import crop_timeline
    from orpheus_mcp.analysis.verify import gate_silence_honesty, overall_verdict

    na = GateResult(verdict="na", value=None)
    stem_data, stem_sr = sf.read(stem_path, always_2d=True, dtype="float32")
    stem = np.asarray(stem_data, dtype=np.float32)

    if timeline is None:
        g3 = na
    else:
        stem_dur = stem.shape[0] / stem_sr if stem_sr else 0.0
        crop = region if region is not None else Window(start_s=0.0, end_s=stem_dur)
        region_tl = crop_timeline(timeline, crop)
        target_idx = list(range(len(region_tl.classes)))
        g3 = gate_silence_honesty(stem, int(stem_sr), region_tl, target_idx)

    return GateSet(
        mix_consistency=na,
        bleed=na,
        silence_honesty=g3,
        agreement=na,
        overall=overall_verdict([na, na, g3, na]),
    )


# --------------------------------------------------------------------------- #
# characterize (Task 6.1, §7/§9) — SoundCharacter descriptors, librosa tier
# --------------------------------------------------------------------------- #

_DESCRIPTORS_SCHEMA_VERSION = 1


def _descriptors_meta(
    source_path: str, meta_stem_type: str, produced_by: str, complement: bool, dereverb_used: bool
) -> dict:
    """The §9 cache sidecar shape for a ``descriptors/<qualifier>`` artifact — the standard
    schema/registry/verify-config ingredients plus WHAT was characterized (stem type, producer,
    complement flag) and WHETHER the vocal dereverb wet−dry path ran (a heuristic-only result
    and a dereverbed result for the same audio must not collide)."""
    return {
        "schema_version": _DESCRIPTORS_SCHEMA_VERSION,
        "registry_version": cachefs.registry_version(),
        "verify_config_hash": cachefs.verify_config_hash(),
        "decoder": cachefs.decoder_for(source_path),
        "stem_type": meta_stem_type,
        "produced_by": produced_by,
        "complement": complement,
        "dereverb_used": dereverb_used,
    }


def _crop_source(src, region: Window):  # type: ignore[no-untyped-def]
    """Return a copy of ``src`` with its PCM cropped to ``region`` (clamped to ``[0, dur]``)
    and ``meta.region`` set — §7 descriptors then measure only the requested span."""
    from dataclasses import replace

    from orpheus_mcp.analysis.cachefs import Pcm

    samples = src.pcm.samples
    if samples.ndim < 2:
        samples = samples.reshape(-1, 1)
    sr = src.pcm.sample_rate
    n = samples.shape[0]
    start_i = max(0, min(int(round(region.start_s * sr)), n))
    end_i = max(start_i, min(int(round(region.end_s * sr)), n))
    cropped = samples[start_i:end_i]
    new_pcm = Pcm(
        samples=cropped,
        sample_rate=sr,
        path=src.pcm.path,
        pcm_hash=src.pcm.pcm_hash,
        duration_s=cropped.shape[0] / sr if sr else 0.0,
        decoder=src.pcm.decoder,
    )
    clamped = Window(start_s=start_i / sr, end_s=end_i / sr) if sr else region
    new_meta = src.meta.model_copy(update={"region": clamped})
    return replace(src, pcm=new_pcm, meta=new_meta)


def characterize(
    source: str | StemHandle,
    region: Window | None = None,
    *,
    cache: CacheFS | None = None,
    dereverb: SeparatorEngine | None = None,
) -> SoundCharacter:
    """§7 ``SoundCharacter``: measure post-FX spectral / dynamics / harmonics / stereo /
    time-fx / pitch descriptors for ``source`` (a literal audio path or a :class:`StemHandle`
    to a cached stem), caching the result under kind ``descriptors/<stem-or-region>`` (§9). A
    cache HIT never re-runs the DSP.

    ``region`` (literal-path sources) crops the analysis to that span before measuring. The
    vocal reverb wet−dry path (``reverb_approx=False``) runs only for a ``stem_type=="vocals"``
    source with a usable dereverb engine — ``dereverb`` may be injected (tests / the ladder);
    otherwise a default engine is resolved when [separation] is present, and the reverb
    heuristic (``reverb_approx=True``, noted in ``SoundCharacter.notes``) runs when it is not.
    Does not modify the input audio or any REAPER project — it does write to the local
    analysis cache."""
    from orpheus_mcp.analysis.descriptors import compute_character, resolve_source

    fs = cache if cache is not None else CacheFS()
    src = resolve_source(source, fs)
    if region is not None:
        src = _crop_source(src, region)

    engine = dereverb
    if engine is None and src.meta.stem_type == "vocals":
        engine = _default_dereverb_engine()
    dereverb_used = engine is not None and src.meta.stem_type == "vocals"

    source_path = source if isinstance(source, str) else source.stem_path
    file_hash = src.pcm.pcm_hash
    qualifier = src.meta.stem_type
    if src.meta.region is not None:
        start_ms = round(src.meta.region.start_s * 1000)
        end_ms = round(src.meta.region.end_s * 1000)
        qualifier = f"{qualifier}_{start_ms}-{end_ms}"
    kind = f"descriptors/{qualifier}"
    meta = _descriptors_meta(
        source_path, src.meta.stem_type, src.meta.produced_by, src.meta.complement, dereverb_used
    )

    hit = fs.read_json(file_hash, kind, meta=meta)
    if hit is not None:
        return SoundCharacter.model_validate(hit)

    character = compute_character(src, dereverb=engine)
    fs.write_json(file_hash, kind, character.model_dump(mode="json"), meta=meta)
    return character


def _default_dereverb_engine() -> SeparatorEngine | None:
    """The default engine for the rank-1 dereverb registry entry, or ``None`` when no entry
    exists or [separation] cannot build one — constructing the engine is cheap (heavy deps load
    only when it actually runs), so a graceful ``None`` here just routes vocals to the reverb
    heuristic."""
    try:
        from orpheus_mcp.analysis import model_registry as _mr
        from orpheus_mcp.analysis.separation import _default_engine

        entry = _mr.resolve_dereverb_entry()
        if entry is None:
            return None
        return _default_engine(entry)
    except Exception:  # noqa: BLE001 - any resolution/build failure degrades to the heuristic
        return None


# --------------------------------------------------------------------------- #
# transcribe_lyrics + locate_lyric (Task 8.2, §14.2) — the vocal-stem-via-ladder ASR index
# and the tool-layer wrapper over the pure fuzzy resolver in analysis/lyrics.py.
# --------------------------------------------------------------------------- #

_LYRICS_KIND = "lyrics"
_LINES_KIND = "lines"
_SECTIONS_KIND = "sections"
_LYRICS_SCHEMA_VERSION = 1
_LYRICS_TASK = "lyrics-asr"
_VOCALS_TARGET = "vocals"


def _asr_model_id() -> str:
    """The rank-1 ``lyrics-asr``/``fasterwhisper`` registry entry's id, or ``"unknown"`` if
    the registry ships none — this is CHEAP (registry JSON read, no import of faster_whisper
    itself) so it can be part of the cache meta on every call, including a HIT (§9: meta must
    be buildable before paying for a decode)."""
    entries = [
        e
        for e in model_registry.entries_for_task(_LYRICS_TASK, allow_unpinned=True)
        if e.engine == "fasterwhisper"
    ]
    if not entries:
        return "unknown"
    return sorted(entries, key=lambda e: e.rank)[0].id


def _lyrics_meta(file: str, asr_model_id: str) -> dict:
    """The §9 cache sidecar shape shared by the three ``lyrics``/``lines``/``sections`` cache
    kinds (§9 — ``lyrics.json``/``lines.json``/``sections.json``, mirrored here as three
    ``CacheFS`` kinds so a re-ask never re-pays ASR/alignment cost). ``asr_model_id`` busts a
    cached transcript when the pinned faster-whisper checkpoint changes, even though
    ``registry_version`` already covers it too (belt-and-suspenders: the brief calls this out
    explicitly since ASR quality is this artifact's whole reason to exist)."""
    return {
        "schema_version": _LYRICS_SCHEMA_VERSION,
        "registry_version": cachefs.registry_version(),
        "verify_config_hash": cachefs.verify_config_hash(),
        "decoder": cachefs.decoder_for(file),
        "asr_model_id": asr_model_id,
    }


def _read_lyrics_cache(fs: CacheFS, file_hash: str, meta: dict) -> LyricsResult | None:
    """A HIT requires all three kinds to be present and meta-matching — a partial write (one
    kind stale/missing) is treated as a full miss, never a Frankenstein reconstruction."""
    core = fs.read_json(file_hash, _LYRICS_KIND, meta=meta)
    lines = fs.read_json(file_hash, _LINES_KIND, meta=meta)
    sections = fs.read_json(file_hash, _SECTIONS_KIND, meta=meta)
    if core is None or lines is None or sections is None:
        return None
    return LyricsResult(
        schema_version=core.get("schema_version", _LYRICS_SCHEMA_VERSION),
        status=core["status"],
        language=core.get("language"),
        aligned=core.get("aligned", False),
        words=core.get("words", []),
        lines=lines.get("lines", []),
        sections=sections.get("sections", []),
    )


def _write_lyrics_cache(fs: CacheFS, file_hash: str, meta: dict, result: LyricsResult) -> None:
    dumped = result.model_dump(mode="json")
    core = {k: dumped[k] for k in ("schema_version", "status", "language", "aligned", "words")}
    fs.write_json(file_hash, _LYRICS_KIND, core, meta=meta)
    fs.write_json(file_hash, _LINES_KIND, {"lines": dumped["lines"]}, meta=meta)
    fs.write_json(file_hash, _SECTIONS_KIND, {"sections": dumped["sections"]}, meta=meta)


def _resolve_tagger(tagger: TaggerEngine | None) -> TaggerEngine:
    if tagger is not None:
        return tagger
    try:
        from orpheus_mcp.analysis.timeline import PannsTagger
    except ImportError as exc:
        raise ReferenceEnvMissing(
            "transcribe_lyrics needs the `[analysis]` extra (panns-inference + torch) to "
            "build the T2 timeline the §6 vocal-stem ladder depends on."
        ) from exc
    return PannsTagger()


def _resolve_asr(asr: AsrEngine | None) -> AsrEngine:
    if asr is not None:
        return asr
    from orpheus_mcp.analysis.lyrics import FasterWhisperAsr

    return FasterWhisperAsr()  # lazy-loaded; LyricsAsrEnvMissing raises at first .transcribe()


def _separator_registry_resolver(
    free_vram_gb: float | None,
) -> Callable[[str], list[ModelEntry]]:
    """A ``router.climb`` ``Resolver`` — every registry candidate producing ``target`` across
    every separator-capable task, rank-merged, the SAME candidate-gathering logic
    ``_resolve_separator_entry`` uses for ``separate_region`` (kept in sync deliberately: the
    ladder and the single-shot tool must agree on what "rank-1 for this target" means)."""

    def _resolve(target: str) -> list[ModelEntry]:
        candidates: list[ModelEntry] = []
        for task in _SEPARATOR_TASKS:
            candidates.extend(model_registry.resolve(task, target, vram_free_gb=free_vram_gb))
        return sorted(candidates, key=lambda e: e.rank)

    return _resolve


def _engine_factory(
    override: SeparatorEngine | None,
) -> Callable[[ModelEntry], SeparatorEngine]:
    from orpheus_mcp.analysis.separation import _default_engine

    def _get(entry: ModelEntry) -> SeparatorEngine:
        return override if override is not None else _default_engine(entry)

    return _get


def _vocal_stem_via_ladder(
    file: str,
    pcm: Pcm,
    fs: CacheFS,
    *,
    tagger: TaggerEngine,
    engine: SeparatorEngine | None,
    max_seconds: float,
) -> tuple[Pcm | None, bool]:
    """Resolve the full-song vocal stem through the §6 ladder (``router.climb``, target
    ``"vocals"``, ``max_tier="T3"`` — the ``QUESTION_TIERS["lyric_navigate"]`` ceiling). A
    "T3" ceiling only ever reaches pre-flight (absent/exposure) + rank-1 (§6: every R1+
    escalation rung needs T4) — intentionally a SINGLE validated separation, not full R1-R6
    escalation: a lyrics index needs one good stem, not the best-of-many-tries answer a
    "sound_at" question would chase. Any genuine [separation]/env failure (``SeparatorEnvMissing``
    / ``SeparationError``) propagates from ``climb`` uncaught — a typed, documented failure a
    future orchestrator translates into ``capped_by_missing_extra`` (§9), not swallowed here.

    Returns ``(vocal_pcm, confirmed_absent)``. ``confirmed_absent=True`` (T2 pre-flight: the
    vocal tagger classes never exceed the activity threshold anywhere in the song) means the
    caller should skip VAD/ASR entirely and return ``no_vocals`` directly — cheap AND honest,
    consistent with §14.1's own audio-level hallucination gate. Any other stem-less rung (the
    rare pre-flight "exposed" case — a window where vocals are already isolated in the raw
    mix) falls back to the RAW MIX ``pcm`` as ``vocal_pcm``; ``build_lyrics``'s own VAD gate
    remains the final honesty backstop, so this can never fabricate lyrics from a genuinely
    silent mix — it can only run ASR without the benefit of separation."""
    from orpheus_mcp.analysis import router

    free = model_registry.free_vram_gb()
    registry = _separator_registry_resolver(free)
    engines = _engine_factory(engine)
    timeline = instrument_timeline(file, cache=fs, tagger=tagger)

    outcome = router.climb(
        pcm,
        _VOCALS_TARGET,
        None,
        registry=registry,
        engines=engines,
        tagger=tagger,
        timeline=timeline,
        max_tier="T3",
        max_seconds=max_seconds,
        cache=fs,
    )

    if outcome.rung == "absent":
        return None, True
    if outcome.stems is not None and _VOCALS_TARGET in outcome.stems:
        stem_path = outcome.stems[_VOCALS_TARGET].stem_path
        return cachefs.decode(stem_path), False
    return pcm, False  # "exposed" (or any other stem-less rung): best-effort on the raw mix


def transcribe_lyrics(
    file: str,
    *,
    cache: CacheFS | None = None,
    asr: AsrEngine | None = None,
    tagger: TaggerEngine | None = None,
    engine: SeparatorEngine | None = None,
    max_seconds: float | None = None,
) -> LyricsResult:
    """§14 lyric index: resolve the vocal stem through the §6 ladder (cached once per §3.1 —
    the SAME ``CacheFS`` the ladder writes its own ``ladder__vocals__...`` entry into, so a
    re-ask never re-separates even across a ``transcribe_lyrics`` cache miss on THIS artifact),
    run the §14.1 VAD-gated ASR + alignment + assembly pipeline, and cache the result under
    the ``lyrics``/``lines``/``sections`` kinds (§9). A cache HIT returns straight from disk —
    no decode, no ladder, no ASR — so a file transcribed once elsewhere (a shared cache dir)
    never re-pays the T3 vocal-stem cost or the ASR cost here.

    ``asr``/``tagger``/``engine`` are injection seams (tests; a production call leaves them
    ``None`` and gets the real ``FasterWhisperAsr``/``PannsTagger``/registry-resolved separator
    engine). Raises ``ReferenceEnvMissing`` ([analysis] extra), ``SeparatorEnvMissing``/
    ``SeparationError`` ([separation] extra / no compatible vocal model), or
    ``LyricsAsrEnvMissing`` (faster-whisper) on a genuine cache MISS without the needed stack —
    never a bare ``ImportError`` escaping the call."""
    from orpheus_mcp.analysis.lyrics import build_lyrics

    fs = cache if cache is not None else CacheFS()
    asr_id = _asr_model_id()
    file_hash = cachefs.pcm_hash(file)
    meta = _lyrics_meta(file, asr_id)

    hit = _read_lyrics_cache(fs, file_hash, meta)
    if hit is not None:
        return hit

    tagger_engine = _resolve_tagger(tagger)
    asr_engine = _resolve_asr(asr)
    budget = max_seconds if max_seconds is not None else CFG.DEFAULT_MAX_SECONDS

    pcm = cachefs.decode(file)
    recon, _ = _recon_read_through(file, fs, pcm=pcm)

    vocal_pcm, confirmed_absent = _vocal_stem_via_ladder(
        file, pcm, fs, tagger=tagger_engine, engine=engine, max_seconds=budget
    )
    if confirmed_absent:
        result = LyricsResult(
            status="no_vocals", language=None, aligned=False, words=[], lines=[], sections=[]
        )
    else:
        assert vocal_pcm is not None
        result = build_lyrics(vocal_pcm, recon, asr_engine)

    _write_lyrics_cache(fs, file_hash, meta, result)
    return result


def locate_lyric(
    file: str,
    quote: str,
    occurrence: int | None = None,
    *,
    cache: CacheFS | None = None,
    asr: AsrEngine | None = None,
    tagger: TaggerEngine | None = None,
    engine: SeparatorEngine | None = None,
) -> LocateResult:
    """§14.2 fuzzy lyric resolver: fetch (cache-first, transcribing on a genuine miss) the
    file's :class:`LyricsResult` via :func:`transcribe_lyrics`, then run the pure scorer
    (``analysis.lyrics.locate_lyric``) — normalize, slide a ± ``CFG.LYRIC_MATCH_WINDOW_SLOP``
    window over the flat word stream, score by token-level edit distance with phonetic +
    masked-profanity equivalence, non-max-suppress overlapping candidates. On a cache HIT this
    only reads the cached index and runs the pure scorer — no separation/ASR stack needed at
    all (registered unconditionally in the librosa tier, §9); a genuine cache MISS without the
    needed stack surfaces the SAME typed errors ``transcribe_lyrics`` does, never a bare
    ``ImportError``. Does not modify the input file or any REAPER project (``readOnlyHint``)."""
    from orpheus_mcp.analysis.lyrics import locate_lyric as _locate_lyric_pure

    fs = cache if cache is not None else CacheFS()
    lyrics = transcribe_lyrics(file, cache=fs, asr=asr, tagger=tagger, engine=engine)
    return _locate_lyric_pure(lyrics, quote, occurrence)


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:  # noqa: ARG001 - uniform signature
    """Registers the ``reference`` category. Probes for librosa BEFORE registering any
    tool: every tool in this category needs it, and ``registry.register_tools``'s
    ``except ImportError`` contract is how optional-dependency categories are skipped
    (with a warning) instead of being advertised and then raising at call time (house
    registry-honesty rule, see registry.py's HONESTY RULE docstring).

    ``instrument_timeline`` additionally needs panns-inference + torch (also part of the
    [analysis] extra, but a heavier ask than librosa alone) — that probe is per-TOOL, not
    per-category: a box with librosa but no torch still gets ``recon_reference`` registered
    (gating the whole category on panns would kill ``recon_reference`` everywhere it isn't
    needed, which the codebase-honesty rule forbids just as much as advertising a tool that
    raises at call time)."""
    import importlib.util

    if importlib.util.find_spec("librosa") is None:
        raise ImportError("reference toolset needs the [analysis] extra (librosa)")

    @mcp.tool(name="recon_reference", annotations=_RO)
    def recon_reference_tool(file: str) -> Recon:
        """T1 recon: LUFS, tempo/beat grid + downbeat/meter + grid confidence, tuning
        offset, key + alternates, structure sections, and an 8-band energy sketch.

        Decodes ``file`` (mp3/wav/flac/m4a/ogg/aiff), caches the result under the
        decoded-PCM hash so repeat calls on the same audio never re-run librosa. Does not
        modify the input file or any REAPER project (``readOnlyHint``) — it does write to
        the local analysis cache (``~/.orpheus_cache``).
        """
        return recon_reference(file)

    @mcp.tool(name="get_chords", annotations=_RO)
    def get_chords_tool(file: str, region: Window | None = None) -> ChordsLane:
        """§13 chord lane: beat-synchronous (or, on a low-confidence beat grid, fixed 1 s
        window) template+Viterbi chord recognition, refined with extensions (add9, 7,
        maj7, 6/13, #11), bass-informed slash/inversion detection, and a music21 theory
        layer (Roman numerals + local key windows). CPU-only, always available (no model
        dependency — the always-on correctness floor the §13 model lane later upgrades).

        Decodes ``file``, caches the WHOLE-SONG lane under the decoded-PCM hash so repeat
        calls on the same audio never re-run the recognizer. ``region``, if given, filters
        the cached lane to segments overlapping it (clamped to ``[0, duration]`` — a region
        reaching past EOF clamps rather than errors, §3.1d); segments are returned whole,
        never re-cut to the region's exact edges. Does not modify the input file or any
        REAPER project (``readOnlyHint``) — it does write to the local analysis cache
        (``~/.orpheus_cache``).
        """
        return get_chords(file, region)

    @mcp.tool(name="verify_stem", annotations=_RO)
    def verify_stem_tool(
        stem_path: str,
        mix_path: str,
        region: Window | None = None,
        timeline: Timeline | None = None,
    ) -> GateSet:
        """Re-verify an already-separated stem on disk against its source mix over ``region``,
        returning the §5 gate set (mix_consistency / bleed / silence_honesty / agreement +
        overall). This is the lightweight librosa-tier verifier: it computes silence honesty
        when you pass the mix's ``timeline`` (from ``instrument_timeline``) and reports the
        other three gates as N/A (mix consistency needs every co-produced stem, bleed needs a
        PANNs run on the stem, agreement needs a second engine's take — all owned by
        ``separate_region``). With no ``timeline`` every gate is N/A and ``overall`` is
        ``unverifiable`` — an honest "nothing to check at this tier". Does not modify the input
        files or any REAPER project (``readOnlyHint``).
        """
        return verify_stem(stem_path, mix_path, region, timeline)

    @mcp.tool(name="characterize", annotations=_RO)
    def characterize_tool(source: str, region: Window | None = None) -> SoundCharacter:
        """§7 sound character: post-FX spectral (centroid/rolloff/tilt + sketch & rule bands),
        dynamics (LUFS/crest/attack/kurtosis), harmonics (even/odd flavor), stereo width +
        correlation (N/A for mono/dual-mono sources — never fabricated), time-based FX (delay
        time + feedback, modulation rate/depth/class, reverb wet−dry or decay-heuristic), and
        pitch (melodic sources only). Also stamps the source's measured audio bandwidth (the
        spectral cliff).

        Decodes ``source`` (optionally cropped to ``region``), caches the result under the
        decoded-PCM hash + descriptor kind so repeat calls on the same audio never re-run the
        DSP. CPU-only (librosa tier); the vocal dereverb wet−dry path is used when the
        [separation] extra is present and degrades to the reverb decay heuristic (flagged in
        the result's notes) when it is not. Does not modify the input audio or any REAPER
        project (``readOnlyHint``) — it does write to the local analysis cache
        (``~/.orpheus_cache``).
        """
        return characterize(source, region)

    @mcp.tool(name="locate_lyric", annotations=_RO)
    def locate_lyric_tool(
        file: str, quote: str, occurrence: int | None = None
    ) -> LocateResult:
        """§14.2 fuzzy lyric resolver: find where ``quote`` is sung in ``file``. Normalizes
        both sides (case/punctuation/whitespace), slides the quote over the transcribed word
        stream (a quote may span a line boundary), and scores each candidate window by
        token-level edit distance with phonetic (double-metaphone-class) and masked-profanity
        (``f***``) token equivalence — so a misheard/misremembered word doesn't tank the match.
        Returns EVERY occurrence scoring ≥ ``CFG.LYRIC_MATCH_MIN_SCORE`` (a repeated chorus
        line returns all of them, each carrying its 1-based TEMPORAL ``occurrence`` — "the
        second chorus" — independent of score rank), or, if the best score falls short of that
        bar, the top-3 candidates with ``low_confidence=True`` so the caller asks the user
        instead of guessing. ``occurrence``, if given, filters to that one occurrence.
        ``status="no_vocals"`` is the DEFINED result for a file whose vocal stem never crossed
        the §14.1 VAD gate — distinct from an ordinary empty match list.

        Registered here in the librosa tier (not gated on panns/torch/faster-whisper): a
        cached lyric index is queryable with ONLY the base extra; a genuine cache MISS without
        the needed stack raises a typed, documented error, never a bare ``ImportError``. Does
        not modify the input file or any REAPER project (``readOnlyHint``) — it does write to
        the local analysis cache (``~/.orpheus_cache``) on a MISS.
        """
        return locate_lyric(file, quote, occurrence)

    _panns_available = (
        importlib.util.find_spec("panns_inference") is not None
        and importlib.util.find_spec("torch") is not None
    )
    if _panns_available:

        @mcp.tool(name="instrument_timeline", annotations=_RO)
        def instrument_timeline_tool(file: str) -> Timeline:
            """T2 instrument-activity timeline: per-second PANNs SED activations for the
            registry's mapped instrument classes (§9), raw and un-thresholded — callers
            apply ``CFG.TIMELINE_ACTIVITY_THRESHOLD`` / ``CFG.EXPOSURE_THRESHOLD``
            themselves (thresholds are never baked into this tool's result).

            Decodes ``file``, caches the result under the decoded-PCM hash + tagger
            identity so repeat calls never re-run the tagger. Does not modify the input
            file or any REAPER project (``readOnlyHint``) — it does write to the local
            analysis cache (``~/.orpheus_cache``).
            """
            return instrument_timeline(file)

        @mcp.tool(name="separate_region", annotations=_RO)
        def separate_region_tool(
            file: str,
            target: str,
            region: Window | None = None,
            model_id: str | None = None,
        ) -> StemResult:
            """Separate ``target`` from ``file`` (optionally cropped to ``region``) and
            verify it: resolves the rank-1 (or explicit ``model_id``) registry entry that
            produces ``target``, runs the region-cropped/VRAM-guarded/length-aligned
            separation engine (audio-separator or the vendored SCNet path), then runs §5
            verification gates 1-3 (mix consistency, bleed via the T2 tagger run ON the
            stem, silence honesty) — gate 4 (cross-model agreement) is not implemented
            yet and always reads N/A. Returns the requested target's stem with ``gates``
            populated; the stem FLAC + a ``verify.json`` are cached under the decoded-PCM
            hash so repeat calls for the same (file, model, region) never re-run
            separation or re-gate.

            **Desktop-tier**: in addition to this category's [analysis] extra (librosa +
            panns-inference/torch, needed for the bleed judge and gated by this
            registration probe), this tool needs the [separation] extra (audio-separator
            / vendored MSST) at CALL time — a missing isolated separator env or an
            unsupported checkpoint raises a clear, typed error rather than a silent
            failure. Does not modify the input file or any REAPER project
            (``readOnlyHint``) — it does write to the local analysis cache
            (``~/.orpheus_cache``).
            """
            return separate_region(file, target, region, model_id)

        _faster_whisper_available = importlib.util.find_spec("faster_whisper") is not None
        if _faster_whisper_available:

            @mcp.tool(name="transcribe_lyrics", annotations=_RO)
            def transcribe_lyrics_tool(file: str) -> LyricsResult:
                """§14 lyric index: resolve the vocal stem through the §6 escalation ladder
                (``max_tier="T3"`` — the ``lyric_navigate`` question-class ceiling; a single
                validated rank-1 separation, cached once per §3.1 and shared with the ladder's
                own cache entry), then run faster-whisper ASR (word timestamps) gated by the
                §14.1 VAD hallucination gate (a stem that never crosses it returns
                ``status="no_vocals"`` WITHOUT ASR ever running), whisperX forced alignment,
                and line/section assembly (repeated blocks → named ``chorus``/``verse``
                sections; through-composed → structural ``section A/B/C``).

                Decodes ``file``, caches the result under the decoded-PCM hash across three
                kinds (``lyrics``/``lines``/``sections``, §9) so repeat calls never re-run ASR.

                **Desktop-tier**: needs the ``[separation]`` extra (audio-separator / vendored
                MSST, for the vocal stem, AND faster-whisper, gated by this registration probe)
                at CALL time — a missing isolated separator env or ASR install raises a clear,
                typed error rather than a silent failure or a bare ``ImportError``. Does not
                modify the input file or any REAPER project (``readOnlyHint``) — it does write
                to the local analysis cache (``~/.orpheus_cache``).
                """
                return transcribe_lyrics(file)
    else:
        import warnings

        warnings.warn(
            "instrument_timeline not registered: needs panns-inference + torch "
            "(pip install 'orpheus-mcp[analysis]'); recon_reference remains available.",
            stacklevel=2,
        )
