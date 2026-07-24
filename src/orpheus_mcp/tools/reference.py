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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from fastmcp import FastMCP

from orpheus_mcp.analysis import cachefs, model_registry
from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
from orpheus_mcp.analysis.separation import SeparationError
from orpheus_mcp.analysis.separation import separate_region as _run_separation_core
from orpheus_mcp.models import (
    BeatGrid,
    ChordsLane,
    GateResult,
    GateSet,
    Recon,
    StemResult,
    Timeline,
    Window,
)

if TYPE_CHECKING:
    from orpheus_mcp.analysis.model_registry import ModelEntry
    from orpheus_mcp.analysis.separation import SeparatorEngine
    from orpheus_mcp.analysis.timeline import TaggerEngine

__all__ = [
    "recon_reference",
    "instrument_timeline",
    "get_chords",
    "separate_region",
    "verify_stem",
    "register",
]

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
    else:
        import warnings

        warnings.warn(
            "instrument_timeline not registered: needs panns-inference + torch "
            "(pip install 'orpheus-mcp[analysis]'); recon_reference remains available.",
            stacklevel=2,
        )
