"""The ``reference`` category (§9): local reference-track analysis tools. Tools:
``recon_reference`` (T1), ``instrument_timeline`` (T2), ``get_chords`` (§13),
``separate_region`` (Task 4.2, §5/§9 — engine adapters + verification gates). Logic lives
in ``orpheus_mcp.analysis`` (pure functions, no REAPER); this layer only handles
decode -> cache -> tool-result plumbing.
"""

from __future__ import annotations

import contextlib
import json
import re
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
from orpheus_mcp.analysis.router import analyze_reference
from orpheus_mcp.analysis.separation import SeparationError
from orpheus_mcp.analysis.separation import separate_region as _run_separation_core
from orpheus_mcp.models import (
    AnswerEnvelope,
    CacheClearResult,
    CharacterDiff,
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
    VstEntry,
    VstSuggestions,
    Window,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from orpheus_mcp.analysis.harmony import ChordModel
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
    "compare_character",
    "transcribe_lyrics",
    "locate_lyric",
    "analyze_reference",
    "suggest_free_vst",
    "clear_analysis_cache",
    "load_vst_catalog",
    "VstCatalogError",
    "register",
]


class ReferenceEnvMissing(Exception):
    """A tool call needs the ``[analysis]`` extra (panns-inference + torch) that isn't
    installed. Raised instead of letting a raw ``ImportError`` escape a tool call — mirrors
    ``analysis.separation.SeparatorEnvMissing`` / ``analysis.lyrics.LyricsAsrEnvMissing`` for
    the base analysis stack ``instrument_timeline``/the §6 ladder needs. A future orchestrator
    (§9 ``analyze_reference``) catches this and records ``capped_by_missing_extra="analysis"``
    in the answer envelope."""


class VstCatalogError(Exception):
    """A malformed ``data/vst_catalog.json`` — a missing/unsupported ``schema_version`` or a
    structurally invalid entry. Raised at load time (fail fast), mirroring
    ``analysis.intents.RulesError``'s contract for ``intent_rules.json``."""

_RO = {"readOnlyHint": True}
_DESTRUCTIVE = {"destructiveHint": True}

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


def _chords_meta(file: str, produced_by: str) -> dict:
    """The cache sidecar shape for kind ``"chords"``. ``produced_by`` names the lane that
    produced the artifact — ``"baseline"`` for the always-available librosa lane, or the §13
    model entry id (``"chord/btc-large-voca"``) when the model lane resolved and re-wrote the
    baseline at T3. A model-lane result and a baseline result for the same audio must not
    collide in the cache (a box that gains/loses the model, or swaps its checkpoint, must
    re-compute), the same way ``instrument_timeline``'s ``tagger_id`` keys the timeline cache;
    ``registry_version`` additionally busts on any checkpoint re-pin."""
    return {
        "schema_version": _CHORDS_SCHEMA_VERSION,
        "registry_version": cachefs.registry_version(),
        "verify_config_hash": cachefs.verify_config_hash(),
        "decoder": cachefs.decoder_for(file),
        "produced_by": produced_by,
    }


_CHORD_TASK = "chord"
_CHORD_TARGET = "chord"


def _btc_checkpoint_path(entry: ModelEntry) -> Path | None:
    """The local path the ``btc`` checkpoint lands at (``fetch-models``), or ``None`` for an
    entry with no single-file ``checkpoint_url`` — a cheap file-stat probe (no torch load) so
    ``_resolve_chord_model_entry`` can gate on checkpoint presence without paying to build the
    model on a cache HIT."""
    from orpheus_mcp.analysis.fetch_models import _filename_for_url

    url = entry.checkpoint_url
    if not isinstance(url, str):
        return None
    return model_registry.checkpoint_dir(entry.id) / _filename_for_url(url, entry.id)


def _resolve_chord_model_entry() -> ModelEntry | None:
    """The rank-1 §13 MODEL entry usable RIGHT NOW, or ``None`` (→ the baseline lane). This is
    the "no phantom tier" gate (§13 placement): ``resolve`` already drops the internal baseline's
    non-model peers by VRAM (BTC's 2 GB entry is excluded when ``free_vram_gb()`` is ``None`` — a
    CPU-only box — leaving only the ``vram_gb==0`` internal baseline, which is NOT a model here)
    and the no-engine rank-2 musicxlab entry (D12). We additionally require the checkpoint to be
    present on disk, so a box that resolves to BTC but never fetched it degrades honestly to the
    baseline rather than advertising a model tier it can't run."""
    free = model_registry.free_vram_gb()
    candidates = model_registry.resolve(_CHORD_TASK, _CHORD_TARGET, vram_free_gb=free)
    for entry in candidates:  # rank order
        if entry.engine == "internal":
            continue  # the baseline lane, not a model tier
        if entry.engine == "btc":
            ckpt = _btc_checkpoint_path(entry)
            if ckpt is not None and ckpt.is_file():
                return entry
    return None


def _build_chord_models(entry: ModelEntry) -> list[ChordModel]:
    """Construct the model-lane recognizers for ``entry`` (rank-1 first). Only ``btc`` is
    buildable today (rank-2 ``musicxlab`` is a no-engine skeleton, D12 — its cross-check runs
    against a scripted stub, never a fabricated checkpoint), so this returns ``[BtcChordModel]``
    or ``[]`` on any load failure (degrading to the baseline the caller already holds). Building
    a model LOADS torch + the checkpoint, so this is only ever called on a cache MISS."""
    from orpheus_mcp.analysis.harmony import BtcChordModel

    if entry.engine != "btc":
        return []
    ckpt = _btc_checkpoint_path(entry)
    if ckpt is None or not ckpt.is_file():
        return []
    try:
        return [BtcChordModel(str(ckpt))]
    except Exception:  # noqa: BLE001 - any torch/checkpoint load failure degrades to baseline
        return []


def _find_cached_bass_stem(fs: CacheFS, pcm_hash: str) -> str | None:
    """A cached full-ish bass stem for this audio, or ``None`` (§13.2 preference). Searches the
    §9 ``stems/<model-id>/<start_ms>-<end_ms>/bass.flac`` layout under this pcm-hash and prefers
    the WIDEST region (a whole-song separation over a narrow region crop), so a chord lane that
    covers the whole song refines its bass against the most-complete stem available."""
    stems_root = fs.root / pcm_hash / "stems"
    if not stems_root.is_dir():
        return None
    candidates = list(stems_root.rglob("bass.flac"))
    if not candidates:
        return None

    def _region_span(p: Path) -> int:
        try:
            start_ms, end_ms = p.parent.name.split("-")
            return int(end_ms) - int(start_ms)
        except (ValueError, AttributeError):
            return 0

    candidates.sort(key=_region_span, reverse=True)
    return str(candidates[0])


def _bass_pc_lane_for(pcm: Pcm, recon: Recon, fs: CacheFS) -> list[int | None]:
    """The §13.2 per-frame bass pitch-class lane, PREFERRING a cached bass STEM when a
    separation exists (pYIN on the isolated bass is far cleaner than on the low-passed mix),
    falling back to the low-passed-mix pYIN lane otherwise. The frame grid matches
    ``compose_lane``'s (recon beat grid when confident, else fixed 1 s windows) so the returned
    lane indexes align with the segments ``detect_slash`` refines."""
    from orpheus_mcp.analysis.harmony import _fixed_frame_grid, bass_pc_lane

    confident = recon.grid.confidence >= CFG.BEAT_GRID_CONFIDENCE_MIN
    frame_grid = recon.grid if confident else _fixed_frame_grid(pcm)

    stem_path = _find_cached_bass_stem(fs, pcm.pcm_hash)
    if stem_path is not None:
        try:
            return bass_pc_lane(cachefs.decode(stem_path), frame_grid)  # STEM preferred (§13.2)
        except cachefs.DecodeError:
            pass  # a corrupt/unreadable cached stem falls through to the mix lane
    return bass_pc_lane(pcm, frame_grid)  # the no-separation low-passed-mix path


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
    file: str,
    region: Window | None = None,
    *,
    cache: CacheFS | None = None,
    models: Sequence[ChordModel] | None = None,
) -> ChordsLane:
    """Decode ``file``, compute the full §13 chord lane (recognizer + extensions + [T3 model
    re-write] + slash + theory via ``harmony.compose_lane``), and cache the WHOLE-SONG lane
    under kind ``"chords"`` (keyed by the decoded-PCM hash + this artifact's meta shape, which
    includes the producing lane's identity). A cache HIT never re-decodes or re-computes the
    lane (§9 T0); recon is reused via the shared ``_recon_read_through`` read-through so a recon
    already cached by ``recon_reference`` is never recomputed.

    The §13 MODEL lane (vendored BTC) engages automatically at T3 when the registry resolves a
    runnable chord model for this box (CUDA + VRAM + the checkpoint fetched — see
    ``_resolve_chord_model_entry``); otherwise the always-available baseline lane is returned,
    with no phantom model tier (§13 placement). The §13.2 slash step prefers a cached bass STEM
    when one exists. ``models`` is a test seam (an explicit sequence of ``ChordModel``s bypassing
    registry resolution); production callers leave it ``None``.

    ``region``, if given, filters the (cached or freshly computed) lane to segments
    overlapping ``[region.start_s, min(region.end_s, duration)]`` — clamped per §3.1d —
    AFTER the cache read/write, so the cache always holds the whole song."""
    from orpheus_mcp.analysis.harmony import compose_lane

    fs = cache if cache is not None else CacheFS()
    file_hash = cachefs.pcm_hash(file)

    if models is not None:
        entry = None
        produced_by = "model:injected"
    else:
        entry = _resolve_chord_model_entry()
        produced_by = entry.id if entry is not None else "baseline"
    meta = _chords_meta(file, produced_by)

    duration_s: float | None
    hit = fs.read_json(file_hash, _CHORDS_KIND, meta=meta)
    if hit is not None:
        lane = ChordsLane.model_validate(hit)
        duration_s = None
    else:
        pcm = cachefs.decode(file)
        recon, _ = _recon_read_through(file, fs, pcm=pcm)
        chord_models = (
            list(models)
            if models is not None
            else (_build_chord_models(entry) if entry is not None else [])
        )
        bass_pc = _bass_pc_lane_for(pcm, recon, fs)
        lane = compose_lane(pcm, recon, models=chord_models or None, bass_pc=bass_pc)
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


def compare_character(
    a: str | StemHandle,
    b: str | StemHandle,
    inventory: list[str] | None = None,
    *,
    cache: CacheFS | None = None,
) -> CharacterDiff:
    """§9/§15.3 ``compare_character``: measure (via :func:`characterize`, cache-first) both
    ``a`` (the REFERENCE) and ``b`` (the USER/render) sources, then run the §15.3 rules engine
    (``analysis.intents``) over the §8-normalized diff. Intent generation lives in the rules
    engine ONLY — this tool returns whatever it fired, Claude reads them but never re-derives
    them (§9 contract).

    Every signed descriptor delta is ``a − b`` (reference minus user). Spectral band diffs are
    restricted to ``min(a, b)``'s measured bandwidth (§8 mandatory normalization — recorded in
    the result's ``band_limit_hz``/``notes`` so a codec-limited reference is never misread as
    the render being 'darker'). ``vocal_only`` rules (rule 13, de-ess) are skipped, with a note,
    unless BOTH sides are a ``"vocals"`` stem. A §6 R3 ``complement=True`` source on either side
    forces the tail-sensitive rules (11 delay, 12 reverb) to recommend-only, with a discount
    note, even if they otherwise fired.

    ``inventory`` (optional — pass ``list_installed_fx``'s result), when given, fills each
    fired intent's ``installed_matches`` with the caller's own plugins whose name substring-
    matches (case-insensitive) that intent's keyword table, so recommendations name the user's
    own gear before the stock/catalog fallback (§8 step 2.1). ``inventory=None`` leaves every
    intent's ``installed_matches`` empty — the stock apply path is unaffected either way.

    Does not modify the input audio or any REAPER project (``readOnlyHint``) — it does write to
    the local analysis cache (``~/.orpheus_cache``), via the same ``characterize`` cache both
    sides go through.
    """
    from orpheus_mcp.analysis.intents import diff_characters, fire_rules, match_inventory

    fs = cache if cache is not None else CacheFS()
    char_a = characterize(a, cache=fs)
    char_b = characterize(b, cache=fs)
    diff = diff_characters(char_a, char_b)
    intents = fire_rules(diff)
    match_inventory(intents, inventory)
    return CharacterDiff(
        diffs=diff.diffs, intents=intents, band_limit_hz=diff.band_limit_hz, notes=diff.notes
    )


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
_VOCALS_TARGET = "vocals"


def _asr_model_id() -> str:
    """The rank-1 ``lyrics-asr``/``fasterwhisper`` registry entry's id, or ``"unknown"`` if
    the registry ships none — this is CHEAP (registry JSON read, no import of faster_whisper
    itself) so it can be part of the cache meta on every call, including a HIT (§9: meta must
    be buildable before paying for a decode). ``_LYRICS_TASK`` is imported from
    ``analysis.lyrics`` (fix wave 1: dedup) rather than redefined here — that module owns the
    task-id constant, and a redefinition here could silently drift out of sync with it."""
    from orpheus_mcp.analysis.lyrics import _LYRICS_TASK

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


# --------------------------------------------------------------------------- #
# suggest_free_vst (Task 12.2, §15.4) — pure data lookup over data/vst_catalog.json
# --------------------------------------------------------------------------- #

_VST_CATALOG_SCHEMA_VERSION = 1


def _vst_catalog_path() -> Path:
    """Locate ``data/vst_catalog.json`` in both layouts (installed wheel: inside the
    package; dev checkout: repo root next to ``src/``) — same two-layout probe as
    ``analysis.intents._rules_path`` / ``analysis.model_registry.registry_path``."""
    package_path = Path(__file__).resolve().parents[1] / "data" / "vst_catalog.json"
    if package_path.is_file():
        return package_path
    return Path(__file__).resolve().parents[3] / "data" / "vst_catalog.json"


def load_vst_catalog(path: str | Path | None = None) -> list[VstEntry]:
    """Parse + validate ``data/vst_catalog.json`` (or an explicit ``path``) into
    :class:`~orpheus_mcp.models.VstEntry` objects — §15.4's curated free-VST catalog, §8.1's
    instruments + effects tables transcribed verbatim (every row: name, vendor, category,
    license, ``account_required``, url, the intents it serves). Raises
    :class:`VstCatalogError` on a missing/unreadable file, a missing/unsupported
    ``schema_version``, or a structurally malformed entry — fail fast at load time, never a
    silently-dropped row."""
    p = Path(path) if path is not None else _vst_catalog_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VstCatalogError(f"cannot read catalog file {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VstCatalogError(f"malformed JSON in catalog file {p}: {exc}") from exc

    schema_version = data.get("schema_version") if isinstance(data, dict) else None
    if schema_version != _VST_CATALOG_SCHEMA_VERSION:
        raise VstCatalogError(
            f"{p}: missing or unsupported 'schema_version' (expected "
            f"{_VST_CATALOG_SCHEMA_VERSION}, got {schema_version!r})."
        )

    raw_entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(raw_entries, list) or not raw_entries:
        raise VstCatalogError(f"{p}: expected a non-empty top-level 'entries' list.")

    entries: list[VstEntry] = []
    for raw in raw_entries:
        try:
            entries.append(VstEntry.model_validate(raw))
        except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, wrapped uniformly
            raise VstCatalogError(f"{p}: malformed catalog entry {raw!r}: {exc}") from exc
    return entries


def suggest_free_vst(
    need: str, limit: int = 3, *, catalog: list[VstEntry] | None = None
) -> VstSuggestions:
    """§15.4 curated free-VST catalog lookup (closes Q7): a PURE data lookup over
    ``data/vst_catalog.json`` — this function makes NO network call, ever, and never
    downloads or installs anything (the standing recommend-only decision from the original
    design, §8 step 2/§15.4). Splits ``need`` into lowercase whitespace tokens and matches
    each against every entry's ``intents`` + ``category`` + ``name`` (case-insensitive
    substring match); entries are ranked by TOKEN MATCH COUNT descending (ties keep catalog
    order), and the top ``limit`` go into the result's ``entries``.

    Recommend-only: the returned entries carry ``url``/``license``/``account_required`` so
    Claude can hand the user a link and an honest account-wall flag, never a
    download/install action of Orpheus's own. ``catalog`` is a test seam (an explicit
    pre-loaded entry list, bypassing the file read); a production call leaves it ``None``
    and loads ``data/vst_catalog.json`` fresh via :func:`load_vst_catalog`.

    ``notes`` is empty whenever ``entries`` is non-empty. On a ZERO-match ``need`` (no
    tokens at all, or no catalog entry scored above zero), ``entries`` is ``[]`` and
    ``notes`` carries exactly one actionable note naming ``need`` and the catalog's
    available categories, instead of a bare empty list the caller has to interpret itself."""
    entries = catalog if catalog is not None else load_vst_catalog()
    tokens = [t for t in re.split(r"\s+", need.strip().lower()) if t]

    scored: list[tuple[int, int, VstEntry]] = []
    if tokens:
        for idx, entry in enumerate(entries):
            haystack = " ".join([entry.name, entry.category, *entry.intents]).lower()
            score = sum(1 for tok in tokens if tok in haystack)
            if score > 0:
                scored.append((score, idx, entry))
        scored.sort(key=lambda item: (-item[0], item[1]))

    matched = [entry for _, _, entry in scored[:limit]]
    if matched:
        return VstSuggestions(entries=matched)

    categories = sorted({entry.category for entry in entries})
    note = (
        f"no catalog entry matched '{need}' — try broader terms "
        f"(categories: {', '.join(categories)})"
    )
    return VstSuggestions(entries=[], notes=[note])


# --------------------------------------------------------------------------- #
# clear_analysis_cache (Task 12.2, §9) — wraps CacheFS.clear, destructiveHint
# --------------------------------------------------------------------------- #


def clear_analysis_cache(*, cache: CacheFS | None = None) -> CacheClearResult:
    """§9 ``clear_analysis_cache``: wipe every cached ANALYSIS artifact under the local
    analysis cache (``~/.orpheus_cache`` by default — every cached recon/timeline/chords/
    lyrics/stems/descriptors artifact across every analyzed file, plus the file-bytes hash
    memo) via :meth:`~orpheus_mcp.analysis.cachefs.CacheFS.clear`. Destructive (unlike every
    other tool in this module) — the NEXT call to any reference tool re-decodes and
    re-computes from scratch. Measures ``bytes_freed``/``entries_cleared`` BEFORE clearing
    (so the result is honest even though the directories are gone by the time it returns),
    counting through :meth:`CacheFS.entry_dirs`/``entry_count`` — the SAME filter
    ``fs.clear()`` itself uses, so this count can never drift out of sync with what actually
    gets deleted.

    NEVER touches ``checkpoints/`` (downloaded model weights, ``model_registry.
    checkpoint_dir()`` — an install-time tool artifact, not an analysis artifact §9 owns):
    that directory is excluded at the source, ``CacheFS.entry_dirs()`` — see
    ``cachefs._RESERVED_TOP_LEVEL``."""
    fs = cache if cache is not None else CacheFS()
    entries_cleared = fs.entry_count()
    bytes_freed = fs.size_bytes()
    fs.clear()
    return CacheClearResult(bytes_freed=bytes_freed, entries_cleared=entries_cleared)


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

        Decodes ``file`` (mp3/wav/flac/m4a/ogg/aiff, or an .mp4 music video — the
        audio stream is extracted, video discarded), caches the result under the
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

    @mcp.tool(name="compare_character", annotations=_RO)
    def compare_character_tool(
        a: str, b: str, inventory: list[str] | None = None
    ) -> CharacterDiff:
        """§15.3 compare two characterized sources: ``a`` (the REFERENCE) vs ``b`` (the USER/
        render), §8-normalized (spectral diffs restricted to the narrower source's measured
        bandwidth — recorded in ``band_limit_hz``/``notes``), and run the 14-rule §15.3 engine
        over the diff. Every signed delta is ``a − b`` (reference minus user). ``vocal_only``
        rules skip (with a note) unless both sides are a ``"vocals"`` stem; a §6 R3
        ``complement=True`` source forces the tail-sensitive delay/reverb rules to
        recommend-only. Pass ``inventory`` (``list_installed_fx``'s result) to have fired
        intents name the user's own plugins first (§8 step 2.1) — omit it for the stock-only
        path. Intent generation lives in this rules engine ONLY; Claude reads the result, it
        does not re-derive intents. Does not modify the input audio or any REAPER project
        (``readOnlyHint``) — it does write to the local analysis cache (``~/.orpheus_cache``).

        **§8 step 3 closed FX loop (D5):** when Claude applies an intent, renders
        (``render_track_region``), and re-calls this tool to check convergence, it MUST
        respect the two loop guards in :data:`~orpheus_mcp.analysis.config.CFG` —
        ``FX_LOOP_MAX_ITERATIONS`` (stop after this many apply/render/re-compare rounds even
        if the diff hasn't converged) and ``FX_LOOP_PLATEAU_EPSILON`` (stop early once the
        diff's per-iteration improvement falls below this fraction — a plateau, not
        convergence). This tool does not enforce either guard itself (it has no memory of
        prior calls); Claude is the loop.
        """
        return compare_character(a, b, inventory)

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

    @mcp.tool(name="analyze_reference", annotations=_RO)
    def analyze_reference_tool(
        file: str,
        question_focus: str | None = None,
        region: Window | None = None,
        max_tier: str | None = None,
        max_seconds: float | None = None,
        target: str | None = None,
        quote: str | None = None,
        occurrence: int | None = None,
    ) -> AnswerEnvelope:
        """§9 orchestrator — route a question about ``file`` to its §3.1 cost tier, run only the
        stages that class needs, and return the answer envelope (``question_class``,
        ``tier_reached``, ``recon_summary``, the needed ``artifacts`` groups, per-stem ``gates``,
        the §6 ``ladder_log``, and honesty flags ``budget_capped`` / ``capped_by_missing_extra``
        / ``notes``).

        ``question_focus`` (``None`` ⇒ ``"recon"``) is one of: ``recon`` (T1: key/tempo/grid/
        loudness), ``timeline`` / ``chords`` (T2), ``sound_at`` / ``compare_track`` (T3, needs a
        ``target`` instrument; +T4 only on §5 gate failure), ``lyric_navigate`` (needs a
        ``quote`` — answers with the instrument-activation diff of the ``[match.end_s, +8 bars]``
        window vs the preceding one), or ``notes`` (T5 polyphonic transcription — needs
        ``max_tier="T5"``). ``region`` scopes the question to a span; ``max_tier`` / ``max_seconds``
        cap cost. ``compare_track`` returns the REFERENCE side only and names the render/
        characterize/compare follow-up calls in ``notes`` (no REAPER-bridge calls here). A tier a
        missing extra can't reach degrades to the highest available tier with the extra named,
        never a hard error. Does not modify the input file or any REAPER project
        (``readOnlyHint``) — it does write to the local analysis cache (``~/.orpheus_cache``).

        ``compare_track`` is the entry point into the §8 step 3 closed FX loop: Claude reads
        this envelope's reference-side descriptors, then drives apply → render → re-compare
        (``compare_character``) rounds itself, bounded by :data:`~orpheus_mcp.analysis.config.CFG`'s
        ``FX_LOOP_MAX_ITERATIONS`` (hard stop) and ``FX_LOOP_PLATEAU_EPSILON`` (early stop on a
        plateau) — this tool runs once per call and holds no loop state of its own.
        """
        return analyze_reference(
            file,
            question_focus,
            region,
            max_tier,
            max_seconds,
            target,
            quote,
            occurrence,
        )

    @mcp.tool(name="suggest_free_vst", annotations=_RO)
    def suggest_free_vst_tool(need: str, limit: int = 3) -> VstSuggestions:
        """§15.4 curated free-VST catalog (closes Q7): keyword-match ``need`` (e.g. "cavern
        reverb", "tape saturation", "concert band strings") against ``data/vst_catalog.json``
        — §8.1's instruments + effects tables — ranked by token match count, top ``limit`` in
        the result's ``entries``. PURE data lookup: this tool makes no network call, ever, and
        Orpheus never downloads or installs a plugin on the user's behalf (standing
        recommend-only decision) — it only hands back names, vendors, licenses, URLs, and an
        honest ``account_required`` flag for the catalog's account/email/Patreon-gated
        exceptions. This is Step 2's source-3 fallback (§8): try the user's installed
        inventory (``list_installed_fx``) and REAPER stock/JSFX (the apply path) first; reach
        for this catalog only when neither covers the intent well. On a ZERO-match ``need``,
        ``entries`` is empty and ``notes`` carries one actionable note (try broader terms)
        instead of leaving Claude to interpret a bare empty list. Does not modify anything
        (``readOnlyHint``) and does not touch the analysis cache.
        """
        return suggest_free_vst(need, limit)

    @mcp.tool(name="clear_analysis_cache", annotations=_DESTRUCTIVE)
    def clear_analysis_cache_tool() -> CacheClearResult:
        """§9 wipe every cached ANALYSIS artifact under the local analysis cache
        (``~/.orpheus_cache``) — every cached recon/timeline/chords/lyrics/stems/descriptors
        artifact for every file analyzed so far, plus the file-bytes hash memo. Returns
        ``bytes_freed``/``entries_cleared`` (measured before the wipe). Destructive
        (``destructiveHint``, unlike every other tool in this module): the next call to any
        reference tool re-decodes and re-computes from scratch. Does not touch REAPER or any
        input audio file — only the local cache.

        Downloaded MODEL CHECKPOINTS (``~/.orpheus_cache/checkpoints/``,
        ``model_registry.checkpoint_dir()``) are NEVER touched by this tool, at any cache
        size — they are install-time tool artifacts, not analysis cache, and are excluded
        from both the wipe and the ``entries_cleared``/``bytes_freed`` counts.
        """
        return clear_analysis_cache()

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
