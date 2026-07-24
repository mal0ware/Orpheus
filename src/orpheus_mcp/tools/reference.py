"""The ``reference`` category (§9): local reference-track analysis tools. Tools:
``recon_reference`` (T1), ``instrument_timeline`` (T2), ``get_chords`` (§13). Logic lives in
``orpheus_mcp.analysis`` (pure functions, no REAPER); this layer only handles
decode -> cache -> tool-result plumbing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from orpheus_mcp.analysis import cachefs
from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
from orpheus_mcp.models import BeatGrid, ChordsLane, Recon, Timeline, Window

if TYPE_CHECKING:
    from orpheus_mcp.analysis.timeline import TaggerEngine

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
    else:
        import warnings

        warnings.warn(
            "instrument_timeline not registered: needs panns-inference + torch "
            "(pip install 'orpheus-mcp[analysis]'); recon_reference remains available.",
            stacklevel=2,
        )
