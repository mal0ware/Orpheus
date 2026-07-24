"""The ``reference`` category (§9): local reference-track analysis tools. Tools:
``recon_reference`` (T1), ``instrument_timeline`` (T2). Logic lives in
``orpheus_mcp.analysis`` (pure functions, no REAPER); this layer only handles
decode -> cache -> tool-result plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import FastMCP

from orpheus_mcp.analysis import cachefs
from orpheus_mcp.analysis.cachefs import CacheFS
from orpheus_mcp.models import Recon, Timeline

if TYPE_CHECKING:
    from orpheus_mcp.analysis.timeline import TaggerEngine

_RO = {"readOnlyHint": True}

_RECON_KIND = "recon"
_RECON_SCHEMA_VERSION = 1

_TIMELINE_KIND = "timeline"
_TIMELINE_SCHEMA_VERSION = 1


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


def recon_reference(file: str, *, cache: CacheFS | None = None) -> Recon:
    """Decode ``file``, run the T1 recon pass, and cache the result under kind
    ``"recon"`` (keyed by the decoded-PCM hash + this artifact's meta shape). A cache HIT
    never re-decodes or re-runs recon (§9 T0).

    Every call to ``cachefs.decode``/``cachefs.pcm_hash`` below is via the module (not a
    bare import) so a test can monkeypatch ``cachefs.decode`` and see every decode this
    function triggers, directly or through ``pcm_hash``'s own memoized decode."""
    from orpheus_mcp.analysis.recon import run_recon

    fs = cache if cache is not None else CacheFS()
    file_hash = cachefs.pcm_hash(file)  # memoized by file-bytes-hash — free on repeat calls
    meta = _recon_meta(file)

    hit = fs.read_json(file_hash, _RECON_KIND, meta=meta)
    if hit is not None:
        return Recon.model_validate(hit)

    pcm = cachefs.decode(file)
    recon = run_recon(pcm)
    fs.write_json(file_hash, _RECON_KIND, recon.model_dump(mode="json"), meta=meta)
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
