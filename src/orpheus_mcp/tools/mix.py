"""``mix`` category (§8 step 3, Task 11.1): the FX apply verbs the closed FX loop drives —
``add_fx_by_name``, ``set_fx_param``, ``get_fx_params``. The bridge handlers speak only
REAPER's native 0..1 normalized param space (locked contract,
``docs/plans/2026-07-23-reference-analysis.md`` Task 11.1); ALL unit translation (dB/Hz/ms/
ratio <-> normalized) happens HERE, against ``data/fx_param_maps.json``.

An unmapped plugin, or a mapped plugin missing the requested param, never guesses an index —
it returns the §8 step 3 human-instructions path instead: a structured note naming the
plugin, the param, and the requested value, for Claude to relay to the user as a manual
instruction ("Serum: unison 4, detune 0.15").

``apply_mix_balance`` (v1-stretch, genre×role dB table + chain-offset calibration) remains a
stub — unrelated to this task, still gated behind ``include_stubs``.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from orpheus_mcp.bridge import BridgeClient

_RO = {"readOnlyHint": True}
_DESTRUCTIVE = {"destructiveHint": True}

__all__ = [
    "FxParamMapError",
    "load_param_maps",
    "to_normalized",
    "from_normalized",
    "add_fx_by_name",
    "set_fx_param",
    "get_fx_params",
    "register",
]


class FxParamMapError(Exception):
    """A malformed ``data/fx_param_maps.json`` entry — an unknown ``to_normalized`` kind,
    or a ``min``/``max``/``index`` of the wrong shape. Raised at translation time (never a
    silently wrong normalized value)."""


def _param_maps_path() -> Path:
    """Locate ``data/fx_param_maps.json`` in both layouts (installed wheel: inside the
    package; dev checkout: repo root next to ``src/``) — same two-layout probe as
    ``analysis.intents._rules_path`` / ``analysis.config.calibrated_thresholds_path``."""
    package_path = Path(__file__).resolve().parents[1] / "data" / "fx_param_maps.json"
    if package_path.is_file():
        return package_path
    return Path(__file__).resolve().parents[3] / "data" / "fx_param_maps.json"


@lru_cache(maxsize=1)
def _load_param_maps_cached(path_str: str) -> dict[str, Any]:
    p = Path(path_str)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise FxParamMapError(f"{p}: expected a top-level JSON object.")
    return data


def load_param_maps(path: str | Path | None = None) -> dict[str, Any]:
    """Parse ``data/fx_param_maps.json`` (or an explicit ``path``): plugin name -> ``{notes,
    params: {param_name -> {index, unit, min, max, to_normalized, live_verified}}}``. Cached
    by resolved path (a test passing a distinct ``tmp_path`` file always gets a fresh parse;
    the shipped file is parsed once per process)."""
    p = Path(path) if path is not None else _param_maps_path()
    return _load_param_maps_cached(str(p))


def _plugin_entry(fx_name: str, maps: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The ``(plugin_key, plugin_entry)`` this REAPER-reported ``fx_name`` maps to, or
    ``None`` if no plugin in the map matches. Exact key match first (the common case —
    REAPER stock plugins report their bare name, e.g. ``"ReaEQ"``); falling back to a
    case-insensitive substring match (``fx_name`` may carry a vendor/category prefix or
    suffix, e.g. ``"VST: ReaEQ (Cockos)"``) so a live-REAPER-reported name still resolves
    without the map needing every vendor-string variant hand-listed. Never guesses across
    plugins — a substring match is only ever the plugin's own map key appearing in the
    reported name, not a fuzzy/edit-distance match."""
    plugins = {k: v for k, v in maps.items() if not k.startswith("_")}
    if fx_name in plugins:
        return fx_name, plugins[fx_name]
    lowered = fx_name.lower()
    for key, entry in plugins.items():
        if key.lower() in lowered:
            return key, entry
    return None


def to_normalized(value: float, entry: dict[str, Any]) -> float:
    """Native-unit ``value`` (dB/Hz/ms/ratio/...) -> REAPER's normalized 0..1, per one
    ``data/fx_param_maps.json`` param entry. ``"linear"``: ``(value - min) / (max - min)``.
    ``"log"`` (frequency-type params, tapered in octaves not linearly):
    ``log(value / min) / log(max / min)``. ``value`` is clamped to ``[min, max]`` before
    conversion (an out-of-range intent still applies at the nearest in-range knob position,
    rather than raising or silently wrapping); the result is clamped to ``[0, 1]`` as a
    final belt-and-suspenders (float edge cases only)."""
    lo, hi = float(entry["min"]), float(entry["max"])
    kind = entry.get("to_normalized", "linear")
    v = min(max(float(value), lo), hi)
    if kind == "linear":
        norm = 0.0 if hi == lo else (v - lo) / (hi - lo)
    elif kind == "log":
        if lo <= 0.0:
            raise FxParamMapError(f"log to_normalized needs min > 0, got min={lo}")
        norm = math.log(v / lo) / math.log(hi / lo)
    else:
        raise FxParamMapError(f"unknown to_normalized kind: {kind!r}")
    return min(max(norm, 0.0), 1.0)


def from_normalized(norm: float, entry: dict[str, Any]) -> float:
    """The inverse of :func:`to_normalized`: REAPER's normalized 0..1 -> the param's native
    unit, per one ``data/fx_param_maps.json`` entry. Used by ``get_fx_params`` to report the
    native value (dB/Hz/ms/...) alongside the raw normalized one the bridge returns."""
    lo, hi = float(entry["min"]), float(entry["max"])
    kind = entry.get("to_normalized", "linear")
    n = min(max(float(norm), 0.0), 1.0)
    if kind == "linear":
        return lo + n * (hi - lo)
    if kind == "log":
        if lo <= 0.0:
            raise FxParamMapError(f"log to_normalized needs min > 0, got min={lo}")
        return lo * ((hi / lo) ** n)
    raise FxParamMapError(f"unknown to_normalized kind: {kind!r}")


def _human_instructions(*, fx_name: str, param: str, value: float, reason: str) -> dict:
    """The §8 step 3 fallback: never guess an index. Structured so Claude can relay it
    verbatim to the user — names the plugin, the requested param, and the requested value."""
    if reason == "unmapped_plugin":
        message = (
            f"No parameter map for '{fx_name}'. In REAPER, open its UI and set "
            f"'{param}' to {value} manually."
        )
    else:
        message = (
            f"'{fx_name}' is mapped, but '{param}' is not one of its mapped params. "
            f"In REAPER, open its UI and set '{param}' to {value} manually."
        )
    return {
        "applied": False,
        "human_instructions": {
            "plugin": fx_name,
            "param": param,
            "requested_value": value,
            "reason": reason,
            "message": message,
        },
    }


def add_fx_by_name(track: str, name: str, *, client: BridgeClient | None = None) -> dict:
    """Add an FX to ``track`` by name, probe-first (idempotent — the same find-then-add
    pattern as ``add_instrument``): a second call for an already-loaded FX returns
    ``already_present=True`` rather than stacking a duplicate instance."""
    bridge = client if client is not None else BridgeClient()
    result = bridge.call("add_fx_by_name", track=track, name=name)
    return {
        "track": result.get("track"),
        "fx_index": result.get("fx_index"),
        "already_present": bool(result.get("already_present", False)),
    }


def set_fx_param(
    track: str,
    fx_index: int,
    param: str,
    value: float,
    *,
    client: BridgeClient | None = None,
) -> dict:
    """Set FX ``param`` (by NAME, e.g. ``"freq"``/``"gain"``/``"ratio"``) on ``track``'s
    ``fx_index``'th FX to ``value`` in that param's NATIVE unit (dB/Hz/ms/ratio — whatever
    ``data/fx_param_maps.json`` declares for it).

    Looks up the FX's name (via ``get_fx_params``) to resolve it against the param map,
    translates ``value`` -> normalized 0..1 (:func:`to_normalized`), and calls the bridge's
    ``set_fx_param`` (normalized). An unmapped plugin, or a param not in that plugin's map,
    returns the §8 step 3 human-instructions response instead of guessing an index —
    ``applied`` is ``False`` and no bridge write happens.
    """
    bridge = client if client is not None else BridgeClient()
    info = bridge.call("get_fx_params", track=track, fx_index=fx_index)
    fx_name = info.get("fx_name", "")

    resolved = _plugin_entry(fx_name, load_param_maps())
    if resolved is None:
        return _human_instructions(
            fx_name=fx_name, param=param, value=value, reason="unmapped_plugin"
        )
    _plugin_key, plugin_entry = resolved
    param_entry = plugin_entry.get("params", {}).get(param)
    if param_entry is None:
        return _human_instructions(
            fx_name=fx_name, param=param, value=value, reason="unmapped_param"
        )

    normalized = to_normalized(value, param_entry)
    bridge.call(
        "set_fx_param",
        track=track,
        fx_index=fx_index,
        param_index=param_entry["index"],
        value=normalized,
    )
    return {
        "applied": True,
        "track": track,
        "fx_index": fx_index,
        "fx_name": fx_name,
        "param": param,
        "value": value,
        "unit": param_entry.get("unit"),
        "normalized": normalized,
    }


def get_fx_params(track: str, fx_index: int, *, client: BridgeClient | None = None) -> dict:
    """Read back ``track``'s ``fx_index``'th FX's full param list. The bridge returns
    ``{fx_name, params: [{index, name, value}]}`` with ``value`` normalized 0..1 (locked
    contract); for every param whose index is in that plugin's ``data/fx_param_maps.json``
    entry, this adds ``mapped_name`` (the map's param key), ``unit``, and
    ``value_native`` (:func:`from_normalized`) — additive, never removes the raw fields. A
    plugin absent from the map (or a param index the map doesn't cover) is returned as-is,
    with no native fields — never a guessed unit/range.
    """
    bridge = client if client is not None else BridgeClient()
    info = bridge.call("get_fx_params", track=track, fx_index=fx_index)
    fx_name = info.get("fx_name", "")

    resolved = _plugin_entry(fx_name, load_param_maps())
    index_lookup: dict[int, tuple[str, dict[str, Any]]] = {}
    if resolved is not None:
        _plugin_key, plugin_entry = resolved
        for pname, pentry in plugin_entry.get("params", {}).items():
            index_lookup[int(pentry["index"])] = (pname, pentry)

    params = []
    for raw in info.get("params", []):
        row: dict[str, Any] = {
            "index": raw.get("index"),
            "name": raw.get("name"),
            "value": raw.get("value"),
        }
        mapped = index_lookup.get(int(raw.get("index", -1)))
        if mapped is not None:
            mapped_name, entry = mapped
            row["mapped_name"] = mapped_name
            row["unit"] = entry.get("unit")
            row["value_native"] = from_normalized(float(raw.get("value", 0.0)), entry)
        params.append(row)
    return {"fx_name": fx_name, "params": params}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(name="add_fx_by_name", annotations=_DESTRUCTIVE)
    def add_fx_by_name_tool(track: str, name: str) -> dict:
        """Add an FX to ``track`` by exact plugin name (e.g. ``"ReaEQ"``), probe-first
        idempotent — a second call for an already-loaded FX returns
        ``already_present=True`` instead of stacking a duplicate instance. Returns
        ``{track, fx_index, already_present}``; ``fx_index`` is the stable identity to pass
        to ``set_fx_param``/``get_fx_params``.
        """
        return add_fx_by_name(track, name)

    @mcp.tool(name="set_fx_param", annotations=_DESTRUCTIVE)
    def set_fx_param_tool(track: str, fx_index: int, param: str, value: float) -> dict:
        """Set FX ``param`` (by name, e.g. ``"freq"``/``"gain"``/``"ratio"``) on
        ``track``'s ``fx_index``'th FX to ``value`` in that param's NATIVE unit (dB/Hz/ms/
        ratio, per ``data/fx_param_maps.json``) — never a raw 0..1 knob position.

        v1 ships param maps for REAPER stock ReaPlugs (ReaEQ/ReaComp/ReaDelay/ReaVerb/
        ReaXcomp — ReaEQ/ReaXcomp cover band 1 only, see the map's notes) plus 3 stock
        JSFX (transient control, saturation, stereo width). An unmapped plugin, or a param
        not in that plugin's map, returns ``{applied: false, human_instructions: {...}}``
        — precise manual instructions instead of a guessed param index (§8 step 3); no
        bridge write happens in that case.
        """
        return set_fx_param(track, fx_index, param, value)

    @mcp.tool(name="get_fx_params", annotations=_RO)
    def get_fx_params_tool(track: str, fx_index: int) -> dict:
        """Read back ``track``'s ``fx_index``'th FX's params: ``{fx_name, params:
        [{index, name, value, mapped_name?, unit?, value_native?}]}``. ``value`` is
        REAPER's raw normalized 0..1; the ``mapped_name``/``unit``/``value_native`` trio is
        present only for params covered by ``data/fx_param_maps.json`` for this plugin —
        an unmapped plugin/param is returned with the raw fields only.
        """
        return get_fx_params(track, fx_index)

    if not include_stubs:
        return  # every tool below is a stub - see the registry honesty rule

    @mcp.tool(annotations={"destructiveHint": True})
    def apply_mix_balance(genre: str) -> dict:
        """[NOT IMPLEMENTED]
        fader_db = mix_target(genre, role) − chain_offset(track FX), clamped to fader range."""
        raise NotImplementedError("M4 — see docs/roadmap.md")

    # list_installed_fx pulled forward from Slice 2 — see orpheus_mcp.tools.instruments.
