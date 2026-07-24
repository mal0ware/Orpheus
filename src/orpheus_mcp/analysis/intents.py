"""``analysis/intents.py`` (Task 10.1, §8/§15.3): the descriptor→intent rules engine.

Three pieces, in the order the §9 ``compare_character`` tool calls them:

- :func:`load_rules` parses ``data/intent_rules.json`` into :class:`Rule` objects, cross-checking
  every rule's descriptor key(s) against the locked §7 ``SoundCharacter`` key set (Task 6.1) —
  an unknown descriptor is a load-time :class:`RulesError`, never a silent no-op rule (fail fast).
- :func:`diff_characters` computes the §8-normalized numeric diff between two already-measured
  :class:`~orpheus_mcp.models.SoundCharacter` objects (``a`` = reference, ``b`` = user/render,
  per §8's ``SoundCharacter(reference_stem) vs SoundCharacter(render_of_user_track)`` ordering —
  every signed trigger below reads ``delta = a − b``, i.e. reference minus user).
- :func:`fire_rules` evaluates the loaded rules against that diff, producing the fired
  :class:`~orpheus_mcp.models.Intent` objects; :func:`match_inventory` then fills each intent's
  ``installed_matches`` from the caller's own plugin inventory (§8 step 2.1).

Intent generation lives HERE, and only here — the §9 tool contract is explicit that Claude
*reads* intents off ``compare_character``'s result, it never re-derives them.

**§8 step 2 note on "resample both to ANALYSIS_SR":** that instruction describes the ORIGINAL
audio-level normalization (``characterize``/``compute_character``, Task 6.1) — by the time two
:class:`SoundCharacter` objects reach this module they are already plain numbers (Hz-denominated
band edges, dB, ms, ratios — all sample-rate-independent units), so there is no audio here to
resample. This module's own normalization duty is the OTHER §8 mandate: restricting every
spectral band diff to the common measured bandwidth (``band_limit_hz``).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orpheus_mcp.analysis import descriptors as _descriptors
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import Intent, SoundCharacter

__all__ = [
    "RulesError",
    "Rule",
    "Diff",
    "ALLOWED_DESCRIPTOR_KEYS",
    "load_rules",
    "diff_characters",
    "fire_rules",
    "match_inventory",
]


class RulesError(Exception):
    """A malformed ``data/intent_rules.json`` — an unknown ``op``, an unknown descriptor key
    (the Task 6.1 cross-check), or a structurally invalid rule/trigger. Raised at load time
    (fail fast), never swallowed into a silently-inert rule."""


# --------------------------------------------------------------------------- #
# The locked §15.3 descriptor key set — derived from analysis/descriptors.py's own band-edge
# constants (not hand-copied) so this can never drift from what compute_character actually
# emits; the remaining (non-band) keys are the fixed §7 contract enumerated in
# SoundCharacter's docstring / descriptors.py's block builders.
# --------------------------------------------------------------------------- #


def _spectral_band_keys() -> frozenset[str]:
    edges = _descriptors._SKETCH_EDGES_HZ
    names = {f"band_{int(lo)}_{int(hi)}_db" for lo, hi in zip(edges, edges[1:], strict=False)}
    names |= {name for name, _lo, _hi in _descriptors._RULE_BANDS_HZ}
    dyn_lo, dyn_hi = _descriptors._RULE_DYN_BAND_HZ
    names.add(f"band_{int(dyn_lo)}_{int(dyn_hi)}_dyn_db")
    return frozenset(names)


_NON_BAND_DESCRIPTOR_KEYS: frozenset[str] = frozenset(
    {
        "centroid_hz",
        "rolloff_hz",
        "tilt_db_per_oct",
        "hp_ratio",
        "lufs_i",
        "lufs_s_max",
        "crest_db",
        "attack_ms_p50",
        "attack_ms_p90",
        "env_kurtosis",
        "even_db",
        "odd_db",
        "inharmonicity",
        "flavor",
        "width_low",
        "width_mid",
        "width_high",
        "corr",
        "delay_ms",
        "delay_feedback",
        "delay_conf",
        "delay_approx",
        "mod_rate_hz",
        "mod_depth",
        "mod_class",
        "reverb_wet_dry",
        "reverb_rt60_class",
        "reverb_early_late",
        "reverb_approx",
    }
)

ALLOWED_DESCRIPTOR_KEYS: frozenset[str] = _spectral_band_keys() | _NON_BAND_DESCRIPTOR_KEYS

# Descriptor keys whose value is string/bool-typed (not a plain float) — excluded from
# ``diff_characters``'s numeric ``diffs`` dict; §15.3 rules that need them (rule 12's
# ``reverb_rt60_class``) read them straight off the source SoundCharacter blocks instead.
_NON_NUMERIC_DESCRIPTOR_KEYS: frozenset[str] = frozenset(
    {"flavor", "mod_class", "reverb_rt60_class", "delay_approx", "reverb_approx"}
)

_BAND_KEY_RE = re.compile(r"^band_(\d+)_(\d+)(?:_dyn)?_db$")

_TOP_LEVEL_OPS = frozenset({"delta_gt", "delta_lt", "abs_delta_gt", "present_in_a_only", "joint"})
# "differs" is only meaningful inside a "joint" sub-trigger (string-typed descriptors, e.g.
# rule 12's decay-class check) — never a valid top-level op on its own.
_SUB_TRIGGER_OPS = frozenset(
    {"delta_gt", "delta_lt", "abs_delta_gt", "present_in_a_only", "differs"}
)


# --------------------------------------------------------------------------- #
# Rule loading + validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rule:
    """One loaded, validated §15.3 rule row."""

    rule_id: int
    trigger: dict[str, Any]
    intent: str
    apply: dict[str, Any] | None
    explanation: str
    vocal_only: bool
    inventory_keywords: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)  # full source dict (requires_opt_in, ...)


def _rules_path() -> Path:
    """Locate ``data/intent_rules.json`` in both layouts (installed wheel: inside the package;
    dev checkout: repo root next to ``src/``) — same two-layout probe as
    ``model_registry.registry_path``."""
    package_path = Path(__file__).resolve().parents[1] / "data" / "intent_rules.json"
    if package_path.is_file():
        return package_path
    return Path(__file__).resolve().parents[3] / "data" / "intent_rules.json"


def _validate_descriptor(name: object, *, rule_id: int, where: str) -> None:
    if not isinstance(name, str) or name not in ALLOWED_DESCRIPTOR_KEYS:
        raise RulesError(
            f"rule {rule_id}: unknown descriptor {name!r} in {where} — not a locked §7 "
            "SoundCharacter key."
        )


def _validate_trigger(trigger: object, *, rule_id: int) -> None:
    if not isinstance(trigger, dict):
        raise RulesError(f"rule {rule_id}: trigger must be an object.")
    op = trigger.get("op")
    if op == "joint":
        subs = trigger.get("of")
        if not isinstance(subs, list) or not subs:
            raise RulesError(f"rule {rule_id}: joint trigger needs a non-empty 'of' list.")
        for sub in subs:
            if not isinstance(sub, dict):
                raise RulesError(f"rule {rule_id}: joint sub-trigger must be an object.")
            sub_op = sub.get("op")
            if sub_op not in _SUB_TRIGGER_OPS:
                raise RulesError(
                    f"rule {rule_id}: unknown joint sub-trigger op {sub_op!r} "
                    f"(allowed: {sorted(_SUB_TRIGGER_OPS)})."
                )
            _validate_descriptor(sub.get("descriptor"), rule_id=rule_id, where="joint sub-trigger")
        return
    if op not in _TOP_LEVEL_OPS:
        raise RulesError(
            f"rule {rule_id}: unknown trigger op {op!r} (allowed: {sorted(_TOP_LEVEL_OPS)})."
        )
    _validate_descriptor(trigger.get("descriptor"), rule_id=rule_id, where="trigger")


def load_rules(path: str | Path | None = None) -> list[Rule]:
    """Parse + validate ``data/intent_rules.json`` (or an explicit ``path``) into
    :class:`Rule` objects. Raises :class:`RulesError` on any unknown trigger ``op``, any
    descriptor key not in the locked §7 ``SoundCharacter`` contract, or a structurally
    malformed rule — fail fast at load time, never a silently-inert rule (Task 6.1's
    review-requested cross-check)."""
    p = Path(path) if path is not None else _rules_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RulesError(f"cannot read rules file {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RulesError(f"malformed JSON in rules file {p}: {exc}") from exc

    raw_rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RulesError(f"{p}: expected a non-empty top-level 'rules' list.")

    rules: list[Rule] = []
    seen_ids: set[int] = set()
    for entry in raw_rules:
        if not isinstance(entry, dict):
            raise RulesError(f"{p}: each rule must be an object, got {entry!r}.")
        rule_id = entry.get("rule_id")
        if not isinstance(rule_id, int):
            raise RulesError(f"{p}: rule missing an integer 'rule_id': {entry!r}.")
        if rule_id in seen_ids:
            raise RulesError(f"{p}: duplicate rule_id {rule_id}.")
        seen_ids.add(rule_id)

        trigger = entry.get("trigger")
        _validate_trigger(trigger, rule_id=rule_id)
        assert isinstance(trigger, dict)  # narrowed by _validate_trigger (raises otherwise)

        intent = entry.get("intent")
        if not isinstance(intent, str) or not intent:
            raise RulesError(f"rule {rule_id}: missing non-empty 'intent'.")
        explanation = entry.get("explanation")
        if not isinstance(explanation, str) or not explanation:
            raise RulesError(f"rule {rule_id}: missing non-empty 'explanation'.")
        apply_block = entry.get("apply")
        if apply_block is not None and not isinstance(apply_block, dict):
            raise RulesError(f"rule {rule_id}: 'apply' must be an object or null.")
        vocal_only = bool(entry.get("vocal_only", False))
        inventory_keywords = entry.get("inventory_keywords", [])
        if not isinstance(inventory_keywords, list):
            raise RulesError(f"rule {rule_id}: 'inventory_keywords' must be a list.")

        rules.append(
            Rule(
                rule_id=rule_id,
                trigger=trigger,
                intent=intent,
                apply=apply_block,
                explanation=explanation,
                vocal_only=vocal_only,
                inventory_keywords=[str(k) for k in inventory_keywords],
                raw=entry,
            )
        )
    return sorted(rules, key=lambda r: r.rule_id)


_rules_lock = threading.Lock()
_RULES_CACHE: list[Rule] | None = None


def _default_rules() -> list[Rule]:
    """The lazily-loaded, process-cached §15.3 rule set (production callers never pass an
    explicit ``rules=`` override) — loaded on first use, not at import, so a broken data file
    only breaks the call that needs it, not every import of this module."""
    global _RULES_CACHE
    with _rules_lock:
        if _RULES_CACHE is None:
            _RULES_CACHE = load_rules()
        return _RULES_CACHE


# --------------------------------------------------------------------------- #
# diff_characters — §8 mandatory band-limited normalization
# --------------------------------------------------------------------------- #


@dataclass
class Diff:
    """The internal (pre-tool) diff: both source characters (rules that need a non-numeric
    field — rule 12's decay class — read them straight off ``a``/``b``), the flattened numeric
    diff dict, the applied band limit, and the notes accumulated so far (mutated further by
    :func:`fire_rules` as vocal-only/complement caveats fire)."""

    a: SoundCharacter
    b: SoundCharacter
    diffs: dict[str, dict[str, float | None]]
    band_limit_hz: float
    notes: list[str]


def _band_hi_hz(key: str) -> float | None:
    m = _BAND_KEY_RE.match(key)
    return float(m.group(2)) if m else None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):  # bool is an int subclass — never treat as numeric here
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def diff_characters(a: SoundCharacter, b: SoundCharacter) -> Diff:
    """§8-normalized diff of two already-measured :class:`SoundCharacter`\\ s. ``a`` is the
    REFERENCE, ``b`` is the USER/render — every returned ``delta`` is ``a − b`` (signed,
    reference minus user, per §15.3's locked trigger convention).

    ``band_limit_hz`` is ``min(a.source.bandwidth_hz, b.source.bandwidth_hz)`` (falling back to
    whichever side measured a bandwidth, or the Nyquist ceiling of :data:`CFG.ANALYSIS_SR` if
    NEITHER did — an honest "couldn't restrict" default, not a silent skip). Every spectral
    ``band_*_db`` descriptor whose upper edge exceeds ``band_limit_hz`` is excluded from
    ``diffs`` entirely (§8 mandatory normalization: comparing outside the narrower source's
    measured bandwidth reads a full-bandwidth render as systematically 'darker' than a
    codec-limited reference, misfiring the air/tilt rules in the wrong direction). Non-numeric
    descriptor blocks (``flavor``/``mod_class``/``reverb_rt60_class``/``delay_approx``/
    ``reverb_approx``) are excluded from ``diffs`` too — §15.3 rules that need them read the
    source characters directly."""
    bw_a = a.source.bandwidth_hz
    bw_b = b.source.bandwidth_hz
    if bw_a is not None and bw_b is not None:
        band_limit_hz = min(bw_a, bw_b)
    elif bw_a is not None:
        band_limit_hz = bw_a
    elif bw_b is not None:
        band_limit_hz = bw_b
    else:
        band_limit_hz = float(CFG.ANALYSIS_SR) / 2.0

    notes = [
        f"spectral descriptors compared up to {band_limit_hz / 1000.0:.1f} kHz "
        "(narrower source's measured bandwidth, §8 normalization)."
    ]

    blocks_a: dict[str, Any] = {**a.spectral, **a.dynamics, **a.harmonics, **a.stereo, **a.time_fx}
    blocks_b: dict[str, Any] = {**b.spectral, **b.dynamics, **b.harmonics, **b.stereo, **b.time_fx}
    spectral_keys = set(a.spectral) | set(b.spectral)

    diffs: dict[str, dict[str, float | None]] = {}
    for key in sorted(set(blocks_a) | set(blocks_b)):
        if key in _NON_NUMERIC_DESCRIPTOR_KEYS:
            continue
        if key in spectral_keys:
            hi = _band_hi_hz(key)
            if hi is not None and hi > band_limit_hz:
                continue  # excluded: partially/wholly outside the common band
        va = _as_float(blocks_a.get(key))
        vb = _as_float(blocks_b.get(key))
        delta = va - vb if (va is not None and vb is not None) else None
        diffs[key] = {"a": va, "b": vb, "delta": delta}

    return Diff(a=a, b=b, diffs=diffs, band_limit_hz=band_limit_hz, notes=notes)


# --------------------------------------------------------------------------- #
# fire_rules — trigger evaluation
# --------------------------------------------------------------------------- #


def _raw_value(char: SoundCharacter, key: str) -> Any:
    for block in (char.spectral, char.dynamics, char.harmonics, char.stereo, char.time_fx):
        if key in block:
            return block[key]
    return None


def _eval_simple(sub: dict[str, Any], diff: Diff, *, joint: bool) -> tuple[bool, dict[str, Any]]:
    """Evaluate one non-joint trigger (or one joint sub-trigger) against ``diff``. Returns
    ``(fired, format_kwargs)``. ``joint=True`` selects descriptor-qualified format-kwarg names
    (``"{descriptor}_delta"``/``"{descriptor}_a"``/``"{descriptor}_b"``) so multiple sub-triggers
    never collide in a single explanation template; ``joint=False`` (a top-level simple trigger)
    additionally emits the generic ``"delta"`` key the §15.3 schema example uses."""
    op = sub["op"]
    descriptor = sub["descriptor"]

    if op == "present_in_a_only":
        min_conf = float(sub.get("min_conf", 0.5))
        a_val = _as_float(_raw_value(diff.a, descriptor))
        a_conf = _as_float(_raw_value(diff.a, "delay_conf"))
        b_val = _as_float(_raw_value(diff.b, descriptor))
        b_conf = _as_float(_raw_value(diff.b, "delay_conf"))
        present_a = a_val is not None and a_conf is not None and a_conf >= min_conf
        present_b = b_val is not None and b_conf is not None and b_conf >= min_conf
        if present_a and not present_b:
            return True, {"delay_ms": a_val, "delay_conf": a_conf}
        return False, {}

    if op == "differs":
        a_val = _raw_value(diff.a, descriptor)
        b_val = _raw_value(diff.b, descriptor)
        if a_val is not None and b_val is not None and a_val != b_val:
            return True, {f"{descriptor}_a": a_val, f"{descriptor}_b": b_val}
        return False, {}

    entry = diff.diffs.get(descriptor)
    delta = entry["delta"] if entry is not None else None
    if delta is None:
        return False, {}
    value = float(sub["value"])
    if op == "delta_gt":
        fired = delta > value
    elif op == "delta_lt":
        fired = delta < value
    elif op == "abs_delta_gt":
        fired = abs(delta) > value
    else:  # pragma: no cover - load-time validation already rejects unknown ops
        raise RulesError(f"unknown trigger op {op!r}")
    if not fired:
        return False, {}
    fmt = {f"{descriptor}_delta": delta}
    if not joint:
        fmt["delta"] = delta
    return True, fmt


def _eval_trigger(trigger: dict[str, Any], diff: Diff) -> tuple[bool, dict[str, Any]]:
    if trigger["op"] == "joint":
        fmt: dict[str, Any] = {}
        for sub in trigger["of"]:
            fired, sub_fmt = _eval_simple(sub, diff, joint=True)
            if not fired:
                return False, {}
            fmt.update(sub_fmt)
        return True, fmt
    return _eval_simple(trigger, diff, joint=False)


_TAIL_SENSITIVE_RULE_IDS = frozenset({11, 12})


def fire_rules(diff: Diff, rules: list[Rule] | None = None) -> list[Intent]:
    """Evaluate every loaded rule against ``diff`` (mutating ``diff.notes`` with vocal-only-skip
    and complement-discount honesty strings as they occur) and return the fired
    :class:`Intent`\\ s, in rule-id order. ``rules`` defaults to the process-cached
    :func:`load_rules` result of ``data/intent_rules.json``.

    - **vocal_only rules** (rule 13) are skipped entirely — never evaluated, never fired — when
      either side's ``source.stem_type`` isn't ``"vocals"`` (§15.3 note); a plain skip note is
      appended to ``diff.notes`` instead.
    - **Tail-sensitive rules** (11, 12) that DO fire against a §6 R3 ``complement=True`` source
      on either side have their ``apply_path`` forced to ``None`` (recommend-only) with a
      discount note appended — a complement/residual stem inherits the tails of everything
      subtracted from it, so its delay/reverb reading is not trustworthy enough to auto-apply.
    - **Rule 10** (widen)'s ``apply_path`` always carries the LIVE
      ``CFG.WIDEN_MONO_SAFETY_CORR_MIN`` (not whatever static number the JSON happens to embed),
      so a recalibration takes effect without touching the data file.
    """
    rules = rules if rules is not None else _default_rules()
    intents: list[Intent] = []

    for rule in rules:
        if rule.vocal_only and not (
            diff.a.source.stem_type == "vocals" and diff.b.source.stem_type == "vocals"
        ):
            diff.notes.append(
                f"rule {rule.rule_id} ({rule.intent}) skipped: vocal_only, but stem_type is "
                f"{diff.a.source.stem_type!r}/{diff.b.source.stem_type!r} (both sides must be "
                "'vocals')."
            )
            continue

        fired, fmt = _eval_trigger(rule.trigger, diff)
        if not fired:
            continue

        apply_path = dict(rule.apply) if rule.apply is not None else None
        if rule.rule_id == 10 and apply_path is not None:
            apply_path["mono_safety_corr_min"] = CFG.WIDEN_MONO_SAFETY_CORR_MIN
        if (
            rule.rule_id in _TAIL_SENSITIVE_RULE_IDS
            and apply_path is not None
            and (diff.a.source.complement or diff.b.source.complement)
        ):
            apply_path = None
            diff.notes.append(
                f"rule {rule.rule_id} ({rule.intent}) forced recommend-only: a §6 R3 "
                "complement/residual source inherits subtracted stems' tails — the "
                "delay/reverb reading is discounted, not auto-applied."
            )

        try:
            explanation = rule.explanation.format(**fmt)
        except (KeyError, IndexError, ValueError):
            # a template/format mismatch degrades, never crashes
            explanation = rule.explanation

        intents.append(
            Intent(
                rule_id=rule.rule_id,
                intent=rule.intent,
                params={k: v for k, v in fmt.items() if isinstance(v, (int, float, str))},
                apply_path=apply_path,
                explanation=explanation,
                vocal_only_skipped=False,
                installed_matches=[],
            )
        )
    return intents


# --------------------------------------------------------------------------- #
# match_inventory — §8 step 2.1 (inventory-first recommendations)
# --------------------------------------------------------------------------- #


def match_inventory(
    intents: list[Intent], inventory: list[str] | None, rules: list[Rule] | None = None
) -> None:
    """Fill each ``intent.installed_matches`` (in place) with the ``inventory`` plugin names
    whose text substring-matches (case-insensitive) that intent's rule's ``inventory_keywords``
    — so a recommendation names the user's OWN plugin before falling back to stock/catalog (§8
    step 2.1). ``inventory=None`` (or empty) is a no-op — every intent keeps its default empty
    ``installed_matches``."""
    if not inventory:
        return
    rules = rules if rules is not None else _default_rules()
    by_id = {r.rule_id: r for r in rules}
    for intent in intents:
        rule = by_id.get(intent.rule_id)
        if rule is None or not rule.inventory_keywords:
            continue
        keywords = [k.lower() for k in rule.inventory_keywords]
        intent.installed_matches = [
            name for name in inventory if any(kw in name.lower() for kw in keywords)
        ]
