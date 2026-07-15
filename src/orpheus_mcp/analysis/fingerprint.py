"""Style fingerprints + the measured-vs-target diff that powers explain_style.

A fingerprint (data/fingerprints/*.json, schema documented by classical.json) is the
analysis pipeline run over reference tracks, aggregated into one profile. The comparison
here is the v0.1 half of the differentiator: every claim in a StyleExplanation is a real
measured-value-vs-fingerprint-value delta ("not a vibe"), and anything we could NOT
measure is declared as a caveat instead of silently guessed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from orpheus_mcp.models import (
    CompositionSpec,
    EditPlan,
    FeatureDelta,
    StyleExplanation,
)
from orpheus_mcp.theory.genre_profiles import GENRE_PROFILES

# Tolerances: how far a measurement may sit from the fingerprint value and still count
# as "in character". Deliberately wide — style is a region, not a point.
_SWING_TOLERANCE = 0.15
_BAND_TOLERANCE_DB = 3.0
_LUFS_TOLERANCE = 3.0
_CHORD_COVERAGE_MIN = 0.7
_LOW_KEY_CONFIDENCE = 0.6


def _fingerprints_dir() -> Path:
    """Locate data/fingerprints in both layouts.

    Installed wheel: pyproject force-includes data/ INSIDE the package
    (orpheus_mcp/data). Dev checkout: data/ sits at the repo root next to src/.
    """
    package_dir = Path(__file__).resolve().parents[1] / "data" / "fingerprints"
    if package_dir.is_dir():
        return package_dir
    return Path(__file__).resolve().parents[3] / "data" / "fingerprints"


def list_fingerprints() -> list[str]:
    """Names of every cached style fingerprint (sorted, extension-free)."""
    return sorted(p.stem for p in _fingerprints_dir().glob("*.json"))


def load_fingerprint(name: str) -> dict:
    """Load one fingerprint by name, or raise with the known names spelled out."""
    path = _fingerprints_dir() / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"No fingerprint named {name!r}; known: {list_fingerprints()}")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The diff
# --------------------------------------------------------------------------- #


def explain_against_fingerprint(spec: CompositionSpec, fingerprint: dict) -> StyleExplanation:
    """Compare a measured CompositionSpec to a fingerprint, dimension by dimension.

    Each comparable dimension yields one FeatureDelta whose explanation contains BOTH
    numbers (measured vs target). Dimensions the spec has no measurement for are skipped
    and reported in caveats — an unmeasured dimension is never evidence either way.
    """
    deltas: list[FeatureDelta] = []
    caveats: list[str] = []

    _compare_tempo(spec, fingerprint, deltas)
    _compare_mode(spec, fingerprint, deltas, caveats)
    _compare_chords(spec, fingerprint, deltas, caveats)
    _compare_swing(spec, fingerprint, deltas, caveats)
    _compare_tonal_balance(spec, fingerprint, deltas, caveats)
    _compare_loudness(spec, fingerprint, deltas, caveats)
    _compare_instrumentation(spec, fingerprint, deltas, caveats)

    confidence = spec.harmony.key_confidence
    if confidence is not None and confidence < _LOW_KEY_CONFIDENCE:
        caveats.append(
            f"Key detection confidence is low ({confidence:.2f}) — the mode and chord "
            "comparisons inherit that uncertainty."
        )

    matched = sum(1 for d in deltas if d.matches)
    score = matched / len(deltas) if deltas else 0.0
    style = str(fingerprint.get("name", "unknown"))
    if not deltas:
        verdict = f"cannot judge '{style}' — nothing was measurable"
    elif score >= 0.75:
        verdict = f"sounds like {style} ({matched}/{len(deltas)} dimensions match)"
    elif score >= 0.4:
        verdict = f"partially sounds like {style} ({matched}/{len(deltas)} dimensions match)"
    else:
        verdict = f"does not sound like {style} ({matched}/{len(deltas)} dimensions match)"

    return StyleExplanation(
        style=style, verdict=verdict, score=score, deltas=deltas, caveats=caveats
    )


def _compare_tempo(spec: CompositionSpec, fp: dict, deltas: list[FeatureDelta]) -> None:
    lo, hi = fp.get("tempo_range_bpm", [0, 999])
    bpm = spec.tempo_bpm
    matches = lo <= bpm <= hi
    if matches:
        text = f"tempo {bpm:g} BPM vs {fp['name']} range {lo}-{hi} BPM — inside the range"
    else:
        edge, direction = (lo, "slower") if bpm < lo else (hi, "faster")
        text = (
            f"tempo {bpm:g} BPM vs {fp['name']} range {lo}-{hi} BPM — "
            f"{abs(bpm - edge):g} BPM {direction} than the style plays"
        )
    deltas.append(
        FeatureDelta(
            feature="tempo", measured=bpm, target=f"{lo}-{hi}", matches=matches,
            explanation=text,
        )
    )


def _compare_mode(
    spec: CompositionSpec, fp: dict, deltas: list[FeatureDelta], caveats: list[str]
) -> None:
    typical = fp.get("typical_modes", [])
    mode = spec.harmony.mode
    if mode is None or not typical:
        caveats.append("mode: not measured (no key detection result)")
        return
    matches = mode.value in typical
    deltas.append(
        FeatureDelta(
            feature="mode", measured=mode.value, target=", ".join(typical), matches=matches,
            explanation=(
                f"mode '{mode.value}' vs typical {typical} — "
                + ("in the style's vocabulary" if matches else "not a typical mode here")
            ),
        )
    )


def _roman_core(figure: str) -> str:
    match = re.match(r"(?i)[b#]?[iv]+", figure)
    return match.group(0) if match else figure


def _compare_chords(
    spec: CompositionSpec, fp: dict, deltas: list[FeatureDelta], caveats: list[str]
) -> None:
    vocabulary = fp.get("chord_vocabulary", [])
    numerals = spec.harmony.roman_numerals
    if not numerals or not vocabulary:
        caveats.append("chord vocabulary: not measured (no Roman-numeral analysis)")
        return
    vocab_cores = {_roman_core(v) for v in vocabulary}
    inside = [n for n in numerals if _roman_core(n) in vocab_cores]
    coverage = len(inside) / len(numerals)
    matches = coverage >= _CHORD_COVERAGE_MIN
    outside = sorted({_roman_core(n) for n in numerals} - vocab_cores)
    deltas.append(
        FeatureDelta(
            feature="chord_vocabulary",
            measured=", ".join(numerals),
            target=", ".join(vocabulary),
            matches=matches,
            explanation=(
                f"chords {numerals} vs {fp['name']} vocabulary {vocabulary} — "
                f"{coverage:.0%} of the progression is in-style"
                + (f"; outsiders: {outside}" if outside else "")
            ),
        )
    )


def _compare_swing(
    spec: CompositionSpec, fp: dict, deltas: list[FeatureDelta], caveats: list[str]
) -> None:
    target = fp.get("swing_pct")
    measured = spec.groove.swing_pct
    if measured is None or target is None:
        caveats.append("swing: not measured (no groove analysis)")
        return
    difference = measured - float(target)
    matches = abs(difference) <= _SWING_TOLERANCE
    deltas.append(
        FeatureDelta(
            feature="swing", measured=measured, target=float(target), matches=matches,
            explanation=(
                f"swing {measured:.2f} vs fingerprint {float(target):.2f} — "
                + ("same feel" if matches else f"{difference:+.2f} off the style's pocket")
            ),
        )
    )


def _compare_tonal_balance(
    spec: CompositionSpec, fp: dict, deltas: list[FeatureDelta], caveats: list[str]
) -> None:
    target = fp.get("tonal_balance_db")
    audio = spec.audio
    have_bands = None not in (audio.low_energy_db, audio.mid_energy_db, audio.high_energy_db)
    if not target or not have_bands:
        caveats.append("tonal balance: not measured (no rendered audio analyzed)")
        return
    # Compare RELATIVE band offsets (low-mid, high-mid): absolute band dB moves with
    # overall level, but the low/high tilt against the mids is the style signature.
    assert audio.low_energy_db is not None  # for the type checker; guarded above
    assert audio.mid_energy_db is not None
    assert audio.high_energy_db is not None
    measured_low = audio.low_energy_db - audio.mid_energy_db
    measured_high = audio.high_energy_db - audio.mid_energy_db
    target_low = float(target["low"]) - float(target["mid"])
    target_high = float(target["high"]) - float(target["mid"])
    low_off = measured_low - target_low
    high_off = measured_high - target_high
    matches = abs(low_off) <= _BAND_TOLERANCE_DB and abs(high_off) <= _BAND_TOLERANCE_DB
    deltas.append(
        FeatureDelta(
            feature="tonal_balance",
            measured=f"low{measured_low:+.1f} dB, high{measured_high:+.1f} dB (vs mids)",
            target=f"low{target_low:+.1f} dB, high{target_high:+.1f} dB (vs mids)",
            matches=matches,
            explanation=(
                f"tonal balance low{measured_low:+.1f}/high{measured_high:+.1f} dB vs "
                f"fingerprint low{target_low:+.1f}/high{target_high:+.1f} dB — "
                + (
                    "same tilt"
                    if matches
                    else f"lows {low_off:+.1f} dB and highs {high_off:+.1f} dB off the target"
                )
            ),
        )
    )


def _compare_loudness(
    spec: CompositionSpec, fp: dict, deltas: list[FeatureDelta], caveats: list[str]
) -> None:
    target = fp.get("lufs_integrated")
    measured = spec.audio.lufs_integrated
    if measured is None or target is None:
        caveats.append("loudness: not measured (no rendered audio analyzed)")
        return
    difference = measured - float(target)
    matches = abs(difference) <= _LUFS_TOLERANCE
    deltas.append(
        FeatureDelta(
            feature="loudness", measured=measured, target=float(target), matches=matches,
            explanation=(
                f"loudness {measured:.1f} LUFS vs fingerprint {float(target):.1f} LUFS — "
                + ("comparable level" if matches else f"{difference:+.1f} dB off the style")
            ),
        )
    )


def _compare_instrumentation(
    spec: CompositionSpec, fp: dict, deltas: list[FeatureDelta], caveats: list[str]
) -> None:
    tags = set(fp.get("instrumentation_tags", []))
    roles = {t.role.value for t in spec.tracks if t.role.value != "other"}
    if not tags or not roles:
        caveats.append("instrumentation: not measured (no typed track roles)")
        return
    overlap = roles & tags
    matches = len(overlap) >= max(1, len(roles) // 2)
    deltas.append(
        FeatureDelta(
            feature="instrumentation",
            measured=", ".join(sorted(roles)),
            target=", ".join(sorted(tags)),
            matches=matches,
            explanation=(
                f"instruments {sorted(roles)} vs {fp['name']} palette {sorted(tags)} — "
                f"{len(overlap)}/{len(roles)} roles are in-palette"
            ),
        )
    )


# --------------------------------------------------------------------------- #
# Plain observations (feeds MusicReport.observations)
# --------------------------------------------------------------------------- #


def explain_features(spec: CompositionSpec) -> list[str]:
    """Human-readable observations, each anchored to a measured value.

    Powers 'why does this sound like X' with no transformation (the v0.1 capability):
    tempo/key/groove/mix facts plus which genre-profile territories the measurements
    land in. Unmeasured things are stated as 'not detected', never invented.
    """
    observations: list[str] = []

    bpm = spec.tempo_bpm
    bucket = "slow" if bpm < 85 else "moderate" if bpm < 115 else "uptempo"
    observations.append(f"{bpm:g} BPM — {bucket}")

    harmony = spec.harmony
    if harmony.key_root and harmony.mode:
        confidence = (
            f" (confidence {harmony.key_confidence:.2f})"
            if harmony.key_confidence is not None
            else ""
        )
        observations.append(f"key: {harmony.key_root} {harmony.mode.value}{confidence}")
    else:
        observations.append("key: not detected (no tonal MIDI analyzed)")
    if harmony.roman_numerals:
        observations.append(f"progression: {'-'.join(harmony.roman_numerals)}")

    groove = spec.groove
    if groove.feel:
        observations.append(f"groove: {groove.feel}")
    if groove.swing_pct is not None:
        observations.append(f"swing: {groove.swing_pct:.2f} (0 = straight, 1 = triplet)")

    audio = spec.audio
    if None not in (audio.low_energy_db, audio.high_energy_db):
        assert audio.low_energy_db is not None and audio.high_energy_db is not None
        tilt = audio.low_energy_db - audio.high_energy_db
        if tilt > 3:
            observations.append(f"low-heavy mix (lows {tilt:+.1f} dB over highs)")
        elif tilt < -3:
            observations.append(f"bright mix (highs {-tilt:+.1f} dB over lows)")
        else:
            observations.append("balanced low/high spectrum")
    if audio.lufs_integrated is not None:
        observations.append(f"integrated loudness {audio.lufs_integrated:.1f} LUFS")

    for genre, profile in GENRE_PROFILES.items():
        lo, hi = profile["bpm_range"]
        bpm_in = lo <= bpm <= hi
        mode_in = harmony.mode is not None and harmony.mode.value in profile["typical_modes"]
        if bpm_in and mode_in:
            assert harmony.mode is not None
            observations.append(
                f"{bpm:g} BPM inside {genre}'s {lo}-{hi} range + {harmony.mode.value} mode "
                f"-> {genre} territory"
            )

    return observations


def diff_to_edit_plan(current: CompositionSpec, fingerprint: dict, intent: str) -> EditPlan:
    """Diff current vs a target fingerprint across the v1 dimensions → reason-annotated EditPlan.

    Dimensions (v1): tempo, key/mode, harmonic vocabulary, instrumentation, mix/master.
    Each delta over threshold → a ProposedEdit. Always attaches honest caveats (low
    detection confidence, taste-based genre→param mappings).
    """
    raise NotImplementedError("M3 — see docs/roadmap.md")
