"""M2 style fingerprints: loading, the measured-vs-fingerprint diff, and explain_features.

The README's promise is "not a vibe": every claim in a style explanation must be a real
measured-value-vs-fingerprint-value delta, and unmeasured dimensions must be declared as
caveats instead of silently guessed.
"""

from __future__ import annotations

import pytest

from orpheus_mcp.analysis.fingerprint import (
    explain_against_fingerprint,
    explain_features,
    list_fingerprints,
    load_fingerprint,
)
from orpheus_mcp.models import (
    AudioCharacter,
    CompositionSpec,
    GrooveAnalysis,
    HarmonyAnalysis,
    Mode,
    TrackRole,
    TrackSpec,
)


def _classical_spec() -> CompositionSpec:
    """A spec engineered to sit inside every threshold of data/fingerprints/classical.json."""
    return CompositionSpec(
        tempo_bpm=75.0,
        harmony=HarmonyAnalysis(
            key_root="C",
            mode=Mode.MAJOR,
            key_confidence=0.93,
            roman_numerals=["I", "IV", "V", "I"],
        ),
        groove=GrooveAnalysis(swing_pct=0.02, tightness=0.8),
        audio=AudioCharacter(
            low_energy_db=-15.0,   # low - mid = -6  (fingerprint: -6)
            mid_energy_db=-9.0,
            high_energy_db=-12.0,  # high - mid = -3 (fingerprint: -3)
            lufs_integrated=-16.0,
        ),
        tracks=[
            TrackSpec(name="Violins", role=TrackRole.STRINGS),
            TrackSpec(name="Grand", role=TrackRole.KEYS),
        ],
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_list_fingerprints_finds_classical():
    assert "classical" in list_fingerprints()


def test_load_fingerprint_roundtrip():
    fp = load_fingerprint("classical")
    assert fp["name"] == "classical"
    assert fp["tempo_range_bpm"] == [60, 90]


def test_load_unknown_fingerprint_names_the_known_ones():
    with pytest.raises(ValueError, match="classical"):
        load_fingerprint("polka-core")


# --------------------------------------------------------------------------- #
# The diff — every claim a measured-vs-target delta
# --------------------------------------------------------------------------- #


def test_matching_spec_sounds_like_the_style():
    result = explain_against_fingerprint(_classical_spec(), load_fingerprint("classical"))
    assert result.style == "classical"
    assert result.score > 0.75
    assert "sounds like" in result.verdict
    assert all(d.matches for d in result.deltas)


def test_every_delta_carries_measured_and_target_values():
    result = explain_against_fingerprint(_classical_spec(), load_fingerprint("classical"))
    assert result.deltas  # a verdict with no evidence is a vibe
    for delta in result.deltas:
        assert delta.explanation
        assert "vs" in delta.explanation  # measured X vs fingerprint Y
        assert delta.measured is not None
        assert delta.target is not None


def test_mismatching_spec_gets_specific_disagreements():
    spec = _classical_spec()
    spec.tempo_bpm = 140.0
    spec.groove.swing_pct = 0.6
    spec.audio.lufs_integrated = -6.0
    result = explain_against_fingerprint(spec, load_fingerprint("classical"))

    failed = {d.feature: d for d in result.deltas if not d.matches}
    assert {"tempo", "swing", "loudness"} <= set(failed)
    assert "140" in failed["tempo"].explanation
    assert "-6" in failed["loudness"].explanation
    assert result.score < 0.75


def test_unmeasured_dimensions_become_caveats_not_claims():
    spec = _classical_spec()
    spec.audio = AudioCharacter()  # nothing rendered/measured
    result = explain_against_fingerprint(spec, load_fingerprint("classical"))
    features = {d.feature for d in result.deltas}
    assert "tonal_balance" not in features
    assert "loudness" not in features
    assert any("not measured" in c for c in result.caveats)


def test_low_key_confidence_is_a_caveat():
    spec = _classical_spec()
    spec.harmony.key_confidence = 0.4
    result = explain_against_fingerprint(spec, load_fingerprint("classical"))
    assert any("confidence" in c.lower() for c in result.caveats)


# --------------------------------------------------------------------------- #
# explain_features — plain observations for the MusicReport
# --------------------------------------------------------------------------- #


def test_explain_features_reports_measured_facts():
    observations = explain_features(_classical_spec())
    text = " ".join(observations)
    assert "75" in text            # tempo shows up as a number
    assert "C major" in text       # key claim carries the detected key
    assert "0.93" in text          # ... and its confidence


def test_explain_features_flags_genre_territory():
    spec = _classical_spec()
    spec.tempo_bpm = 92.0
    spec.harmony.mode = Mode.MINOR
    observations = explain_features(spec)
    assert any("hiphop" in o for o in observations)


def test_explain_features_on_empty_spec_stays_honest():
    observations = explain_features(CompositionSpec(tempo_bpm=120.0))
    text = " ".join(observations).lower()
    assert "not detected" in text or "no " in text
