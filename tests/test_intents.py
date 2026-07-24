"""``analysis/intents.py`` (Task 10.1, §8/§15.3): the descriptor→intent rules engine, plus
``tools/reference.py::compare_character``'s wiring.

Section 1 exercises the 14 rules on constructed descriptor pairs (no audio) — exact trigger
boundaries, joint triggers, vocal-only skip, complement discount, apply_path=None. Section 2
uses REAL rendered ``SoundCharacter`` objects (constructed signals, ``compute_character``) for
the §8 band-limit normalization bar (the mp3-vs-wav must-NOT-fire test) and the Task 6.1
cross-check (every rule descriptor key exists on a real ``SoundCharacter``). Section 3 is the
``compare_character`` tool contract (cache wiring, inventory matching).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.descriptors import AudioSource, compute_character
from orpheus_mcp.analysis.intents import (
    ALLOWED_DESCRIPTOR_KEYS,
    RulesError,
    diff_characters,
    fire_rules,
    load_rules,
    match_inventory,
)
from orpheus_mcp.models import SoundCharacter, SourceMeta
from orpheus_mcp.tools.reference import compare_character

SR = 44100

# --------------------------------------------------------------------------- #
# Section 1 fixtures — constructed SoundCharacter pairs (no audio)
# --------------------------------------------------------------------------- #


def _char(
    *,
    stem_type: str = "mix-region",
    bandwidth_hz: float | None = 20000.0,
    complement: bool = False,
    spectral: dict | None = None,
    dynamics: dict | None = None,
    harmonics: dict | None = None,
    stereo: dict | None = None,
    time_fx: dict | None = None,
) -> SoundCharacter:
    return SoundCharacter(
        source=SourceMeta(
            stem_type=stem_type,
            produced_by="test",
            region=None,
            sample_rate=SR,
            bandwidth_hz=bandwidth_hz,
            channels=1,
            complement=complement,
        ),
        spectral=spectral or {},
        dynamics=dynamics or {},
        harmonics=harmonics or {},
        stereo=stereo or {},
        time_fx=time_fx or {},
        pitch=None,
        notes=[],
    )


def _pair(descriptor: str, a_val: float, b_val: float, block: str, **char_kwargs):
    a = _char(**{block: {descriptor: a_val}}, **char_kwargs)
    b = _char(**{block: {descriptor: b_val}}, **char_kwargs)
    return a, b


def _fired_ids(a: SoundCharacter, b: SoundCharacter) -> set[int]:
    return {i.rule_id for i in fire_rules(diff_characters(a, b))}


# --------------------------------------------------------------------------- #
# load_rules: the shipped data file loads clean, has exactly 14 rules
# --------------------------------------------------------------------------- #


def test_shipped_rules_file_loads_14_rules():
    rules = load_rules()
    assert len(rules) == 14
    assert [r.rule_id for r in rules] == list(range(1, 15))


def test_unknown_descriptor_raises_rules_error(tmp_path):
    bad = {
        "schema_version": 1,
        "rules": [
            {
                "rule_id": 1,
                "trigger": {"descriptor": "band_9999_99999_db", "op": "delta_gt", "value": 1.0},
                "intent": "bogus",
                "apply": None,
                "explanation": "n/a",
                "vocal_only": False,
            }
        ],
    }
    path = tmp_path / "bad_rules.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RulesError, match="unknown descriptor"):
        load_rules(path)


def test_unknown_top_level_op_raises_rules_error(tmp_path):
    bad = {
        "schema_version": 1,
        "rules": [
            {
                "rule_id": 1,
                "trigger": {"descriptor": "crest_db", "op": "delta_between", "value": 1.0},
                "intent": "bogus",
                "apply": None,
                "explanation": "n/a",
                "vocal_only": False,
            }
        ],
    }
    path = tmp_path / "bad_op.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RulesError, match="unknown trigger op"):
        load_rules(path)


# --------------------------------------------------------------------------- #
# The 14 exact-boundary rule tests (§10/§12 slice-10 bar)
# --------------------------------------------------------------------------- #


def test_rule1_presence_boost_boundary():
    fire_a, fire_b = _pair("band_2000_5000_db", 3.1, 0.0, "spectral")
    assert 1 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("band_2000_5000_db", 2.9, 0.0, "spectral")
    assert 1 not in _fired_ids(no_a, no_b)


def test_rule2_mud_cut_boundary():
    fire_a, fire_b = _pair("band_200_400_db", 0.0, 3.1, "spectral")  # delta = -3.1
    assert 2 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("band_200_400_db", 0.0, 2.9, "spectral")  # delta = -2.9
    assert 2 not in _fired_ids(no_a, no_b)


def test_rule3_air_shelf_boundary():
    fire_a, fire_b = _pair("band_10000_16000_db", 3.1, 0.0, "spectral")
    assert 3 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("band_10000_16000_db", 2.9, 0.0, "spectral")
    assert 3 not in _fired_ids(no_a, no_b)


def test_rule4_warmth_shelf_boundary():
    fire_a, fire_b = _pair("band_100_250_db", 3.1, 0.0, "spectral")
    assert 4 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("band_100_250_db", 2.9, 0.0, "spectral")
    assert 4 not in _fired_ids(no_a, no_b)


def test_rule5_tilt_boundary():
    fire_a, fire_b = _pair("tilt_db_per_oct", 1.6, 0.0, "spectral")
    assert 5 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("tilt_db_per_oct", 1.4, 0.0, "spectral")
    assert 5 not in _fired_ids(no_a, no_b)


def test_rule6_transient_snap_boundary_both_directions():
    fire_a, fire_b = _pair("attack_ms_p50", 5.1, 0.0, "dynamics")
    assert 6 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("attack_ms_p50", 4.9, 0.0, "dynamics")
    assert 6 not in _fired_ids(no_a, no_b)
    # abs_delta_gt: the reverse direction (user faster than reference) must also fire.
    fire_neg_a, fire_neg_b = _pair("attack_ms_p50", 0.0, 5.1, "dynamics")
    assert 6 in _fired_ids(fire_neg_a, fire_neg_b)


def test_rule7_compression_boundary():
    fire_a, fire_b = _pair("crest_db", 0.0, 4.1, "dynamics")  # delta = -4.1
    assert 7 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("crest_db", 0.0, 3.9, "dynamics")  # delta = -3.9
    assert 7 not in _fired_ids(no_a, no_b)


def test_rule8_saturation_warm_even_boundary():
    fire_a, fire_b = _pair("even_db", 3.1, 0.0, "harmonics")
    assert 8 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("even_db", 2.9, 0.0, "harmonics")
    assert 8 not in _fired_ids(no_a, no_b)


def test_rule9_saturation_drive_odd_boundary():
    fire_a, fire_b = _pair("odd_db", 3.1, 0.0, "harmonics")
    assert 9 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair("odd_db", 2.9, 0.0, "harmonics")
    assert 9 not in _fired_ids(no_a, no_b)


def test_rule10_widen_boundary_and_mono_safety_from_cfg():
    fire_a, fire_b = _pair("width_mid", 0.11, 0.0, "stereo")
    intents = fire_rules(diff_characters(fire_a, fire_b))
    fired = {i.rule_id: i for i in intents}
    assert 10 in fired
    assert fired[10].apply_path is not None
    assert fired[10].apply_path["mono_safety_corr_min"] == CFG.WIDEN_MONO_SAFETY_CORR_MIN

    no_a, no_b = _pair("width_mid", 0.09, 0.0, "stereo")
    assert 10 not in _fired_ids(no_a, no_b)


def test_rule11_delay_present_in_reference_only_boundary():
    fire_a = _char(time_fx={"delay_ms": 28.0, "delay_conf": 0.51})
    fire_b = _char(time_fx={"delay_ms": None, "delay_conf": 0.1})
    assert 11 in _fired_ids(fire_a, fire_b)

    no_a = _char(time_fx={"delay_ms": 28.0, "delay_conf": 0.49})
    no_b = _char(time_fx={"delay_ms": None, "delay_conf": 0.1})
    assert 11 not in _fired_ids(no_a, no_b)

    # present in BOTH (echo already in the user's track too) must not fire.
    both_a = _char(time_fx={"delay_ms": 28.0, "delay_conf": 0.9})
    both_b = _char(time_fx={"delay_ms": 30.0, "delay_conf": 0.8})
    assert 11 not in _fired_ids(both_a, both_b)


def test_rule12_reverb_joint_trigger():
    fire_a = _char(time_fx={"reverb_wet_dry": 0.30, "reverb_rt60_class": "long"})
    fire_b = _char(time_fx={"reverb_wet_dry": 0.10, "reverb_rt60_class": "short"})
    assert 12 in _fired_ids(fire_a, fire_b)

    # wet/dry delta insufficient, even though class differs -> ALL sub-triggers must fire.
    weak_a = _char(time_fx={"reverb_wet_dry": 0.15, "reverb_rt60_class": "long"})
    weak_b = _char(time_fx={"reverb_wet_dry": 0.10, "reverb_rt60_class": "short"})
    assert 12 not in _fired_ids(weak_a, weak_b)

    # wet/dry delta sufficient, but the decay class is the SAME -> must not fire.
    same_class_a = _char(time_fx={"reverb_wet_dry": 0.40, "reverb_rt60_class": "long"})
    same_class_b = _char(time_fx={"reverb_wet_dry": 0.10, "reverb_rt60_class": "long"})
    assert 12 not in _fired_ids(same_class_a, same_class_b)


def test_rule13_de_ess_boundary_on_vocals():
    fire_a, fire_b = _pair(
        "band_5000_9000_dyn_db", 2.1, 0.0, "spectral", stem_type="vocals"
    )
    assert 13 in _fired_ids(fire_a, fire_b)
    no_a, no_b = _pair(
        "band_5000_9000_dyn_db", 1.9, 0.0, "spectral", stem_type="vocals"
    )
    assert 13 not in _fired_ids(no_a, no_b)


def test_rule13_skipped_and_noted_on_mix_region():
    # would fire on the numbers alone, but stem_type is "mix-region" not "vocals".
    a, b = _pair("band_5000_9000_dyn_db", 2.1, 0.0, "spectral", stem_type="mix-region")
    diff = diff_characters(a, b)
    intents = fire_rules(diff)
    assert 13 not in {i.rule_id for i in intents}
    assert any("rule 13" in n and "vocal_only" in n for n in diff.notes), diff.notes


def test_rule14_limiting_joint_trigger_and_apply_path_none():
    fire_a = _char(dynamics={"lufs_i": -8.0, "crest_db": 6.0})
    fire_b = _char(dynamics={"lufs_i": -12.0, "crest_db": 10.0})
    intents = fire_rules(diff_characters(fire_a, fire_b))
    fired = {i.rule_id: i for i in intents}
    assert 14 in fired
    assert fired[14].apply_path is None

    # LUFS delta just under the threshold -> must not fire even with crest satisfied.
    weak_a = _char(dynamics={"lufs_i": -10.5, "crest_db": 6.0})
    weak_b = _char(dynamics={"lufs_i": -12.0, "crest_db": 10.0})
    assert 14 not in _fired_ids(weak_a, weak_b)


# --------------------------------------------------------------------------- #
# Complement (§6 R3) discount — tail-sensitive rules 11/12 forced recommend-only
# --------------------------------------------------------------------------- #


def test_complement_forces_rule11_and_rule12_recommend_only():
    a = _char(
        complement=True,
        time_fx={
            "delay_ms": 28.0,
            "delay_conf": 0.9,
            "reverb_wet_dry": 0.4,
            "reverb_rt60_class": "long",
        },
    )
    b = _char(
        time_fx={
            "delay_ms": None,
            "delay_conf": 0.1,
            "reverb_wet_dry": 0.1,
            "reverb_rt60_class": "short",
        }
    )
    diff = diff_characters(a, b)
    intents = fire_rules(diff)
    fired = {i.rule_id: i for i in intents}
    assert fired[11].apply_path is None
    assert fired[12].apply_path is None
    assert any("complement" in n and "rule 11" in n for n in diff.notes)
    assert any("complement" in n and "rule 12" in n for n in diff.notes)


# --------------------------------------------------------------------------- #
# match_inventory
# --------------------------------------------------------------------------- #


def test_inventory_match_names_users_own_plugin():
    fire_a, fire_b = _pair("band_2000_5000_db", 3.1, 0.0, "spectral")  # fires rule 1 (EQ)
    diff = diff_characters(fire_a, fire_b)
    intents = fire_rules(diff)
    match_inventory(intents, ["TDR Nova (Tokyo Dawn Labs)"])
    rule1 = next(i for i in intents if i.rule_id == 1)
    assert "TDR Nova (Tokyo Dawn Labs)" in rule1.installed_matches


def test_inventory_none_leaves_matches_empty():
    fire_a, fire_b = _pair("band_2000_5000_db", 3.1, 0.0, "spectral")
    intents = fire_rules(diff_characters(fire_a, fire_b))
    match_inventory(intents, None)
    rule1 = next(i for i in intents if i.rule_id == 1)
    assert rule1.installed_matches == []


# --------------------------------------------------------------------------- #
# Section 2 — real audio: §8 band-limit normalization + Task 6.1 cross-check
# --------------------------------------------------------------------------- #


def _mono_source(sig: np.ndarray, *, stem_type: str = "mix-region", sr: int = SR) -> AudioSource:
    sig = np.ascontiguousarray(sig.reshape(-1, 1), dtype=np.float32)
    pcm = Pcm(
        samples=sig, sample_rate=sr, path="", pcm_hash="test",
        duration_s=sig.shape[0] / sr, decoder="test",
    )
    meta = SourceMeta(
        stem_type=stem_type, produced_by="test", region=None, sample_rate=sr, channels=1
    )
    return AudioSource(pcm=pcm, meta=meta)


def _stereo_source(sig: np.ndarray, *, stem_type: str = "mix-region", sr: int = SR) -> AudioSource:
    sig = np.ascontiguousarray(sig, dtype=np.float32)
    pcm = Pcm(
        samples=sig, sample_rate=sr, path="", pcm_hash="test",
        duration_s=sig.shape[0] / sr, decoder="test",
    )
    meta = SourceMeta(
        stem_type=stem_type, produced_by="test", region=None, sample_rate=sr, channels=2
    )
    return AudioSource(pcm=pcm, meta=meta)


def test_mp3_vs_wav_must_not_fire_rule3():
    """§12 slice-10 bar: a full-band render vs a 16 kHz-limited ("mp3-like") reference,
    identical below 16 kHz -> rule 3 (air) must NOT fire, band_limit_hz reads back ~16 kHz,
    and a band-limit note is present."""
    rng = np.random.default_rng(7)
    n = int(SR * 1.0)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)

    full_band = np.fft.irfft(spectrum, n).astype(np.float32)  # the "render" (user) — b
    ref_spectrum = spectrum.copy()
    ref_spectrum[freqs >= 16000.0] = 0.0
    reference = np.fft.irfft(ref_spectrum, n).astype(np.float32)  # the "reference" — a

    ref_char = compute_character(_mono_source(reference))
    render_char = compute_character(_mono_source(full_band))

    assert 15000.0 <= ref_char.source.bandwidth_hz <= 17000.0

    diff = diff_characters(ref_char, render_char)
    assert 15000.0 <= diff.band_limit_hz <= 17000.0
    assert any("kHz" in n for n in diff.notes)

    intents = fire_rules(diff)
    assert 3 not in {i.rule_id for i in intents}


def test_every_rule_descriptor_key_exists_on_a_real_soundcharacter():
    """The Task 6.1 review-requested cross-check: every descriptor referenced by any rule
    (including joint sub-triggers) must be a real key on a real, fully-populated
    SoundCharacter — not just a member of the locked constant set this module derives its
    own validation from."""
    n = int(SR * 1.5)
    left = 0.4 * np.sin(2 * np.pi * 220.0 * np.arange(n) / SR)
    right = 0.4 * np.sin(2 * np.pi * 330.0 * np.arange(n) / SR)  # decorrelated -> true stereo
    stereo_sig = np.stack([left, right], axis=1)
    ch = compute_character(_stereo_source(stereo_sig, stem_type="vocals"))

    real_keys = (
        set(ch.spectral) | set(ch.dynamics) | set(ch.harmonics) | set(ch.stereo) | set(ch.time_fx)
    )

    referenced: set[str] = set()

    def _collect(trigger: dict) -> None:
        if trigger.get("op") == "joint":
            for sub in trigger["of"]:
                referenced.add(sub["descriptor"])
        else:
            referenced.add(trigger["descriptor"])

    for rule in load_rules():
        _collect(rule.trigger)

    missing = referenced - real_keys
    assert not missing, f"rule descriptor(s) not present on a real SoundCharacter: {missing}"
    # every referenced key must also be in the module's own locked allow-list (no drift).
    assert referenced <= ALLOWED_DESCRIPTOR_KEYS


# --------------------------------------------------------------------------- #
# Section 3 — compare_character tool contract
# --------------------------------------------------------------------------- #


@pytest.fixture
def fs(tmp_path) -> CacheFS:
    return CacheFS(tmp_path / "cache")


def _write_wav(path: Path, sig: np.ndarray, sr: int = SR) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sig.ndim < 2:
        sig = sig.reshape(-1, 1)
    sf.write(str(path), sig.astype(np.float32), sr, subtype="FLOAT")
    return str(path)


def _sine_at_dbfs(dbfs: float, freq: float, dur: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    amp = (10.0 ** (dbfs / 20.0)) * np.sqrt(2.0)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_compare_character_tool_returns_character_diff(tmp_path, fs):
    ref_path = _write_wav(tmp_path / "ref.wav", _sine_at_dbfs(-14.0, 440.0))
    user_path = _write_wav(tmp_path / "user.wav", _sine_at_dbfs(-20.0, 440.0))

    result = compare_character(ref_path, user_path, cache=fs)
    assert result.band_limit_hz > 0
    assert result.diffs
    assert isinstance(result.notes, list) and result.notes


def test_compare_character_tool_inventory_wired_through(tmp_path, fs):
    # A broadband noise floor (shared, so both sides read a wide measured bandwidth — a pure
    # tone would cliff-detect as narrowband and get its band excluded by §8 normalization)
    # plus a 3.5 kHz bump on the reference only — a clean, real "presence boost" delta.
    rng = np.random.default_rng(3)
    n = int(SR * 1.0)
    base = rng.standard_normal(n).astype(np.float32) * 0.05
    t = np.arange(n) / SR
    bump = (0.3 * np.sin(2 * np.pi * 3500.0 * t)).astype(np.float32)
    ref_path = _write_wav(tmp_path / "ref2.wav", base + bump)
    user_path = _write_wav(tmp_path / "user2.wav", base.copy())

    result = compare_character(
        ref_path, user_path, inventory=["TDR Nova (Tokyo Dawn Labs)"], cache=fs
    )
    eq_intents = [i for i in result.intents if i.rule_id in (1, 2, 3, 4, 5)]
    assert eq_intents, "expected at least one EQ-family rule to fire on this presence delta"
    assert any("TDR Nova (Tokyo Dawn Labs)" in i.installed_matches for i in eq_intents)
