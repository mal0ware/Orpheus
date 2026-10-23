"""The minimal SFZ reader behind the FreePats sound pack (spec 2026-10-23; sound-pack design).

Real CC0 SFZ files live in tests/fixtures/sfz/. Synthetic snippets cover the syntax those three
files don't use but other FreePats banks do (note names, spaces in sample paths, repeated
<global>, release triggers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orpheus_mcp.sfz import SfzRegion, note_number, parse_sfz, rs5k_zones

FIX = Path(__file__).resolve().parent / "fixtures" / "sfz"


def _load(name: str) -> list[SfzRegion]:
    return parse_sfz((FIX / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("value", "midi"),
    [("60", 60), ("C4", 60), ("c4", 60), ("A0", 21), ("C-1", 0), ("Db2", 37), ("F#3", 54),
     ("B#3", 60), ("Cb4", 59)],
)
def test_note_number(value, midi):
    assert note_number(value) == midi


def test_note_number_rejects_garbage():
    with pytest.raises(ValueError):
        note_number("H2")


def test_guitar_parses_every_region_with_key_shorthand():
    regions = _load("SpanishClassicalGuitar-20190618.sfz")
    assert len(regions) == 48
    single = [r for r in regions if r.lokey == r.hikey]
    assert len(single) == 40 and all(r.pitch_keycenter == r.lokey for r in single)
    assert min(r.lokey for r in regions) == 29 and max(r.hikey for r in regions) == 88
    assert all(r.sample.startswith("samples/") and r.sample.endswith(".flac") for r in regions)
    assert all(r.loop_mode == "no_loop" for r in regions)  # inherited from <group>


def test_bass_keeps_hash_in_sample_paths():
    regions = _load("PickedBassYR 20190930.sfz")
    assert len(regions) == 13
    assert "samples/pick/F#.flac" in {r.sample for r in regions}
    assert (regions[0].lokey, regions[0].hikey, regions[0].pitch_keycenter) == (26, 28, 28)


def test_piano_velocity_layers_come_from_groups():
    regions = _load("UprightPianoKW-20220221.sfz")
    assert len(regions) == 66
    quiet = [r for r in regions if r.hivel == 80]
    loud = [r for r in regions if r.lovel == 81]
    assert len(quiet) == 30 and len(loud) == 36
    assert all(r.lovel == 1 for r in quiet) and all(r.hivel == 127 for r in loud)


def test_spaces_in_sample_paths_and_inline_regions():
    regions = parse_sfz(
        "<group> loop_mode=one_shot\n"
        "<region> sample=Button Accordion HN B3.flac pitch_keycenter=47 lokey=46 hikey=48\n"
        "<region> sample=rel.flac key=50 trigger=release\n"
    )
    assert regions[0].sample == "Button Accordion HN B3.flac"
    assert (regions[0].lokey, regions[0].hikey, regions[0].pitch_keycenter) == (46, 48, 47)
    assert regions[1].trigger == "release"


def test_repeated_global_resets_scope_and_note_names_work():
    regions = parse_sfz(
        "<global> hivel=69\n<group>\n<region> sample=a.flac lokey=G1 hikey=Db2 pitch_keycenter=C2\n"
        "<global> lovel=70\n<group>\n<region> sample=b.flac key=60\n"
    )
    assert (regions[0].lovel, regions[0].hivel) == (1, 69)
    assert (regions[0].lokey, regions[0].hikey, regions[0].pitch_keycenter) == (31, 37, 36)
    assert (regions[1].lovel, regions[1].hivel) == (70, 127)


def test_comments_are_ignored_but_not_inside_values():
    regions = parse_sfz(
        "// a comment <region> sample=nope.flac\n<region> sample=x#1.flac key=61 // tail\n"
    )
    assert [r.sample for r in regions] == ["x#1.flac"]


def test_rs5k_zones_pick_one_velocity_layer_and_skip_release_triggers(tmp_path):
    regions = parse_sfz(
        "<group> hivel=80\n<region> sample=q.flac key=60\n"
        "<group> lovel=81\n<region> sample=l.flac key=60\n"
        "<region> sample=r.flac key=61 trigger=release\n"
    )
    zones = rs5k_zones(regions, base_dir=tmp_path, velocity=100)
    assert zones == [
        {"file": str(tmp_path / "l.flac"), "note_lo": 60, "note_hi": 60, "root": 60, "obey": True}
    ]


def test_rs5k_zones_widen_the_outer_zones(tmp_path):
    zones = rs5k_zones(_load("PickedBassYR 20190930.sfz"), base_dir=tmp_path, widen=(24, 60))
    assert zones[0]["note_lo"] == 24 and zones[-1]["note_hi"] == 60
    assert zones[0]["root"] == 28 and zones[-1]["root"] == 40  # roots never move
    assert [z["note_lo"] for z in zones] == sorted(z["note_lo"] for z in zones)


def test_rs5k_zones_resolve_paths_against_the_sfz_folder(tmp_path):
    zones = rs5k_zones(_load("PickedBassYR 20190930.sfz"), base_dir=tmp_path)
    assert zones[1]["file"] == str(tmp_path / "samples" / "pick" / "F.flac")
