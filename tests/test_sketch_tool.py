"""render_sketch end to end (spec 2026-10-24 sketch): a lead sheet's sections go to YuE2 (the
fake CLI in tests/fixtures) and come back as a new REAPER track on the section's bar (the fake
bridge)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.backends import yue2
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.models import LeadSheet
from orpheus_mcp.notation.yue2_check import check_yue2_abc, parse_yue2_abc
from orpheus_mcp.registry import register_tools
from orpheus_mcp.tools.sketch import SKETCH_PREFIX, render_sketch_impl

FAKE_CLI = Path(__file__).parent / "fixtures" / "fake_yue_synth.py"

SHEET = {
    "key": "A", "mode": "minor", "tempo_bpm": 92, "style": "indie pop, clean guitar, male vocal",
    "sections": [
        {"name": "Intro", "bars": 2, "chart": "Am | Fmaj7"},
        {"name": "Verse 1", "bars": 4, "chart": "Am / Fmaj7 | C / E (x2)",
         "melody": "A4:q C5:q B4:q A4:q G#4:h E4:h A4:q C5:q E5:h D5:w",
         "syllables": "Some-thing in the way you lea-ving say _",
         "lyrics": ["Something in the way"]},
        {"name": "Chorus A", "bars": 2, "chart": "F | E---", "melody": "C5:w B4:w",
         "syllables": "Hold on"},
    ],
}


def _sheet(**changes) -> LeadSheet:
    data = json.loads(json.dumps(SHEET))
    data.update(changes)
    return LeadSheet.model_validate(data)


@pytest.fixture
def install(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    return yue2.Yue2Install(command=[sys.executable, "-I", str(FAKE_CLI)],
                            model=models / "YuE2-3B-Q8_0.gguf", vae=models / "YuE2-Vae-F32.gguf",
                            bin_dir=tmp_path)


@pytest.fixture
def sketches(tmp_path, monkeypatch):
    root = tmp_path / "sketches"
    monkeypatch.setenv(yue2.SKETCH_DIR_ENV, str(root))
    return root


@pytest.fixture
def project():
    return FakeReaperProject(tempo=92.0)


@pytest.fixture
def run(project, install, sketches, tmp_path, monkeypatch):
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", bridge_dir)

    def _run(sheet=None, **kw):
        with FakeReaperBridge(bridge_dir, make_handlers(project)):
            return render_sketch_impl(BridgeClient(), sheet or _sheet(), install=install, **kw)
    return _run


def _bar_s(qn: float, bpm: float = 92.0) -> float:
    return qn * 60.0 / bpm


def _request(out: dict) -> dict:
    return json.loads((Path(out["sketch_dir"]) / "request.json").read_text(encoding="utf-8"))


def test_one_section_lands_on_a_new_track_at_its_bar(run, project):
    out = run(sections=["Verse 1"], seed=1234)
    assert out["track"] == "SKETCH (YuE2) Verse 1 #1234"
    assert out["start_bar"] == 3
    assert out["start_s"] == pytest.approx(_bar_s(8))
    track = project.tracks[-1]
    assert track.name == out["track"] and track.guid == out["track_guid"]
    item = track.items[0]
    assert item.position == pytest.approx(_bar_s(8))
    assert item.source_file == out["wav"] and item.beat_attach_mode == 0  # time-locked
    assert Path(out["wav"]).parent == Path(out["sketch_dir"])
    assert (out["lm_seed"], out["seed"]) == (1234, 1234)
    assert out["truncated"] is False and out["render_s"] > 0 and out["audio_s"] == 0.5


def test_placement_follows_reapers_meter_map_and_a_mismatch_is_warned(run, project):
    project.meter_changes = [(2, 3, 4)]  # REAPER: bar 1 in 4/4, then 3/4
    out = run(sections=["Verse 1"], seed=5)
    assert out["start_s"] == pytest.approx(_bar_s(4 + 3))  # bar 3 = after one 4/4 + one 3/4 bar
    assert project.tracks[-1].items[0].position == pytest.approx(_bar_s(7))
    assert any("3/4" in w and "4/4" in w for w in out["warnings"])


def test_the_request_holds_only_the_chosen_sections_score(run, sketches):
    out = run(sections=["Verse 1"], seed=77)
    req = _request(out)
    assert req["cot"] == "full" and req["output_format"] == "wav24"
    assert (req["lm_seed"], req["seed"]) == (77, 77)
    assert req["style"] == SHEET["style"]
    assert req["lyrics"] == "[Verse]\nSomething in the way"
    assert check_yue2_abc(req["abc"]) == []
    assert [g.labels for g in parse_yue2_abc(req["abc"]).groups] == [("verse",)]
    assert req["duration"] == pytest.approx(4 * 4 * 60 / 92 + 4.0)  # 4 bars + a 4 s tail
    assert out["duration_s"] == req["duration"] and out["cot"] == "full"
    assert re.fullmatch(r"\d{8}-\d{6}-77", Path(out["sketch_dir"]).name)
    assert Path(out["sketch_dir"]).parent == sketches


def test_a_run_of_sections_is_named_by_its_ends(run):
    out = run(sections=["Verse 1", "Chorus A"], seed=9)
    assert out["track"] == "SKETCH (YuE2) Verse 1 to Chorus A #9"
    assert out["start_bar"] == 3
    assert _request(out)["duration"] == pytest.approx(6 * 4 * 60 / 92 + 4.0)


def test_every_section_by_default(run):
    out = run(seed=3)
    assert out["start_bar"] == 1 and out["start_s"] == 0.0
    assert out["track"] == "SKETCH (YuE2) Intro to Chorus A #3"


def test_section_names_match_without_case(run):
    assert run(sections=["chorus a"], seed=3)["start_bar"] == 7


@pytest.mark.parametrize(("sections", "match"), [
    (["Intro", "Chorus A"], "contiguous"), (["Bridge"], "Intro, Verse 1, Chorus A"),
    ([], "at least one")])
def test_bad_section_choices_are_refused_before_rendering(run, sketches, sections, match):
    with pytest.raises(ValueError, match=match):
        run(sections=sections, seed=1)
    assert not sketches.exists()


def test_a_sheet_without_chords_is_sung_from_the_melody(run):
    sheet = _sheet(sections=[{"name": "Verse", "bars": 1, "melody": "A4:h C5:h",
                              "syllables": "hold on"}])
    out = run(sheet, seed=2)
    assert out["cot"] == "melody" and _request(out)["cot"] == "melody"
    assert '"' not in _request(out)["abc"].split("K:", 1)[1]  # no chord symbols


def test_sections_without_a_melody_are_warned(run):
    out = run(sections=["Intro"], seed=2)
    assert any("no melody" in w for w in out["warnings"])


def test_style_can_be_given_and_a_missing_one_is_warned(run):
    assert _request(run(seed=1, style="lo-fi, female vocal"))["style"] == "lo-fi, female vocal"
    out = run(_sheet(style=None), seed=1)
    assert _request(out)["style"] == ""
    assert any("style" in w for w in out["warnings"])


def test_a_project_tempo_that_differs_from_the_sheet_is_warned(run, project):
    project.tempo = 120.0
    out = run(sections=["Verse 1"], seed=4)
    assert any("120" in w and "92" in w for w in out["warnings"])


def test_a_fractional_tempo_is_warned_because_the_score_rounds_it(run, project):
    project.tempo = 92.6
    out = run(_sheet(tempo_bpm=92.6), sections=["Verse 1"], seed=4)
    assert "Q:1/4=93\n" in _request(out)["abc"]
    assert any("92.6" in w and "93" in w for w in out["warnings"])
    assert _request(out)["duration"] == pytest.approx(4 * 4 * 60 / 93 + 4.0)  # the score's tempo


def test_a_sketch_cut_off_past_the_score_gets_a_fade_out(run, project, monkeypatch):
    # YuE2 often keeps writing past the owner's last bar; the 4 s tail cuts it, and the owner
    # chose a 2 s fade-out on the REAPER item for that cut (2026-10-25), reported as normal.
    monkeypatch.setenv("FAKE_YUE_MODE", "truncate")
    out = run(sections=["Verse 1"], seed=4)
    assert out["truncated"] is True
    assert project.tracks[-1].items[0].fade_out == pytest.approx(0.5)  # clamped to the fake's 0.5 s
    assert not any("cut" in w for w in out["warnings"])
    assert "fade" in out["ending"]


def test_a_sketch_that_ends_by_itself_is_left_unfaded(run, project):
    out = run(sections=["Verse 1"], seed=4)
    assert out["truncated"] is False
    assert project.tracks[-1].items[0].fade_out == 0.0
    assert "on its own" in out["ending"]


def test_the_fade_asked_of_reaper_is_two_seconds(install, sketches, monkeypatch):
    monkeypatch.setenv("FAKE_YUE_MODE", "truncate")
    bridge = _StubBridge()
    render_sketch_impl(bridge, _sheet(), install=install, seed=1)
    assert bridge.params["import_audio_files"]["fade_out_s"] == 2.0


def test_a_failed_render_adds_no_track(run, project, monkeypatch):
    monkeypatch.setenv("FAKE_YUE_MODE", "fatal")
    before = len(project.tracks)
    with pytest.raises(yue2.Yue2RenderError):
        run(sections=["Verse 1"], seed=4)
    assert len(project.tracks) == before


def test_unsafe_file_name_characters_are_replaced(run):
    sheet = _sheet(sections=[{"name": 'Verse: 1/2 "b"?', "bars": 1, "chart": "Am",
                              "melody": "A4:w", "syllables": "la"}])
    out = run(sheet, seed=8)
    assert not set(Path(out["wav"]).name) & set('<>:"/\\|?*')
    assert out["track"].startswith(SKETCH_PREFIX + "Verse") and out["track"].endswith(" #8")


def test_a_random_seed_is_chosen_and_reported(run):
    out = run(sections=["Verse 1"])
    assert 0 <= out["seed"] < 2**31 and out["track"].endswith(f" #{out['seed']}")


class _StubBridge:
    def __init__(self, *, down=False, import_fails=False):
        self.down, self.import_fails, self.calls = down, import_fails, []

    def call(self, fn, **params):
        self.calls.append(fn)
        self.params = getattr(self, "params", {})
        self.params[fn] = params
        if self.down:
            raise RuntimeError("REAPER bridge is not running")
        if fn == "get_bar_time":
            return {"bar": params["bar"], "time_s": 0.0, "qn": 0.0, "bpm": 92.0, "meter": [4, 4]}
        if fn == "import_audio_files":
            if self.import_fails:
                return {"imported": [], "failed": [{"file": params["files"][0],
                                                    "error": "could not open audio file"}]}
            return {"imported": [{"track_guid": "{G}"}], "failed": []}
        raise AssertionError(fn)


def test_reaper_is_asked_where_the_bar_is_before_anything_renders(install, sketches):
    with pytest.raises(RuntimeError, match="not running"):
        render_sketch_impl(_StubBridge(down=True), _sheet(), install=install, seed=1)
    assert not sketches.exists()


def test_a_failed_import_still_returns_the_rendered_file(install, sketches):
    out = render_sketch_impl(_StubBridge(import_fails=True), _sheet(), install=install, seed=1)
    assert out["track"] is None and out["track_guid"] is None
    assert Path(out["wav"]).is_file()
    assert "could not open audio file" in out["import_error"]


async def _tool_names(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(yue2.NC_ENV, raising=False)
    else:
        monkeypatch.setenv(yue2.NC_ENV, value)
    mcp = FastMCP(name="SketchTest")
    register_tools(mcp, profile="default")
    async with Client(mcp) as c:
        return {t.name: t for t in await c.list_tools()}


async def test_the_tool_exists_only_with_the_yue2_opt_in(monkeypatch):
    assert "render_sketch" not in await _tool_names(monkeypatch, None)
    assert "render_sketch" not in await _tool_names(monkeypatch, "songbloom")
    tools = await _tool_names(monkeypatch, "yue2")
    assert "render_sketch" in tools
    assert tools["render_sketch"].annotations.destructiveHint is True
