"""Lead-sheet tools end to end (spec 2026-10-24 §4, §6): the owner's sheet text in, a YuE2
score or a REAPER layout out. Through the in-memory MCP client and the fake bridge."""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge.client import BridgeClient
from orpheus_mcp.notation.yue2_check import parse_yue2_abc
from orpheus_mcp.registry import register_tools
from orpheus_mcp.tools.leadsheet import arrange_lead_sheet_impl

SHEET = {
    "key": "A", "mode": "minor", "tempo_bpm": 92, "style": "indie pop, clean guitar, male vocal",
    "sections": [
        {"name": "Intro", "bars": 2, "chart": "Am | Fmaj7",
         "tab": "Am | A0 -  A0 -  A0 -  A0 - |\nF | E1 -  E1 -  E1 -  E1 - |"},
        {"name": "Verse 1", "bars": 4, "chart": "Am / Fmaj7 | C / E (x2)", "drums": "backbeat",
         "melody": "A4:q C5:q B4:q A4:q G#4:h E4:h A4:q C5:q E5:h D5:w",
         "syllables": "Some-thing in the way you lea-ving say _",
         "lyrics": ["Something in the way"]},
        {"name": "Chorus A", "bars": 2, "chart": "F | E---", "melody": "C5:w B4:w",
         "syllables": "Hold on"},
    ],
}


@pytest.fixture
def project():
    return FakeReaperProject(tempo=120.0)


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    mcp = FastMCP(name="LeadSheetTest")
    register_tools(mcp, profile="default")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


async def test_export_yue2_score_is_valid_and_parses_back(mcp_client):
    async with mcp_client as c:
        out = (await c.call_tool("export_yue2_score", {"lead_sheet": SHEET})).structured_content
    assert out["valid"] is True and out["errors"] == []
    assert out["reductions"] == [] and out["warnings"] == []  # Fmaj7 is already native
    assert out["lyrics"].startswith("[Intro]\n\n[Verse]\nSomething in the way\n\n[Chorus]")
    score = parse_yue2_abc(out["abc"])
    assert [g.labels for g in score.groups] == [("intro",), ("verse",), ("chorus",)]
    assert score.vocal[0].pitch == 69 and float(score.vocal[0].onset_q) == 8.0  # after 2 bars
    assert out["style"] == SHEET["style"]
    assert len(out["sidecar"]) == 11  # 9 verse syllables (the last one held) + 2 chorus
    assert out["sidecar"][8]["note_indices"] == [8, 9]  # "say" + its melisma


async def test_parse_tools_read_the_owners_notation(mcp_client):
    async with mcp_client as c:
        chart = (await c.call_tool("parse_chord_chart",
                                   {"chart": "Fmaj7 | C | E | E (stop)", "bars": 5})).data
        tab = (await c.call_tool("parse_riff_tab", {"tab": "C/E | A3 -  -  -  E0 -  -  - |"})).data
    assert chart["bars"] == 5 and chart["notes"] == ["bar 4: stop", "bar 5: no chord"]
    assert [n["pitch"] for n in tab["notes"]] == [36, 28]


async def test_preview_reports_warnings_without_touching_reaper(mcp_client, project):
    bad = {**SHEET, "sections": [{**SHEET["sections"][1], "syllables": "too few"}]}
    async with mcp_client as c:
        out = (await c.call_tool("preview_lead_sheet", {"lead_sheet": bad})).data
    assert any("notes but 2 syllables" in w for w in out["warnings"])
    assert project.tracks == []


def test_arrange_lead_sheet_lays_out_markers_tracks_and_notes(project, tmp_path):
    from orpheus_mcp.models import LeadSheet

    with FakeReaperBridge(tmp_path, make_handlers(project)):
        out = arrange_lead_sheet_impl(BridgeClient(bridge_dir=tmp_path),
                                      LeadSheet.model_validate(SHEET))
    assert project.tempo == 92.0
    assert out["tracks"] == ["Chords", "Bass", "Lead", "Drums"]
    assert [(m["name"], m["bar"]) for m in project.markers] == [
        ("Intro", 1), ("Verse 1", 3), ("Something in the way", 3), ("Chorus A", 7),
        ("Hold on", 7)]
    lead = project.resolve_track("Lead").takes[0].notes
    assert lead[0].pitch == 69
    assert project.ppq_to_qn(lead[0].start_ppq) == 8.0  # verse starts at bar 3 = QN 8
    bass = project.resolve_track("Bass").takes[0].notes
    assert [n.pitch for n in bass] == [33] * 4 + [29] * 4
    assert set(out["instruments"]) == {"Chords", "Bass", "Lead", "Drums"}
    assert [s["bar"] for s in out["sections"]] == [1, 3, 7]


def test_arrange_respects_a_three_four_meter(project, tmp_path):
    from orpheus_mcp.models import LeadSheet

    sheet = LeadSheet(key="C", meter=(3, 4), tempo_bpm=100, sections=[
        {"name": "Verse", "bars": 2, "chart": "C | G", "melody": "C4:h. D4:h."},
        {"name": "Chorus", "bars": 1, "chart": "F", "melody": "F4:h."}])
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        out = arrange_lead_sheet_impl(BridgeClient(bridge_dir=tmp_path), sheet)
    assert (project.ts_num, project.ts_den) == (3, 4)
    assert [s["bar"] for s in out["sections"]] == [1, 3]
    lead = project.resolve_track("Lead").takes[0].notes
    assert [project.ppq_to_qn(n.start_ppq) for n in lead] == [0.0, 3.0, 6.0]  # bar 3 = QN 6


def test_rearranging_does_not_reload_instruments(project, tmp_path):
    from orpheus_mcp.models import LeadSheet

    sheet = LeadSheet.model_validate(SHEET)
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        bridge = BridgeClient(bridge_dir=tmp_path)
        arrange_lead_sheet_impl(bridge, sheet)
        drums_fx = list(project.resolve_track("Drums").fx)
        again = arrange_lead_sheet_impl(bridge, sheet)
    assert again["instruments"] == {}
    assert project.resolve_track("Drums").fx == drums_fx
