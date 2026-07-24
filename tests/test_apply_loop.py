"""Task 11.2 (§12 slice-11 bar): the closed FX-loop convergence test against the fake bridge.

Drives the full §8 apply loop AS CLAUDE WOULD, entirely through the real TOOLS layer (an
in-memory FastMCP ``Client`` — ``tests/test_reference_tools.py``'s pattern) and the real
``BridgeClient`` against ``FakeReaperBridge`` (``tests/test_render_region.py``'s pattern):

    characterize(ref) vs characterize(render) -> compare_character -> rule 1 fires ->
    apply via add_fx_by_name + set_fx_param -> re-render -> re-compare -> repeat.

``fake_reaper.py``'s render handler (this task's extension) makes this a REAL closed loop,
not a scripted one: the rendered WAV's spectrum genuinely reflects whatever ReaEQ band-1
params ``set_fx_param`` wrote, via a hand-rolled RBJ peaking biquad converted from normalized
wire values through the SAME ``data/fx_param_maps.json`` map ``tools/mix.py`` reads (Task
11.1's locked contract). A bug anywhere in that chain — unit translation, map indices, band
math — would make this loop fail to converge, not silently pass.

The REFERENCE audio is the SAME deterministic track content
(``fake_reaper._synthesize_track_audio``) with a FIXED, narrow bell (+13 dB @ 3.5 kHz, Q 3.5)
applied OFFLINE (``fake_reaper._apply_peaking_eq`` — the identical filter the fake bridge's
render path uses) — a real, closable rule-1 gap, not a scripted one. The narrow Q means only
the 3500 Hz partial is fully boosted (the 2500/4500 Hz neighbors inside the same rule band are
barely touched), so the band-AVERAGED 2-5 kHz delta undershoots the true peak gain on each
correction — the loop needs two real iterations to close it, not one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
from fastmcp import Client, FastMCP

from fake_reaper import (
    FakeReaperBridge,
    FakeReaperProject,
    FakeTake,
    FakeTrack,
    _apply_peaking_eq,
    _synthesize_track_audio,
    make_handlers,
)
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.registry import register_tools

_TRACK_SR = 48000
_REF_GAIN_DB = 13.0
_REF_FREQ_HZ = 3500.0
# A narrow (~0.4-octave) bell: mostly boosts the 3500 Hz partial alone, only lightly touching
# the 2500/4500 Hz neighbors inside the same rule band -- chosen (by direct measurement, see
# task-11.2-report.md) so the loop needs TWO corrections to close a real, closable gap,
# rather than one, giving this test a genuinely multi-iteration trajectory to assert over.
_REF_Q = 3.5


@pytest.fixture
def project() -> FakeReaperProject:
    return FakeReaperProject(
        srate=_TRACK_SR,
        tracks=[FakeTrack(guid="{T1}", name="Vox", takes=[FakeTake()])],
    )


@pytest.fixture
def mcp_client(project, tmp_path, monkeypatch):
    """The real render_track_region / add_fx_by_name / set_fx_param / characterize /
    compare_character tools, wired against FakeReaperBridge — driven exactly as Claude would
    (test_render_region.py's + test_fx_tools.py's + test_reference_tools.py's patterns,
    combined for this one closed-loop scenario)."""
    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path / "analysis_cache"))
    mcp = FastMCP(name="OrpheusApplyLoopTest")
    register_tools(mcp, profile="default")
    with FakeReaperBridge(tmp_path, make_handlers(project)):
        yield Client(mcp)


@pytest.fixture
def reference_wav(tmp_path) -> Path:
    """The reference: the SAME deterministic track content the fake bridge renders from
    (``_synthesize_track_audio``), with a FIXED +6 dB bell at 3.5 kHz applied OFFLINE (not
    through the fake bridge's render path) — a real, closable rule-1 gap."""
    base = _synthesize_track_audio(_TRACK_SR)
    boosted = _apply_peaking_eq(base, _TRACK_SR, _REF_FREQ_HZ, _REF_GAIN_DB, _REF_Q)
    path = tmp_path / "reference.wav"
    sf.write(str(path), boosted.astype("float32"), _TRACK_SR, subtype="FLOAT")
    return path


async def _band_delta_and_rule1(client, ref_path, render_path):
    """One compare_character round-trip through the real tool: the 2-5 kHz delta plus rule
    1's fired Intent (or None if it no longer fires)."""
    result = await client.call_tool(
        "compare_character", {"a": str(ref_path), "b": render_path}
    )
    diff = result.data
    entry = diff.diffs.get("band_2000_5000_db")
    assert entry is not None, (
        "band_2000_5000_db excluded by band_limit_hz -- fixture bandwidth regression"
    )
    rule1 = next((i for i in diff.intents if i.rule_id == 1), None)
    return float(entry["delta"]), rule1


async def test_closed_fx_loop_converges_against_the_fake_bridge(mcp_client, reference_wav):
    async with mcp_client as c:
        render = await c.call_tool(
            "render_track_region", {"track": "1", "start_s": 0.0, "end_s": 1.0}
        )
        render_path = render.data["path"]

        delta, rule1 = await _band_delta_and_rule1(c, reference_wav, render_path)
        deltas = [delta]
        # The scripted gap is real (+6 dB @ 3.5 kHz on the reference only) -- rule 1 MUST
        # fire on the very first, uncorrected render.
        assert rule1 is not None, "rule 1 never fired on the initial render -- no gap to close"

        iterations = 0
        cumulative_gain_db = 0.0  # set_fx_param writes an ABSOLUTE native value (a real EQ
        # knob position, not a delta) -- band 1 is ONE filter instance, so re-applying rule
        # 1's freshly-measured residual means ADDING it to whatever's already dialed in, not
        # replacing it (replacing would throw away the previous correction entirely, since
        # every render re-synthesizes from the untouched base track audio through whatever
        # single gain is currently set).
        while rule1 is not None and iterations < CFG.FX_LOOP_MAX_ITERATIONS:
            # Apply rule 1's intent via the mix tools, exactly as Claude would: apply_path
            # names the plugin + center freq; "gain_from": "delta" says the dB to apply lives
            # in the intent's own params under that key (§15.3's fmt contract).
            apply_path = rule1.apply_path
            assert apply_path["plugin"] == "ReaEQ"
            cumulative_gain_db += float(rule1.params[apply_path["gain_from"]])

            added = await c.call_tool("add_fx_by_name", {"track": "1", "name": "ReaEQ"})
            fx_index = added.data["fx_index"]
            await c.call_tool(
                "set_fx_param",
                {
                    "track": "1",
                    "fx_index": fx_index,
                    "param": "freq",
                    "value": apply_path["freq_hz"],
                },
            )
            await c.call_tool(
                "set_fx_param",
                {"track": "1", "fx_index": fx_index, "param": "gain", "value": cumulative_gain_db},
            )
            await c.call_tool(
                "set_fx_param",
                {"track": "1", "fx_index": fx_index, "param": "q", "value": _REF_Q},
            )

            render = await c.call_tool(
                "render_track_region", {"track": "1", "start_s": 0.0, "end_s": 1.0}
            )
            render_path = render.data["path"]

            delta, rule1 = await _band_delta_and_rule1(c, reference_wav, render_path)
            deltas.append(delta)
            iterations += 1

    # §12 slice-11 convergence bar:
    assert iterations <= CFG.FX_LOOP_MAX_ITERATIONS, deltas
    # the 2-5 kHz delta shrinks MONOTONICALLY every iteration ...
    assert all(
        abs(deltas[i + 1]) < abs(deltas[i]) for i in range(len(deltas) - 1)
    ), f"non-monotonic delta trajectory: {deltas}"
    # ... and lands under rule 1's own trigger (3.0 dB) ...
    assert abs(deltas[-1]) < 3.0, deltas
    # ... at which point rule 1 no longer fires at all.
    assert rule1 is None, rule1

    print(f"\nband_2000_5000_db delta trajectory ({iterations} iterations): {deltas}")
