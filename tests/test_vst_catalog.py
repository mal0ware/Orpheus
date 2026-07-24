"""Task 12.2 (§15.4): the curated free-VST catalog (``data/vst_catalog.json``, §8.1's
instruments + effects tables) plus the two data-lookup/cache-admin tools that round out
the reference-analysis engine's tool surface.

Section 1 parses+validates the catalog data file itself (schema, both §8.1 tables present,
the exact account-walled set the plan pins). Section 2 exercises ``suggest_free_vst`` as a
pure function (ranking, limit, the "cavern reverb" -> Valhalla Supermassive bar, the
NO-NETWORK-CALL invariant). Section 3 exercises ``clear_analysis_cache`` against a real
``CacheFS``. Section 4 is the MCP tool-registration contract (readOnlyHint / destructiveHint,
in-memory ``Client``, ``tests/test_m2_tools.py``'s pattern).
"""

from __future__ import annotations

import json
import socket

import pytest
from fastmcp import Client, FastMCP

from orpheus_mcp.analysis.cachefs import CacheFS
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import CacheClearResult, VstEntry, VstSuggestions
from orpheus_mcp.registry import register_tools
from orpheus_mcp.tools.reference import (
    VstCatalogError,
    clear_analysis_cache,
    load_vst_catalog,
    suggest_free_vst,
)

# --------------------------------------------------------------------------- #
# Section 1: data/vst_catalog.json itself
# --------------------------------------------------------------------------- #

# The exact account-walled set the plan pins (§8.1 flagged rows) -- no more, no fewer.
_ACCOUNT_WALLED = {"Vital", "Decent Sampler", "Splice", "Analog Obsession", "Polyverse Wider"}

# A representative spot-check of names from BOTH §8.1 tables (not exhaustive -- the row
# count assertion below covers completeness).
_INSTRUMENT_SPOT_NAMES = {
    "Surge XT", "Vital", "Dexed", "Odin 2", "OB-Xd", "MT Power Drum Kit 2",
    "Neural Amp Modeler (NAM)", "Splice",
}
_EFFECT_SPOT_NAMES = {
    "Airwindows Consolidated", "Valhalla Supermassive", "TDR Nova", "TDR Kotelnikov",
    "Xfer OTT", "Dragonfly Reverbs", "Analog Obsession", "Polyverse Wider",
}


def test_catalog_parses_into_vst_entries():
    entries = load_vst_catalog()
    assert entries
    assert all(isinstance(e, VstEntry) for e in entries)


def test_catalog_both_8_1_tables_present_by_name():
    names = {e.name for e in load_vst_catalog()}
    assert names >= _INSTRUMENT_SPOT_NAMES
    assert names >= _EFFECT_SPOT_NAMES


def test_catalog_has_both_categories_with_plausible_counts():
    entries = load_vst_catalog()
    instruments = [e for e in entries if e.category == "instrument"]
    effects = [e for e in entries if e.category == "effect"]
    # §8.1's instrument table has 10 rows, the effects table 13 (three rows -- Valhalla's
    # trio and Voxengo/Youlean -- are split into their own catalog entries since each is a
    # distinct installable plugin with its own intents; see the module docstring in
    # data/vst_catalog.json's neighbor, analysis/intents.py, for the same "don't collapse
    # distinct things into one row" principle).
    assert len(instruments) == 10
    assert len(effects) == 16
    assert len(entries) == len(instruments) + len(effects)


def test_catalog_account_required_is_exactly_the_pinned_set():
    entries = load_vst_catalog()
    walled = {e.name for e in entries if e.account_required}
    assert walled == _ACCOUNT_WALLED


def test_catalog_account_required_entries_still_carry_url_and_license():
    entries = load_vst_catalog()
    for e in entries:
        if e.name in _ACCOUNT_WALLED:
            assert e.url
            assert e.license


def test_catalog_every_entry_has_urls_and_nonempty_intents():
    for e in load_vst_catalog():
        assert e.url.startswith("http"), e.name
        assert e.intents, e.name


def test_catalog_ob_xd_pins_legacy_2x_gpl3_line():
    entries = {e.name: e for e in load_vst_catalog()}
    ob_xd = entries["OB-Xd"]
    assert ob_xd.license == "GPL3 (legacy line)"
    assert any("legacy 2.x" in n for n in ob_xd.notes)


def test_catalog_nam_is_mit_with_tone3000_note():
    entries = {e.name: e for e in load_vst_catalog()}
    nam = entries["Neural Amp Modeler (NAM)"]
    assert nam.license == "MIT"
    assert any("TONE3000" in n for n in nam.notes)


def test_catalog_splice_notes_labs_absorption():
    entries = {e.name: e for e in load_vst_catalog()}
    splice = entries["Splice"]
    assert splice.account_required is True
    assert any("LABS" in n for n in splice.notes)


def test_catalog_missing_schema_version_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(VstCatalogError, match="schema_version"):
        load_vst_catalog(bad)


def test_catalog_malformed_entry_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"schema_version": 1, "entries": [{"name": "Incomplete"}]}),
        encoding="utf-8",
    )
    with pytest.raises(VstCatalogError):
        load_vst_catalog(bad)


def test_catalog_empty_entries_list_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "entries": []}), encoding="utf-8")
    with pytest.raises(VstCatalogError, match="non-empty"):
        load_vst_catalog(bad)


# --------------------------------------------------------------------------- #
# Section 2: suggest_free_vst -- pure function, ranking, the named bar, no-network invariant
# --------------------------------------------------------------------------- #


def test_suggest_free_vst_cavern_reverb_returns_valhalla_supermassive_top3():
    result = suggest_free_vst("cavern reverb")
    assert isinstance(result, VstSuggestions)
    assert len(result.entries) <= 3
    assert any(e.name == "Valhalla Supermassive" for e in result.entries)
    assert result.notes == []


def test_suggest_free_vst_ranks_by_match_count_descending():
    # "Valhalla Supermassive" matches BOTH tokens ("cavern" is unique to it); "Dragonfly
    # Reverbs" and other reverb entries match only "reverb" -- Supermassive must rank first.
    result = suggest_free_vst("cavern reverb", limit=5)
    assert result.entries[0].name == "Valhalla Supermassive"


def test_suggest_free_vst_respects_limit():
    result = suggest_free_vst("reverb", limit=1)
    assert len(result.entries) == 1


def test_suggest_free_vst_default_limit_is_3():
    result = suggest_free_vst("reverb")
    assert len(result.entries) <= 3


def test_suggest_free_vst_no_match_returns_empty_entries_with_one_note():
    result = suggest_free_vst("definitely-not-a-real-plugin-need-xyzzy")
    assert result.entries == []
    assert len(result.notes) == 1
    assert "definitely-not-a-real-plugin-need-xyzzy" in result.notes[0]
    assert "categories" in result.notes[0]


def test_suggest_free_vst_case_insensitive():
    lower = [e.name for e in suggest_free_vst("cavern reverb").entries]
    upper = [e.name for e in suggest_free_vst("CAVERN REVERB").entries]
    assert lower == upper


def test_suggest_free_vst_entries_carry_license_and_account_flags():
    result = suggest_free_vst("tape saturation")
    assert result.entries
    for e in result.entries:
        assert isinstance(e.account_required, bool)
        assert e.license
        assert e.url


def test_suggest_free_vst_matches_over_category_too():
    # "instrument" as a literal need-token should surface instrument-category entries.
    result = suggest_free_vst("instrument", limit=10)
    assert any(e.category == "instrument" for e in result.entries)


def test_suggest_free_vst_never_calls_the_network(monkeypatch):
    """Recommend-only invariant (§8/§15.4): suggest_free_vst is a pure data lookup over the
    already-loaded catalog list -- it must NEVER open a socket, even transitively. Monkeypatch
    ``socket.socket`` to explode if anything tries."""

    def _boom(*args, **kwargs):
        raise AssertionError("suggest_free_vst must never touch the network")

    monkeypatch.setattr(socket, "socket", _boom)
    catalog = load_vst_catalog()  # loaded BEFORE the patch -- pure list, no I/O from here on
    result = suggest_free_vst("cavern reverb", catalog=catalog)
    assert result.entries


def test_suggest_free_vst_result_contains_urls_never_a_download_action():
    result = suggest_free_vst("cavern reverb")
    for e in result.entries:
        assert e.url
        # The contract is data-only: no field on VstEntry represents an install/download
        # action, only descriptive/recommend fields.
        assert set(VstEntry.model_fields) == {
            "name", "vendor", "category", "license", "account_required", "url",
            "intents", "notes",
        }


# --------------------------------------------------------------------------- #
# Section 3: clear_analysis_cache -- wraps CacheFS.clear
# --------------------------------------------------------------------------- #


def test_clear_analysis_cache_reports_bytes_and_entries_cleared(tmp_path):
    fs = CacheFS(tmp_path / "cache")
    fs.write_json("hash1", "recon", {"a": 1}, meta={"schema_version": 1})
    fs.write_json("hash2", "recon", {"b": 2}, meta={"schema_version": 1})
    assert fs.size_bytes() > 0

    result = clear_analysis_cache(cache=fs)

    assert isinstance(result, CacheClearResult)
    assert result.entries_cleared == 2
    assert result.bytes_freed > 0
    assert fs.size_bytes() == 0


def test_clear_analysis_cache_on_empty_cache_is_a_harmless_zero(tmp_path):
    fs = CacheFS(tmp_path / "cache")
    result = clear_analysis_cache(cache=fs)
    assert result.entries_cleared == 0
    assert result.bytes_freed == 0


def test_clear_analysis_cache_next_call_recomputes(tmp_path):
    """After clearing, a cache read for a previously-cached key is a genuine MISS (the
    directory is gone) -- proves this isn't a soft/logical clear."""
    fs = CacheFS(tmp_path / "cache")
    meta = {"schema_version": 1}
    fs.write_json("hash1", "recon", {"a": 1}, meta=meta)
    assert fs.read_json("hash1", "recon", meta=meta) is not None

    clear_analysis_cache(cache=fs)

    assert fs.read_json("hash1", "recon", meta=meta) is None


def test_clear_analysis_cache_never_touches_model_checkpoints(tmp_path, monkeypatch):
    """The Critical review finding: clear_analysis_cache must never delete downloaded model
    weights under checkpoints/ (model_registry.checkpoint_dir()), which lives under the SAME
    cache root as the analysis artifacts this tool is meant to wipe. Env-override the cache
    root so CacheFS() and checkpoint_dir() resolve to the identical root, exactly as they do
    in production."""
    from orpheus_mcp.analysis import model_registry

    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path))
    fs = CacheFS()
    fs.write_json("hash1", "recon", {"a": 1}, meta={"schema_version": 1})

    ckpt_dir = model_registry.checkpoint_dir("some-model")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "weights.bin"
    payload = b"x" * 10_000_000  # stands in for a real ~10 GB checkpoint payload
    weights.write_bytes(payload)

    result = clear_analysis_cache(cache=fs)

    # Only the analysis hash entry counted/cleared -- checkpoints/ is neither.
    assert result.entries_cleared == 1
    assert result.bytes_freed < len(payload)
    assert fs.read_json("hash1", "recon", meta={"schema_version": 1}) is None
    assert weights.is_file()
    assert weights.read_bytes() == payload


# --------------------------------------------------------------------------- #
# Section 4: MCP tool registration contract
# --------------------------------------------------------------------------- #


@pytest.fixture
def mcp_client(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path / "orpheus_cache"))
    mcp = FastMCP(name="OrpheusVstCatalogTest")
    register_tools(mcp, profile="default")
    return Client(mcp)


async def test_suggest_free_vst_tool_registered_read_only(mcp_client):
    async with mcp_client as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert "suggest_free_vst" in tools
    assert tools["suggest_free_vst"].annotations.readOnlyHint is True


async def test_clear_analysis_cache_tool_registered_destructive(mcp_client):
    async with mcp_client as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert "clear_analysis_cache" in tools
    assert tools["clear_analysis_cache"].annotations.destructiveHint is True


async def test_suggest_free_vst_tool_via_client(mcp_client):
    async with mcp_client as c:
        result = await c.call_tool("suggest_free_vst", {"need": "cavern reverb"})
    names = [e.name for e in result.data.entries]
    assert "Valhalla Supermassive" in names


async def test_clear_analysis_cache_tool_via_client(mcp_client, tmp_path):
    async with mcp_client as c:
        # Populate the cache this client's server points at, then clear through the tool.
        fs = CacheFS(tmp_path / "orpheus_cache")
        fs.write_json("hash1", "recon", {"a": 1}, meta={"schema_version": 1})
        result = await c.call_tool("clear_analysis_cache", {})
    assert result.data.entries_cleared >= 1
