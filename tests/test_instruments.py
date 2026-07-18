"""Pure instrument-selection ladder — no REAPER."""
from __future__ import annotations

from orpheus_mcp.instruments import select_instrument


def test_override_wins():
    got = select_instrument("keys", inventory=["Vital"], override="My Piano")
    assert got == {"kind": "named", "name": "My Piano", "source": "override"}


def test_prefers_installed_allowlisted():
    got = select_instrument("keys", inventory=["ReaSynth", "Surge XT"])
    assert got["source"] == "installed"
    assert got["name"] == "Surge XT"
    assert got["kind"] == "named"


def test_pack_when_no_install_match():
    got = select_instrument("keys", inventory=["ReaSynth"], pack_installed=True)
    assert got == {"kind": "named", "name": "sfizz", "source": "pack"}


def test_stock_pitched_fallback():
    got = select_instrument("bass", inventory=["ReaSynth"])
    assert got == {"kind": "named", "name": "ReaSynth", "source": "stock"}


def test_stock_drum_fallback_is_drumkit():
    got = select_instrument("drums", inventory=["ReaSynth"])
    assert got == {"kind": "drumkit", "name": None, "source": "stock"}


def test_installed_drum_match():
    got = select_instrument("drums", inventory=["MT-PowerDrumKit"])
    assert got["source"] == "installed"
    assert got["kind"] == "named"
    assert got["name"] == "MT-PowerDrumKit"
