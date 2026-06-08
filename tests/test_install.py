"""Tests for `orpheus-mcp install-bridge` — locating REAPER and copying the Lua bridge.

Pure stdlib (no fastmcp, no REAPER process needed)."""

from __future__ import annotations

import pytest

from orpheus_mcp.install import BUNDLED_LUA, find_reaper_resource_dir, install_bridge


def test_finds_macos_resource_dir(tmp_path):
    res = tmp_path / "Library" / "Application Support" / "REAPER"
    res.mkdir(parents=True)
    assert find_reaper_resource_dir(system="Darwin", home=tmp_path) == res


def test_finds_linux_resource_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    res = tmp_path / ".config" / "REAPER"
    res.mkdir(parents=True)
    assert find_reaper_resource_dir(system="Linux", home=tmp_path) == res


def test_finds_windows_resource_dir(tmp_path, monkeypatch):
    appdata = tmp_path / "AppData" / "Roaming"
    (appdata / "REAPER").mkdir(parents=True)
    monkeypatch.setenv("APPDATA", str(appdata))
    assert find_reaper_resource_dir(system="Windows", home=tmp_path) == appdata / "REAPER"


def test_returns_none_when_reaper_absent(tmp_path):
    assert find_reaper_resource_dir(system="Darwin", home=tmp_path) is None


def test_bundled_lua_exists():
    # The script we ship must actually be on disk in the package.
    assert BUNDLED_LUA.is_file()


def test_install_copies_script_into_scripts_orpheus(tmp_path):
    resource = tmp_path / "REAPER"
    resource.mkdir()
    dest = install_bridge(resource_dir=resource)
    assert dest == resource / "Scripts" / "orpheus" / "orpheus_bridge.lua"
    assert dest.is_file()
    assert dest.read_text() == BUNDLED_LUA.read_text()


def test_install_is_idempotent(tmp_path):
    resource = tmp_path / "REAPER"
    resource.mkdir()
    first = install_bridge(resource_dir=resource)
    second = install_bridge(resource_dir=resource)  # must not raise on re-install
    assert first == second


def test_install_raises_without_reaper(monkeypatch):
    monkeypatch.setattr("orpheus_mcp.install.find_reaper_resource_dir", lambda: None)
    with pytest.raises(FileNotFoundError, match="REAPER"):
        install_bridge()
