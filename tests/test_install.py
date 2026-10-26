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


# -- auto-start through REAPER's Scripts/__startup.lua (2026-10-25) --------------------- #

from orpheus_mcp.install import (  # noqa: E402
    STARTUP_BEGIN,
    STARTUP_END,
    remove_autostart,
    startup_script,
)


def _resource(tmp_path):
    resource = tmp_path / "REAPER"
    resource.mkdir()
    return resource


def test_install_adds_the_autostart_block(tmp_path):
    resource = _resource(tmp_path)
    install_bridge(resource_dir=resource)
    text = startup_script(resource).read_text(encoding="utf-8")
    assert startup_script(resource) == resource / "Scripts" / "__startup.lua"
    assert text.startswith(STARTUP_BEGIN) and text.rstrip().endswith(STARTUP_END)
    assert "/Scripts/orpheus/orpheus_bridge.lua" in text and "dofile" in text


def test_install_keeps_the_owners_startup_lines(tmp_path):
    resource = _resource(tmp_path)
    (resource / "Scripts").mkdir()
    mine = 'reaper.ShowConsoleMsg("my startup")\r\n-- keep me\r\n'
    startup_script(resource).write_bytes(mine.encode("utf-8"))
    install_bridge(resource_dir=resource)
    raw = startup_script(resource).read_bytes().decode("utf-8")
    assert raw.startswith(mine + "\r\n" + STARTUP_BEGIN)  # theirs byte for byte, a blank line
    assert raw.count(STARTUP_BEGIN) == 1
    assert raw.count("\n") == raw.count("\r\n")  # the block follows the file's CRLF endings


def test_reinstall_keeps_one_block_and_refreshes_it(tmp_path):
    resource = _resource(tmp_path)
    (resource / "Scripts").mkdir()
    stale = f"print('mine')\n\n{STARTUP_BEGIN}\nold body\n{STARTUP_END}\nprint('after')\n"
    startup_script(resource).write_text(stale, encoding="utf-8")
    install_bridge(resource_dir=resource)
    install_bridge(resource_dir=resource)
    text = startup_script(resource).read_text(encoding="utf-8")
    assert text.count(STARTUP_BEGIN) == 1 and "old body" not in text
    assert text.startswith("print('mine')\n") and "print('after')" in text


def test_install_without_autostart_leaves_startup_alone(tmp_path):
    resource = _resource(tmp_path)
    install_bridge(resource_dir=resource, autostart=False)
    assert not startup_script(resource).exists()


def test_remove_autostart_takes_out_only_the_block(tmp_path):
    resource = _resource(tmp_path)
    (resource / "Scripts").mkdir()
    mine = "print('mine')\n"
    startup_script(resource).write_text(mine, encoding="utf-8")
    install_bridge(resource_dir=resource)
    assert remove_autostart(resource) is True
    assert startup_script(resource).read_text(encoding="utf-8") == mine
    assert remove_autostart(resource) is False


def test_remove_autostart_deletes_a_startup_file_that_was_only_ours(tmp_path):
    resource = _resource(tmp_path)
    install_bridge(resource_dir=resource)
    assert remove_autostart(resource) is True
    assert not startup_script(resource).exists()


def _run_block(tmp_path, bridge_source: str):
    """Run the installed __startup.lua under Lua with a stub `reaper`, as REAPER would."""
    from lupa import LuaRuntime

    resource = _resource(tmp_path)
    install_bridge(resource_dir=resource)
    (resource / "Scripts" / "orpheus" / "orpheus_bridge.lua").write_text(
        bridge_source, encoding="utf-8")
    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute(
        "console = {}\n"
        "reaper = { GetResourcePath = function() return RESOURCE end,\n"
        "  file_exists = function(p) local f = io.open(p, 'r'); if f then f:close() end\n"
        "    return f ~= nil end,\n"
        "  ShowConsoleMsg = function(s) console[#console + 1] = s end }\n")
    lua.globals().RESOURCE = str(resource).replace("\\", "/")
    lua.execute(startup_script(resource).read_text(encoding="utf-8"))
    return lua.globals()


def test_the_startup_block_runs_the_installed_bridge(tmp_path):
    g = _run_block(tmp_path, "STARTED = true")
    assert g.STARTED is True


def test_a_bridge_that_fails_to_load_is_reported_not_fatal(tmp_path):
    g = _run_block(tmp_path, "error('boom')")
    assert "did not start" in g.console[1] and "boom" in g.console[1]
