"""Starting the bridge without the owner (2026-10-25): ``start_bridge`` asks an open REAPER to
run the installed bridge (``REAPER.exe -nonewinst <script>``, verified live on REAPER 7.83),
and a bridge call or get_connection_status tries it once when REAPER is open but the bridge is
off. Nothing here touches the real REAPER: the program, the process check and the launch are
stand-ins."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from fake_reaper import FakeReaperBridge, FakeReaperProject, make_handlers
from orpheus_mcp.bridge import launch
from orpheus_mcp.bridge.client import BridgeClient, BridgeTimeout


@pytest.fixture
def script(tmp_path):
    p = tmp_path / "REAPER" / "Scripts" / "orpheus" / "orpheus_bridge.lua"
    p.parent.mkdir(parents=True)
    p.write_text("-- bridge", encoding="utf-8")
    return p


@pytest.fixture
def exe(tmp_path):
    p = tmp_path / "REAPER.exe"
    p.write_bytes(b"x")
    return p


class Launcher:
    def __init__(self, on_launch=None):
        self.commands, self.on_launch = [], on_launch

    def __call__(self, cmd):
        self.commands.append(cmd)
        if self.on_launch:
            self.on_launch()


def _start(tmp_path, *, alive=lambda: False, running=lambda: True, exe=None, script=None,
           launcher=None, wait_s=0.2):
    return launch.start_bridge(tmp_path / "bridge", wait_s=wait_s, find_exe=lambda: exe,
                               script=script,
                               running=running, launch=launcher or Launcher(),
                               is_alive=alive, poll_s=0.01)


# -- settings and discovery ----------------------------------------------------------- #

@pytest.mark.parametrize(("value", "on"), [(None, True), ("1", True), ("0", False),
                                           ("false", False), ("off", False), ("no", False)])
def test_autostart_is_on_unless_switched_off(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(launch.AUTOSTART_ENV, raising=False)
    else:
        monkeypatch.setenv(launch.AUTOSTART_ENV, value)
    assert launch.autostart_enabled() is on


@pytest.mark.parametrize(("stdout", "running"), [
    ('"reaper.exe","102204","Console","1","89,012 K"\r\n', True),
    ('"REAPER.exe","5","Console","1","9 K"\r\n', True),
    ("INFO: No tasks are running which match the specified criteria.\r\n", False)])
def test_reaper_running_reads_tasklist_on_windows(stdout, running):
    def fake_run(cmd, **_kw):
        assert cmd[0] == "tasklist"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    assert launch.reaper_running(system="Windows", run=fake_run) is running


def test_reaper_running_uses_pgrep_elsewhere():
    seen = []

    def fake_run(cmd, **_kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
    assert launch.reaper_running(system="Darwin", run=fake_run) is False
    assert seen[0][0] == "pgrep"


def test_reaper_exe_from_the_env(monkeypatch, exe):
    monkeypatch.setenv(launch.REAPER_EXE_ENV, str(exe))
    assert launch.find_reaper_exe(system="Windows") == exe


def test_reaper_exe_from_the_windows_install_folder(monkeypatch, tmp_path):
    monkeypatch.delenv(launch.REAPER_EXE_ENV, raising=False)
    program_files = tmp_path / "Program Files"
    target = program_files / "REAPER (x64)" / "reaper.exe"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("ProgramW6432", str(tmp_path / "nowhere"))
    assert launch.find_reaper_exe(system="Windows", running_path=lambda: None) == target


def test_reaper_exe_falls_back_to_the_running_program(monkeypatch, tmp_path, exe):
    monkeypatch.delenv(launch.REAPER_EXE_ENV, raising=False)
    for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        monkeypatch.setenv(var, str(tmp_path / "nowhere"))
    assert launch.find_reaper_exe(system="Windows", running_path=lambda: exe) == exe


# -- start_bridge ---------------------------------------------------------------------- #

def test_an_alive_bridge_is_left_alone(tmp_path, exe, script):
    launcher = Launcher()
    out = _start(tmp_path, alive=lambda: True, exe=exe, script=script, launcher=launcher)
    assert out["status"] == "already_running" and launcher.commands == []


def test_reaper_is_never_opened_by_orpheus(tmp_path, exe, script):
    launcher = Launcher()
    out = _start(tmp_path, running=lambda: False, exe=exe, script=script, launcher=launcher)
    assert out["status"] == "reaper_not_open" and launcher.commands == []
    assert "Open REAPER" in out["message"]


def test_a_missing_bridge_script_says_to_install(tmp_path, exe):
    out = _start(tmp_path, exe=exe, script=tmp_path / "missing.lua")
    assert out["status"] == "not_installed" and "install-bridge" in out["message"]


def test_a_missing_reaper_program_names_the_setting(tmp_path, script):
    out = _start(tmp_path, exe=None, script=script)
    assert out["status"] == "no_reaper_exe" and launch.REAPER_EXE_ENV in out["message"]


def test_the_open_reaper_is_asked_to_run_the_bridge(tmp_path, exe, script):
    state = {"alive": False}
    launcher = Launcher(on_launch=lambda: state.update(alive=True))
    out = _start(tmp_path, alive=lambda: state["alive"], exe=exe, script=script,
                 launcher=launcher)
    assert out["status"] == "started"
    assert launcher.commands == [[str(exe), "-nonewinst", str(script)]]


def test_no_heartbeat_after_the_launch_is_reported(tmp_path, exe, script):
    out = _start(tmp_path, exe=exe, script=script, wait_s=0.05)
    assert out["status"] == "no_heartbeat" and "Actions" in out["message"]


# -- automatic start from a bridge call -------------------------------------------------- #

@pytest.fixture
def autostart_on(monkeypatch):
    monkeypatch.setenv(launch.AUTOSTART_ENV, "1")
    launch.reset_backoff()
    yield
    launch.reset_backoff()


def test_a_call_starts_the_bridge_when_reaper_is_open(tmp_path, monkeypatch, autostart_on):
    with FakeReaperBridge(tmp_path, make_handlers(FakeReaperProject()), beat=False) as fake:
        calls = []

        def fake_start(bridge_dir, **_kw):
            calls.append(Path(bridge_dir))
            fake._beat = True  # the bridge comes up, as REAPER would run it
            deadline = time.monotonic() + 2
            while not BridgeClient(bridge_dir=tmp_path).is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            return {"status": "started", "message": "started"}

        monkeypatch.setattr(launch, "start_bridge", fake_start)
        info = BridgeClient(bridge_dir=tmp_path).call("get_project_info")
    assert calls == [tmp_path] and "tempo" in info


def test_autostart_switched_off_fails_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv(launch.AUTOSTART_ENV, "0")
    monkeypatch.setattr(launch, "start_bridge", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(BridgeTimeout, match="start-bridge"):
        BridgeClient(bridge_dir=tmp_path).call("get_project_info")


def test_a_failed_start_is_not_retried_on_every_call(tmp_path, monkeypatch, autostart_on):
    calls = []

    def fake_start(bridge_dir, **_kw):
        calls.append(bridge_dir)
        return {"status": "reaper_not_open", "message": "Open REAPER first."}

    monkeypatch.setattr(launch, "start_bridge", fake_start)
    client = BridgeClient(bridge_dir=tmp_path)
    for _ in range(3):
        with pytest.raises(BridgeTimeout, match="Open REAPER first"):
            client.call("get_project_info")
    assert len(calls) == 1


async def test_connection_status_tries_to_start_the_bridge(tmp_path, monkeypatch, autostart_on):
    from orpheus_mcp.registry import register_tools

    monkeypatch.setattr("orpheus_mcp.bridge.client.DEFAULT_BRIDGE_DIR", tmp_path)
    monkeypatch.setattr(launch, "start_bridge",
                        lambda *a, **k: {"status": "reaper_not_open",
                                         "message": "REAPER isn't open. Open REAPER."})
    mcp = FastMCP(name="launch")
    register_tools(mcp, profile="default")
    async with Client(mcp) as c:
        res = (await c.call_tool("get_connection_status", {})).structured_content
    assert res["connected"] is False
    assert res["autostart"] == {"status": "reaper_not_open",
                                "message": "REAPER isn't open. Open REAPER."}
