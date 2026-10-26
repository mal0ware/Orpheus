"""Start the bridge in a REAPER that is already open (auto-start, 2026-10-25).

REAPER hands a script named on its command line to the running copy when given
``-nonewinst`` (verified live on REAPER 7.83, Windows): ``REAPER.exe -nonewinst
<Scripts/orpheus/orpheus_bridge.lua>`` starts the bridge with no clicks. Orpheus never opens
REAPER itself; once REAPER is open, ``install-bridge``'s ``__startup.lua`` block or this
function brings the bridge up. A second copy takes over from the first (the bridge's
newest-copy-wins rule), so starting it twice is harmless.

``ORPHEUS_AUTOSTART_BRIDGE=0`` stops bridge calls from trying this on their own;
``ORPHEUS_REAPER_EXE`` names the REAPER program when it is not in the usual place.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

AUTOSTART_ENV = "ORPHEUS_AUTOSTART_BRIDGE"
REAPER_EXE_ENV = "ORPHEUS_REAPER_EXE"
RETRY_AFTER_S = 30.0  # a failed automatic start is not retried on every call
WAIT_S = 10.0

_backoff_lock = threading.Lock()
_last_attempt: dict[str, float] = {}
_last_result: dict[str, dict] = {}


def autostart_enabled() -> bool:
    return os.environ.get(AUTOSTART_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def reaper_running(*, system: str | None = None, run: Callable[..., subprocess.CompletedProcess]
                   = subprocess.run) -> bool:
    """Whether a REAPER process is open (tasklist on Windows, pgrep elsewhere)."""
    system = system or platform.system()
    try:
        if system == "Windows":
            out = run(["tasklist", "/FI", "IMAGENAME eq reaper.exe", "/NH", "/FO", "CSV"],
                      capture_output=True, text=True, timeout=10, check=False)
            return '"reaper.exe"' in (out.stdout or "").lower()
        name = "REAPER" if system == "Darwin" else "reaper"
        out = run(["pgrep", "-x", name], capture_output=True, text=True, timeout=10,
                  check=False)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _running_reaper_path() -> Path | None:
    """The open REAPER's program path on Windows (covers portable installs)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process reaper -ErrorAction SilentlyContinue | Select-Object -First 1).Path"],
            capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or "").strip().splitlines()
    return Path(line[0]) if line and Path(line[0]).is_file() else None


def find_reaper_exe(*, system: str | None = None,
                    running_path: Callable[[], Path | None] = _running_reaper_path
                    ) -> Path | None:
    """The REAPER program: ``ORPHEUS_REAPER_EXE``, the usual install folders, then (Windows)
    the path of the REAPER that is running."""
    env = os.environ.get(REAPER_EXE_ENV)
    if env and Path(env).is_file():
        return Path(env)
    system = system or platform.system()
    candidates: list[Path] = []
    if system == "Windows":
        for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            base = os.environ.get(var)
            if base:
                candidates += [Path(base) / "REAPER (x64)" / "reaper.exe",
                               Path(base) / "REAPER" / "reaper.exe"]
    elif system == "Darwin":
        candidates += [Path("/Applications/REAPER.app/Contents/MacOS/REAPER"),
                       Path("/Applications/REAPER64.app/Contents/MacOS/REAPER")]
    else:
        which = shutil.which("reaper")
        candidates += [Path(which)] if which else []
        candidates += [Path.home() / "opt" / "REAPER" / "reaper", Path("/opt/REAPER/reaper")]
    for c in candidates:
        if c.is_file():
            return c
    return running_path() if system == "Windows" else None


def installed_script() -> Path | None:
    from orpheus_mcp.install import find_reaper_resource_dir

    resource = find_reaper_resource_dir()
    return resource / "Scripts" / "orpheus" / "orpheus_bridge.lua" if resource else None


def _launch(cmd: list[str]) -> None:
    # Detached: the launcher hands the script to the open REAPER and exits.
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,  # noqa: S603
                     stderr=subprocess.DEVNULL, close_fds=True)


def start_bridge(bridge_dir: Path | None = None, *, wait_s: float = WAIT_S,
                 script: Path | None = None,
                 find_exe: Callable[[], Path | None] | None = None,
                 running: Callable[[], bool] | None = None,
                 launch: Callable[[list[str]], None] | None = None,
                 is_alive: Callable[[], bool] | None = None,
                 poll_s: float = 0.2) -> dict:
    """Ask the open REAPER to run the installed bridge, then wait for its heartbeat.
    Returns ``{"status", "message"}``; status is one of ``already_running``, ``started``,
    ``reaper_not_open``, ``not_installed``, ``no_reaper_exe``, ``no_heartbeat``."""
    if is_alive is None:
        from orpheus_mcp.bridge.client import BridgeClient

        client = BridgeClient(bridge_dir=bridge_dir)
        is_alive = client.is_alive
    if is_alive():
        return {"status": "already_running", "message": "The bridge is already running."}
    if not (running or reaper_running)():
        return {"status": "reaper_not_open",
                "message": "REAPER isn't open. Open REAPER and the bridge starts with it "
                           "(after `orpheus-mcp install-bridge`)."}
    script = script if script is not None else installed_script()
    if script is None or not Path(script).is_file():
        return {"status": "not_installed",
                "message": "The bridge isn't installed in REAPER. Run `orpheus-mcp "
                           "install-bridge`, then `orpheus-mcp start-bridge`."}
    exe = (find_exe or find_reaper_exe)()
    if exe is None:
        return {"status": "no_reaper_exe",
                "message": f"Couldn't find the REAPER program. Set {REAPER_EXE_ENV} to its "
                           "path, or restart REAPER (the bridge starts with it)."}
    (launch or _launch)([str(exe), "-nonewinst", str(script)])
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if is_alive():
            return {"status": "started", "message": "Started the bridge in the open REAPER."}
        time.sleep(poll_s)
    return {"status": "no_heartbeat",
            "message": "Asked REAPER to start the bridge, but no heartbeat came. Restart "
                       "REAPER (the bridge starts with it), or run orpheus_bridge.lua from "
                       "REAPER's Actions list (search 'orpheus_bridge')."}


def try_autostart(bridge_dir: Path) -> dict | None:
    """One automatic start attempt for a bridge call, at most every ``RETRY_AFTER_S`` per
    bridge folder; while backing off it returns the last attempt's result again. None when
    switched off."""
    if not autostart_enabled():
        return None
    key = str(Path(bridge_dir).resolve())
    with _backoff_lock:
        now = time.monotonic()
        if now - _last_attempt.get(key, float("-inf")) < RETRY_AFTER_S:
            return _last_result.get(key)
        _last_attempt[key] = now
    result = start_bridge(bridge_dir)
    with _backoff_lock:
        if result["status"] in ("started", "already_running"):
            _last_attempt.pop(key, None)
            _last_result.pop(key, None)
        else:
            _last_result[key] = result
    return result


def reset_backoff() -> None:
    with _backoff_lock:
        _last_attempt.clear()
        _last_result.clear()
