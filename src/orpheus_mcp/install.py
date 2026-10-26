"""`orpheus-mcp install-bridge` — copy the Lua bridge into REAPER's Scripts folder.

The in-REAPER half of the bridge is a single Lua script. This locates REAPER's resource
directory per-OS and copies the bundled script to a stable path
(``<resource>/Scripts/orpheus/orpheus_bridge.lua``) so the server config never changes.
Pure stdlib — importable and testable without fastmcp or a running REAPER.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
from pathlib import Path

BUNDLED_LUA = Path(__file__).resolve().parent / "bridge" / "lua" / "orpheus_bridge.lua"


def find_reaper_resource_dir(system: str | None = None, home: Path | None = None) -> Path | None:
    """Return REAPER's resource directory for this OS, or None if it isn't there.

    macOS:   ~/Library/Application Support/REAPER
    Windows: %APPDATA%/REAPER
    Linux:   $XDG_CONFIG_HOME/REAPER or ~/.config/REAPER
    """
    system = system or platform.system()
    home = Path(home) if home else Path.home()

    if system == "Darwin":
        candidate = home / "Library" / "Application Support" / "REAPER"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        candidate = base / "REAPER"
    else:  # Linux / other
        xdg = os.environ.get("XDG_CONFIG_HOME")
        candidate = (Path(xdg) / "REAPER") if xdg else home / ".config" / "REAPER"

    return candidate if candidate.is_dir() else None


def install_bridge(resource_dir: Path | None = None, *, autostart: bool = True) -> Path:
    """Copy the bundled Lua bridge into REAPER's Scripts folder; return the destination path.

    With ``autostart`` (the default) it also adds a marked block to REAPER's
    ``Scripts/__startup.lua`` so the bridge starts whenever REAPER opens (see
    ``install_autostart``). Idempotent (re-running overwrites the script and refreshes the
    block). Raises FileNotFoundError if no REAPER resource directory can be found and none was
    given explicitly.
    """
    resource = _resource_or_raise(resource_dir)
    dest_dir = resource / "Scripts" / "orpheus"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "orpheus_bridge.lua"
    shutil.copyfile(BUNDLED_LUA, dest)
    if autostart:
        install_autostart(resource)
    return dest


def _resource_or_raise(resource_dir: Path | None) -> Path:
    if resource_dir is None:
        resource_dir = find_reaper_resource_dir()
    if resource_dir is None:
        raise FileNotFoundError(
            "Could not find a REAPER resource directory. Is REAPER installed and run at least "
            "once? Pass resource_dir explicitly, or set it via the REAPER preferences path."
        )
    return Path(resource_dir)


# -- auto-start (2026-10-25) ------------------------------------------------------------- #
# REAPER runs Scripts/__startup.lua every time it opens. Orpheus owns only the block between
# these two markers; everything else in that file is the owner's and is kept byte for byte.

STARTUP_BEGIN = ("-- >>> Orpheus bridge auto-start (added by `orpheus-mcp install-bridge`; "
                 "`orpheus-mcp remove-autostart` takes it out) >>>")
STARTUP_END = "-- <<< Orpheus bridge auto-start <<<"
_STARTUP_BODY = [
    "do",
    '  local bridge = reaper.GetResourcePath() .. "/Scripts/orpheus/orpheus_bridge.lua"',
    "  if reaper.file_exists(bridge) then",
    "    local ok, err = pcall(dofile, bridge)",
    "    if not ok then",
    '      reaper.ShowConsoleMsg("Orpheus bridge did not start: " .. tostring(err) .. "\\n")',
    "    end",
    "  end",
    "end",
]
_BLOCK_RE = re.compile(
    r"(?:\r?\n)?" + re.escape(STARTUP_BEGIN) + r".*?" + re.escape(STARTUP_END) + r"(?:\r?\n)?",
    re.S)


def startup_script(resource_dir: Path) -> Path:
    return Path(resource_dir) / "Scripts" / "__startup.lua"


def _read_raw(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", newline="") as f:
        return f.read()


def install_autostart(resource_dir: Path | None = None) -> Path:
    """Add (or refresh) Orpheus's block in REAPER's ``Scripts/__startup.lua``, keeping the
    rest of the file and its line endings. The block runs the installed bridge if it is there
    and prints to REAPER's console, instead of failing REAPER's startup, if it can't load."""
    path = startup_script(_resource_or_raise(resource_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    theirs = _BLOCK_RE.sub("", _read_raw(path))
    nl = "\r\n" if "\r\n" in theirs else "\n"
    if theirs and not theirs.endswith(("\n", "\r")):
        theirs += nl
    block = nl.join([STARTUP_BEGIN, *_STARTUP_BODY, STARTUP_END]) + nl
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(theirs + (nl if theirs.strip() else "") + block)
    return path


def remove_autostart(resource_dir: Path | None = None) -> bool:
    """Take Orpheus's block out of ``__startup.lua``. The file is deleted only if nothing of
    the owner's is left in it. Returns whether a block was there."""
    path = startup_script(_resource_or_raise(resource_dir))
    text = _read_raw(path)
    rest, n = _BLOCK_RE.subn("", text)
    if not n:
        return False
    if rest.strip():
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write(rest)
    else:
        path.unlink()
    return True
