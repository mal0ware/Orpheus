"""`orpheus-mcp install-bridge` — copy the Lua bridge into REAPER's Scripts folder.

The in-REAPER half of the bridge is a single Lua script. This locates REAPER's resource
directory per-OS and copies the bundled script to a stable path
(``<resource>/Scripts/orpheus/orpheus_bridge.lua``) so the server config never changes.
Pure stdlib — importable and testable without fastmcp or a running REAPER.
"""

from __future__ import annotations

import os
import platform
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


def install_bridge(resource_dir: Path | None = None) -> Path:
    """Copy the bundled Lua bridge into REAPER's Scripts folder; return the destination path.

    Idempotent (re-running overwrites the script in place). Raises FileNotFoundError if no
    REAPER resource directory can be found and none was given explicitly.
    """
    if resource_dir is None:
        resource_dir = find_reaper_resource_dir()
    if resource_dir is None:
        raise FileNotFoundError(
            "Could not find a REAPER resource directory. Is REAPER installed and run at least "
            "once? Pass resource_dir explicitly, or set it via the REAPER preferences path."
        )

    dest_dir = Path(resource_dir) / "Scripts" / "orpheus"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "orpheus_bridge.lua"
    shutil.copyfile(BUNDLED_LUA, dest)
    return dest
