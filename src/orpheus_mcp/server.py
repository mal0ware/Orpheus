"""Orpheus FastMCP server entrypoint.

Run directly with ``orpheus-mcp`` (console script), ``fastmcp run``, or
``fastmcp dev`` (MCP Inspector). The actual tools live in :mod:`orpheus_mcp.tools`
and are wired up through :mod:`orpheus_mcp.registry`.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

from orpheus_mcp.registry import register_tools

_CORE = Path("instructions") / "00_core.md"


def load_instructions() -> str:
    """Claude's operating instructions, read from ``data/instructions/00_core.md``.

    Installed wheel: ``orpheus_mcp/data/...`` (hatch force-include). Source checkout:
    ``<repo>/data/...``. One file, so what a maintainer edits is what Claude receives.
    """
    package_path = Path(__file__).resolve().parent / "data" / _CORE
    if package_path.exists():
        return package_path.read_text(encoding="utf-8")
    return (Path(__file__).resolve().parents[2] / "data" / _CORE).read_text(encoding="utf-8")


INSTRUCTIONS = load_instructions()

# The FastMCP instance fastmcp.json points at.
mcp: FastMCP = FastMCP(name="Orpheus", instructions=INSTRUCTIONS)

# Profile is selectable via env so a client can request the lean "explain" surface.
_PROFILE = os.environ.get("ORPHEUS_TOOLSET", "default")
register_tools(mcp, profile=_PROFILE)


def main() -> None:
    """Console-script entrypoint.

    ``orpheus-mcp``                    → run the MCP server (stdio transport)
    ``orpheus-mcp install-bridge``     → copy the Lua bridge into REAPER's Scripts folder
    ``orpheus-mcp fetch-models``       → download + hash-verify every model registry checkpoint
    ``orpheus-mcp fetch-models --pin`` → download + RECORD hashes (Task 0.3 pin workflow only)
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "install-bridge":
        from orpheus_mcp.install import install_bridge

        dest = install_bridge()
        # Windows consoles often use a legacy codepage (cp1252) that can't encode
        # "✓"/"→" — degrade to replacement chars rather than crash after a
        # successful install.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
        print(f"✓ Installed Orpheus bridge → {dest}")
        print("Next, in REAPER: Actions → Show action list → Run ReaScript → select "
              "orpheus_bridge.lua (or assign it a toolbar button).")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "fetch-models":
        from orpheus_mcp.analysis.fetch_models import (
            FetchModelsError,
            FfmpegMissing,
            fetch_models,
        )

        # Same cp1252-safe console degradation as install-bridge.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")

        pin_mode = "--pin" in sys.argv[2:]
        try:
            results = fetch_models(pin=pin_mode)
        except (FfmpegMissing, FetchModelsError) as exc:
            print(f"✗ fetch-models failed: {exc}")
            sys.exit(1)

        for model_id, status in results.items():
            print(f"{model_id}: {status}")
        print(f"✓ fetch-models complete ({len(results)} registry entries).")
        return

    mcp.run()


if __name__ == "__main__":
    main()
