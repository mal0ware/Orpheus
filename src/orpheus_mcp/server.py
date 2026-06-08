"""Orpheus FastMCP server entrypoint.

Run directly with ``orpheus-mcp`` (console script), ``fastmcp run``, or
``fastmcp dev`` (MCP Inspector). The actual tools live in :mod:`orpheus_mcp.tools`
and are wired up through :mod:`orpheus_mcp.registry`.
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

from orpheus_mcp.registry import register_tools

INSTRUCTIONS = """\
Orpheus lets you analyze, explain, and reshape music inside a running REAPER project.

Workflow:
  1. Always call get_connection_status first to confirm REAPER + the bridge are live.
  2. To UNDERSTAND a track, use the analyze_* tools, then explain_style for
     "why does this sound like X".
  3. To TRANSFORM a track, call recommend_changes (read-only) to get an EditPlan of
     reasoned changes, surface those reasons to the user, and only then call
     apply_changes with the approved plan.

Always speak musical time in BEATS, never ticks. Surface harmony-detection confidence
to the user and let them correct it — key/chord detection is probabilistic, especially
on drum-heavy material.
"""

# The FastMCP instance fastmcp.json points at.
mcp: FastMCP = FastMCP(name="Orpheus", instructions=INSTRUCTIONS)

# Profile is selectable via env so a client can request the lean "explain" surface.
_PROFILE = os.environ.get("ORPHEUS_TOOLSET", "default")
register_tools(mcp, profile=_PROFILE)


def main() -> None:
    """Console-script entrypoint.

    ``orpheus-mcp``                 → run the MCP server (stdio transport)
    ``orpheus-mcp install-bridge``  → copy the Lua bridge into REAPER's Scripts folder
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "install-bridge":
        from orpheus_mcp.install import install_bridge

        dest = install_bridge()
        print(f"✓ Installed Orpheus bridge → {dest}")
        print("Next, in REAPER: Actions → Show action list → Run ReaScript → select "
              "orpheus_bridge.lua (or assign it a toolbar button).")
        return

    mcp.run()


if __name__ == "__main__":
    main()
