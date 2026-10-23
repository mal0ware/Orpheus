"""The server's MCP instructions come from data/instructions/00_core.md and only name real tools."""
from __future__ import annotations

import re

from fastmcp import FastMCP

from orpheus_mcp.registry import register_tools


async def _tool_names(profile: str) -> set[str]:
    m = FastMCP(name="probe")
    register_tools(m, profile=profile)
    return {t.name for t in await m.list_tools()}


def test_instructions_load_from_core_file():
    from orpheus_mcp import server

    text = server.load_instructions()
    assert text.startswith("# Orpheus")
    assert server.mcp.instructions == text


async def test_instructions_name_only_default_profile_tools():
    from orpheus_mcp import server

    default = await _tool_names("default")
    every_tool = await _tool_names("full")
    named = set(re.findall(r"`([a-z][a-z0-9_]*)`", server.load_instructions()))
    # Only identifiers that ARE tool names somewhere (incl. stubs) are checked.
    mentioned_tools = named & every_tool
    assert mentioned_tools, "instructions should name at least one tool"
    assert mentioned_tools <= default, f"not in default profile: {mentioned_tools - default}"
