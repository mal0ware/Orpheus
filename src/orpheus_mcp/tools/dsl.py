"""NL ergonomics (v1-stretch): fuzzy resolvers + conversational memory. Thin orchestrators
over primitives — the pattern from shiehn's DSL layer."""

from __future__ import annotations

from fastmcp import FastMCP

_RO = {"readOnlyHint": True}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_RO)
    def resolve_track(query: str) -> dict:
        """Fuzzy-match a track by name/role with disambiguation ('I found 2 drum tracks: which?')."""
        raise NotImplementedError("M4 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def resolve_time(expr: str) -> dict:
        """Parse '8 bars' / '1:30' / keywords into concrete tempo-aware positions."""
        raise NotImplementedError("M4 — see docs/roadmap.md")
