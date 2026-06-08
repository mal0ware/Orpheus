"""Plugin-aware deterministic mixing (v1-stretch): the genre×role dB table + chain-offset
calibration, so AI mixes land at musical levels instead of clipping."""

from __future__ import annotations

from fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations={"destructiveHint": True})
    def apply_mix_balance(genre: str) -> dict:
        """fader_db = mix_target(genre, role) − chain_offset(track FX), clamped to fader range."""
        raise NotImplementedError("M4 — see docs/roadmap.md")

    @mcp.tool(annotations={"readOnlyHint": True})
    def list_installed_fx(kind: str | None = None) -> dict:
        """Parse REAPER's plugin cache .ini files so recommendations only suggest owned plugins."""
        raise NotImplementedError("M4 — see docs/roadmap.md")
