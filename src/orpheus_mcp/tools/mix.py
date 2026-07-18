"""Plugin-aware deterministic mixing (v1-stretch): the genre×role dB table + chain-offset
calibration, so AI mixes land at musical levels instead of clipping."""

from __future__ import annotations

from fastmcp import FastMCP


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    if not include_stubs:
        return  # every tool below is a stub - see the registry honesty rule

    @mcp.tool(annotations={"destructiveHint": True})
    def apply_mix_balance(genre: str) -> dict:
        """[NOT IMPLEMENTED]
        fader_db = mix_target(genre, role) − chain_offset(track FX), clamped to fader range."""
        raise NotImplementedError("M4 — see docs/roadmap.md")

    # list_installed_fx pulled forward from Slice 2 — see orpheus_mcp.tools.instruments.
