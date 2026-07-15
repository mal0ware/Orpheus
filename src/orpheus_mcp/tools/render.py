"""Headless render/bounce — for auditioning and for feeding the audio analyzer."""

from __future__ import annotations

from fastmcp import FastMCP


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    if not include_stubs:
        return  # every tool below is a stub - see the registry honesty rule

    @mcp.tool(annotations={"destructiveHint": True})
    def render_project(out_path: str, fmt: str = "wav") -> dict:
        """[NOT IMPLEMENTED]
        Headless render to WAV/MP3 (RENDER_FILE + format/bounds + Main_OnCommand(41824))."""
        raise NotImplementedError("M1 — see docs/roadmap.md")

    @mcp.tool(annotations={"destructiveHint": True})
    def render_stems(out_dir: str) -> dict:
        """[NOT IMPLEMENTED]
        Render per-track stems by sequential solo — feeds analyze_audio_character."""
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations={"readOnlyHint": True})
    def render_and_audit() -> dict:
        """[NOT IMPLEMENTED] Render, then measure LUFS/spectrum in one call so the agent self-checks
        ('loud enough? too bright?') and iterates without a human ear."""
        raise NotImplementedError("M2 — see docs/roadmap.md")
