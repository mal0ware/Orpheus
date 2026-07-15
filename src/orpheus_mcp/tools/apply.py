"""The gated transform step. Takes an approved EditPlan and executes it through the bridge
in a SINGLE undo block, so the whole transformation is one Ctrl+Z. destructiveHint=True."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.models import EditPlan

_DESTRUCTIVE = {"destructiveHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    if not include_stubs:
        return  # every tool below is a stub - see the registry honesty rule

    @mcp.tool(annotations=_DESTRUCTIVE)
    def apply_changes(plan: EditPlan) -> dict:
        """[NOT IMPLEMENTED]
        Execute an approved EditPlan: symbolic edits (transpose, rewrite progression MIDI,
        swap/add FX, retarget tempo, write grooved patterns), all wrapped in one
        Undo_BeginBlock/EndBlock so a single Ctrl+Z reverts everything.

        Pass back the exact plan the user approved — do not re-derive it here.
        """
        raise NotImplementedError("M3 — see docs/roadmap.md")

    @mcp.tool(annotations=_DESTRUCTIVE)
    def apply_master_match(fingerprint: str) -> dict:
        """[NOT IMPLEMENTED]
        The spectral half: render the master → Matchering against the fingerprint's
        reference WAV → bake the matched curve onto a master-bus ReaEQ + limiter."""
        raise NotImplementedError("M3 — see docs/roadmap.md")
