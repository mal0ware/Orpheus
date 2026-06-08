"""The RECOMMEND engine (north-star half 2). Diffs the current project against a cached
style fingerprint and emits a reason-annotated EditPlan. READ-ONLY — it decides, never mutates.
The approval gate lives between this and apply_changes."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.models import EditPlan

_RO = {"readOnlyHint": True}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=_RO)
    def list_style_fingerprints() -> list[str]:
        """List cached target fingerprints (e.g. 'classical', 'hiphop', 'dominic-fike-sunburn').
        Each is the analysis pipeline run over 3–5 per-era reference tracks."""
        raise NotImplementedError("M3 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def explain_style() -> dict:
        """"Why does this sound like X?" — run feature thresholds over the current project and
        return human-readable reasons (each matched threshold = one reason).

        This is the v0.1 headline capability: understanding without transformation.
        """
        raise NotImplementedError("M2 — see docs/roadmap.md")

    @mcp.tool(annotations=_RO)
    def recommend_changes(target_style: str) -> EditPlan:
        """Diff the current project against a target fingerprint and return an EditPlan —
        a list of ProposedEdit{target, action, reason, params} across the v1 dimensions
        (tempo, key, harmony, instrumentation, mastering-match), each annotated with a
        human-readable reason. Read-only: surface these reasons to the user for approval,
        THEN call apply_changes. Includes honest caveats (detection confidence, taste calls).
        """
        raise NotImplementedError("M3 — see docs/roadmap.md")
