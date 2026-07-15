"""The RECOMMEND engine (north-star half 2). Diffs the current project against a cached
style fingerprint and emits a reason-annotated EditPlan. READ-ONLY — it decides, never mutates.
The approval gate lives between this and apply_changes."""

from __future__ import annotations

from fastmcp import FastMCP

from orpheus_mcp.analysis.fingerprint import (
    explain_against_fingerprint,
    list_fingerprints,
    load_fingerprint,
)
from orpheus_mcp.models import EditPlan, StyleExplanation
from orpheus_mcp.tools.analyze import build_current_spec

_RO = {"readOnlyHint": True}


def register(mcp: FastMCP, *, include_stubs: bool = False) -> None:
    @mcp.tool(annotations=_RO)
    def list_style_fingerprints() -> list[str]:
        """List cached target fingerprints (e.g. 'classical', 'hiphop', 'dominic-fike-sunburn').
        Each is the analysis pipeline run over 3–5 per-era reference tracks."""
        return list_fingerprints()

    @mcp.tool(annotations=_RO)
    def explain_style(style: str, wav_path: str | None = None) -> StyleExplanation:
        """"Why does/doesn't this sound like X?" — the v0.1 headline capability.

        Snapshots the current project over the bridge, diffs it against the named
        fingerprint, and returns one FeatureDelta per comparable dimension — every claim
        is a measured-vs-fingerprint delta with both numbers in it, never a vibe. Pass a
        rendered WAV to include the audio dimensions (loudness/tonal balance); without
        one they are declared as caveats.
        """
        fingerprint = load_fingerprint(style)
        return explain_against_fingerprint(build_current_spec(wav_path), fingerprint)

    if not include_stubs:
        return

    @mcp.tool(annotations=_RO)
    def recommend_changes(target_style: str) -> EditPlan:
        """[NOT IMPLEMENTED]
        Diff the current project against a target fingerprint and return an EditPlan —
        a list of ProposedEdit{target, action, reason, params} across the v1 dimensions
        (tempo, key, harmony, instrumentation, mastering-match), each annotated with a
        human-readable reason. Read-only: surface these reasons to the user for approval,
        THEN call apply_changes. Includes honest caveats (detection confidence, taste calls).
        """
        raise NotImplementedError("M3 — see docs/roadmap.md")
