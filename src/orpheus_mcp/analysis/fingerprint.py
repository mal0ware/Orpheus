"""Style fingerprints + the diff that powers recommend_changes.

A fingerprint is the analysis pipeline run over 3–5 per-era reference tracks, aggregated
into one profile. `diff_to_edit_plan` is the heart of the differentiator: compare the
current Spec to a target fingerprint and turn each significant delta into a ProposedEdit
whose `reason` IS the delta in plain English.
"""

from __future__ import annotations

from orpheus_mcp.models import CompositionSpec, EditPlan


def explain_features(spec: CompositionSpec) -> list[str]:
    """guess_genre()-style: which measurable feature thresholds does this track match?

    Each matched threshold becomes one human-readable reason, e.g.
    "92 BPM + sub-bass at 19%% + low onset density → hip-hop signature". This powers
    "why does this sound like X" with no transformation (the v0.1 capability).
    """
    raise NotImplementedError("M2 — see docs/roadmap.md")


def diff_to_edit_plan(current: CompositionSpec, fingerprint: dict, intent: str) -> EditPlan:
    """Diff current vs a target fingerprint across the v1 dimensions → reason-annotated EditPlan.

    Dimensions (v1): tempo, key/mode, harmonic vocabulary, instrumentation, mix/master.
    Each delta over threshold → a ProposedEdit. Always attaches honest caveats (low
    detection confidence, taste-based genre→param mappings).
    """
    raise NotImplementedError("M3 — see docs/roadmap.md")
