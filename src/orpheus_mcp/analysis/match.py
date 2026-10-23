"""The mix/master 'sound like' layer (M3, not built).

Plan: render the REAPER master to WAV, match it against a reference WAV the user owns, and bake
the resulting curve onto a master-bus ReaEQ + limiter. Matchering (GPL-3.0) is the likely engine,
but it is NOT a dependency of this MIT package: when this is built it comes back as an optional
``[master]`` extra the user installs on purpose (spec 2026-10-23 §3).
"""

from __future__ import annotations


def match_master(target_wav: str, reference_wav: str, out_wav: str) -> dict:
    """Run Matchering's mg.process(target, reference) and return the derived correction
    (EQ curve + gain/limiter settings) to bake onto the master bus.

    Implementation (M3): behind the optional ``[master]`` extra,
    ``import matchering as mg; mg.process(target, reference, results=[...])``, then translate the
    matched curve into ReaEQ band gains.
    """
    raise NotImplementedError("M3 — see docs/roadmap.md")
