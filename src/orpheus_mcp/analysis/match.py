"""The mix/master 'sound like' layer — a thin wrapper over Matchering.

This dimension is a solved stitch-together: render the REAPER master to WAV, match it
against the reference WAV, and bake the resulting curve onto a master-bus ReaEQ + limiter.
"""

from __future__ import annotations


def match_master(target_wav: str, reference_wav: str, out_wav: str) -> dict:
    """Run Matchering's mg.process(target, reference) and return the derived correction
    (EQ curve + gain/limiter settings) to bake onto the master bus.

    Implementation (M3): ``import matchering as mg; mg.process(target, reference, results=[...])``,
    then translate the matched curve into ReaEQ band gains.
    """
    raise NotImplementedError("M3 — see docs/roadmap.md")
