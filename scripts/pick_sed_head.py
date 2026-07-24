"""``scripts/pick_sed_head.py`` — SED-head selection + model registry pin run (d), §15.6.

Scores each candidate PANNs SED head (``DecisionLevelMax`` / ``DecisionLevelAvg`` /
``DecisionLevelAtt``) against the desktop fixture set (``~/.orpheus_fixtures``, Task 0.4):
for every fixture, build a ``Timeline`` for each head over ``mix.flac`` and compare its
per-role predicted-active frames (``activation > CFG.TIMELINE_ACTIVITY_THRESHOLD``) against
a truth-active frame series built from each role's own stem (``stem RMS > -50 dBFS``, 1 s
frames matching the Timeline hop — the brief's rule). Frame-level F1 is pooled (micro-
averaged: TP/FP/FN summed across every frame of every scored fixture) per role and overall;
ties on the overall score break to ``DecisionLevelAtt`` (§15.6, ``TIE_BREAK_HEAD`` below).

Only roles with a registry ``tagger_classes`` mapping are scored (``ROLE_CLASS_MAP``) — the
fixture set's ``lead`` role (rendered via a solo flute sample) has no AudioSet class mapped
into any registry entry's ``tagger_classes`` today (registry targets are vocals/drums/bass/
guitar/piano/other; nothing maps a solo-melody instrument), so it is honestly excluded from
scoring rather than silently guessed at.

Known, honestly-stated scope limit (mirrors ``analysis/timeline.py``'s ``PannsTagger``
docstring): the installed ``panns-inference`` package loads ONLY the ``Cnn14_DecisionLevelMax``
architecture regardless of the ``head`` string passed to ``PannsTagger`` — the Avg/Att
checkpoints are not part of that PyPI distribution. All three "heads" therefore score
identically in this run (same weights, same forward pass), which is a genuine three-way tie
resolved by the locked §15.6 rule, not a fabricated result — this is recorded plainly in the
pin report rather than hidden behind a single number.

Usage:
    uv run --extra analysis python scripts/pick_sed_head.py                # run + write pins
    uv run --extra analysis python scripts/pick_sed_head.py --dry-run      # print table only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from orpheus_mcp.analysis.cachefs import Pcm, decode
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.model_registry import RegistryError
from orpheus_mcp.models import Timeline

# The three candidate PANNs SED heads (§15.6). Order matters for the tie-break fallback.
CANDIDATE_HEADS: tuple[str, ...] = ("DecisionLevelMax", "DecisionLevelAvg", "DecisionLevelAtt")
TIE_BREAK_HEAD = "DecisionLevelAtt"

# Truth-activity rule (the brief): a role is truth-active in a 1 s frame when its stem's
# RMS in that frame exceeds this dBFS floor.
TRUTH_ACTIVITY_DBFS = -50.0
_LOCKED_HOP_S = 1.0  # matches analysis/timeline.py's locked Timeline hop

# Role -> the registry's mapped AudioSet class family that role's stem should activate.
# Kept in sync with data/model_registry.json's tagger_classes unions by hand (small, stable
# vocabulary) rather than derived, so a role with NO honest mapping (see module docstring)
# is a visible omission here, not a silent `if class in registry: ...` guess.
ROLE_CLASS_MAP: dict[str, list[str]] = {
    "drums": ["Drum kit", "Drum", "Snare drum", "Hi-hat", "Cymbal", "Bass drum", "Drum machine"],
    "bass": ["Bass guitar"],
    "chords": ["Piano"],
    "vocals": ["Singing", "Speech"],
    # "lead" intentionally absent -- see module docstring.
}

_PIN_REPORT = Path("docs/pins/2026-07-model-pins.md")
_PIN_RUN_D_HEADER = "## Pin run (d): SED head selection"
_TAGGER_ENTRY_ID = "tagger/panns-sed"


class FixtureSetStale(RuntimeError):
    """The desktop fixture set no longer matches its manifest (a file changed/went missing,
    or the recorded ``fixture_set_hash`` doesn't recompute) — regenerate it via
    ``scripts/make_fixtures.py`` (Task 0.4); this script never patches or regenerates it."""


# --------------------------------------------------------------------------- #
# Frame-level F1 math (CI-safe, pure) — hand-built arrays exercise this directly.
# --------------------------------------------------------------------------- #


def frame_counts(pred: Sequence[bool], truth: Sequence[bool]) -> tuple[int, int, int]:
    """(TP, FP, FN) over two same-length boolean frame series."""
    if len(pred) != len(truth):
        raise ValueError(f"pred/truth length mismatch: {len(pred)} vs {len(truth)}")
    tp = fp = fn = 0
    for p, t in zip(pred, truth, strict=True):
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif t and not p:
            fn += 1
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """F1 = 2*TP / (2*TP + FP + FN); defined as 1.0 (trivial perfect agreement, not an
    undefined 0/0) when TP=FP=FN=0 -- every frame was a true negative, both sides agreed
    on 'inactive' throughout."""
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return (2 * tp) / denom


def frame_f1(pred: Sequence[bool], truth: Sequence[bool]) -> float:
    """Frame-level F1 between a predicted-active and a truth-active boolean series."""
    return f1_from_counts(*frame_counts(pred, truth))


def _add_counts(acc: dict[str, list[int]], key: str, tp: int, fp: int, fn: int) -> None:
    slot = acc.setdefault(key, [0, 0, 0])
    slot[0] += tp
    slot[1] += fp
    slot[2] += fn


# --------------------------------------------------------------------------- #
# Tie-break (§15.6)
# --------------------------------------------------------------------------- #


def pick_winner(
    f1_by_head: dict[str, float],
    *,
    candidates: Sequence[str] = CANDIDATE_HEADS,
    tie_break: str = TIE_BREAK_HEAD,
) -> str:
    """The candidate head (in ``candidates`` order) with the highest F1 in ``f1_by_head``.

    On an exact tie for the top score, §15.6 breaks toward ``tie_break`` when it is one of
    the tied heads; if ``tie_break`` itself was never scored (a checkpoint genuinely
    unavailable), the tie instead resolves deterministically to the first tied head in
    ``candidates`` order -- never an arbitrary dict-iteration pick.
    """
    available = [h for h in candidates if h in f1_by_head]
    if not available:
        raise ValueError("pick_winner: no candidate heads were scored")

    best = max(f1_by_head[h] for h in available)
    tied = [h for h in available if f1_by_head[h] == best]
    if len(tied) == 1:
        return tied[0]
    return tie_break if tie_break in tied else tied[0]


# --------------------------------------------------------------------------- #
# Truth-activity frames from a stem's audio (CI-safe math; a real Pcm at call time)
# --------------------------------------------------------------------------- #


def rms_dbfs(chunk: np.ndarray) -> float:
    """20*log10(RMS) of a float sample chunk; -inf for an empty or all-silent chunk."""
    if chunk.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))
    return 20.0 * math.log10(rms) if rms > 1e-12 else -math.inf


def activity_frames_from_mono(
    mono: np.ndarray,
    sample_rate: int,
    n_frames: int,
    *,
    hop_s: float = _LOCKED_HOP_S,
    threshold_dbfs: float = TRUTH_ACTIVITY_DBFS,
) -> list[bool]:
    """Per-``hop_s`` truth-activity series: frame ``i`` is active iff the mono chunk
    ``[i*hop_s, (i+1)*hop_s)`` (in samples, at ``sample_rate``) has RMS > ``threshold_dbfs``.
    A frame that runs past the end of ``mono`` (a short trailing stem) reads as silence
    (never active) rather than raising."""
    frames: list[bool] = []
    for i in range(n_frames):
        start = int(round(i * hop_s * sample_rate))
        end = int(round((i + 1) * hop_s * sample_rate))
        frames.append(rms_dbfs(mono[start:end]) > threshold_dbfs)
    return frames


def stem_truth_frames(stem_pcm: Pcm, n_frames: int) -> list[bool]:
    """``activity_frames_from_mono`` over a decoded stem, downmixed to mono the same way
    ``PannsTagger.activations`` downmixes the mix (mean across channels)."""
    samples = stem_pcm.samples
    is_multichannel = samples.ndim > 1 and samples.shape[1] > 1
    mono = samples.mean(axis=1) if is_multichannel else samples.reshape(-1)
    return activity_frames_from_mono(mono, stem_pcm.sample_rate, n_frames)


# --------------------------------------------------------------------------- #
# Predicted-active frames from a Timeline
# --------------------------------------------------------------------------- #


def predicted_frames(
    timeline: Timeline,
    classes: Sequence[str],
    *,
    threshold: float = CFG.TIMELINE_ACTIVITY_THRESHOLD,
) -> list[bool]:
    """Per-frame predicted-active series: True iff the max activation across ``classes``
    (only the ones actually present in ``timeline.classes``) exceeds ``threshold``. All of
    ``classes`` missing from ``timeline.classes`` (the role has no mapping in THIS
    timeline) reads every frame as inactive, matching ``analysis/timeline.py``'s N/A
    convention for an unmapped target."""
    idx = [timeline.classes.index(c) for c in classes if c in timeline.classes]
    if not idx:
        return [False] * len(timeline.frames)
    return [max(frame[j] for j in idx) > threshold for frame in timeline.frames]


def mapped_roles(truth_roles: Iterable[str], timeline_classes: Sequence[str]) -> list[str]:
    """Which of ``truth_roles`` both appear in ``ROLE_CLASS_MAP`` AND have at least one of
    their mapped classes present in ``timeline_classes`` -- the "roles that HAVE a mapping"
    the brief scopes scoring to."""
    tl_classes = set(timeline_classes)
    return [
        r for r in truth_roles if r in ROLE_CLASS_MAP and set(ROLE_CLASS_MAP[r]) & tl_classes
    ]


# --------------------------------------------------------------------------- #
# Fixture-set manifest integrity (§0.4 / Task step 4: STOP on a stale set)
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_integrity(fixtures_dir: Path) -> dict:
    """Load + verify ``fixtures_dir/manifest.json`` (Task 0.4's identity: every file's
    sha256 matches, and the recorded ``fixture_set_hash`` recomputes from them). Raises
    :class:`FixtureSetStale` -- never silently repairs or regenerates -- on any mismatch."""
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FixtureSetStale(f"No manifest.json at {fixtures_dir} -- desktop fixtures not found.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for rel, want in manifest["files"].items():
        path = fixtures_dir / rel
        if not path.is_file():
            mismatches.append(f"missing: {rel}")
            continue
        if _sha256_file(path) != want:
            mismatches.append(f"hash drift: {rel}")
    if mismatches:
        shown = mismatches[:5]
        more = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        raise FixtureSetStale(
            f"Fixture set at {fixtures_dir} does not match its manifest "
            f"({len(mismatches)} mismatch(es)): {shown}{more}. Regenerate via "
            "`scripts/make_fixtures.py` (Task 0.4) -- this script never does that itself."
        )

    recomputed = hashlib.sha256(
        "".join(sorted(manifest["files"].values())).encode("utf-8")
    ).hexdigest()
    if recomputed != manifest.get("fixture_set_hash"):
        raise FixtureSetStale(
            f"manifest.json at {fixtures_dir} has a fixture_set_hash that does not "
            "recompute from its own file list -- the manifest itself is corrupt/stale."
        )
    return manifest


def _fixture_dirs(manifest: dict, fixtures_dir: Path) -> list[Path]:
    top_level = sorted({rel.split("/", 1)[0] for rel in manifest["files"]})
    return [fixtures_dir / name for name in top_level]


# --------------------------------------------------------------------------- #
# The real (desktop) scoring run
# --------------------------------------------------------------------------- #


@dataclass
class ScoreReport:
    """``f1_table``/``counts`` are keyed ``head -> (role | "__overall__") -> value`` —
    ``"__overall__"`` is the micro-pooled F1/counts across every scored role and fixture,
    the number ``pick_winner`` actually decides on."""

    f1_table: dict[str, dict[str, float]] = field(default_factory=dict)
    counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    overall_f1: dict[str, float] = field(default_factory=dict)
    winner: str = ""
    n_fixtures: int = 0
    roles_scored: list[str] = field(default_factory=list)


def score_fixture_set(
    fixtures_dir: Path, *, heads: Sequence[str] = CANDIDATE_HEADS
) -> ScoreReport:
    """Score every candidate head's frame-F1 against the whole desktop fixture set.

    Real inference only -- needs the [analysis] extra (torch/panns_inference) and a GPU
    checkpoint at ``model_registry.checkpoint_dir("tagger/panns-sed")``. Never called from
    the CI-safe test suite.
    """
    from orpheus_mcp.analysis.timeline import PannsTagger, build_timeline

    manifest = verify_manifest_integrity(fixtures_dir)
    fixture_dirs = _fixture_dirs(manifest, fixtures_dir)
    taggers = {head: PannsTagger(head=head) for head in heads}

    counts: dict[str, dict[str, list[int]]] = {h: {} for h in heads}
    roles_seen: set[str] = set()
    n_fixtures = 0

    for fixture_dir in fixture_dirs:
        truth_path = fixture_dir / "truth.json"
        if not truth_path.is_file():
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))

        mix_pcm = decode(str(fixture_dir / "mix.flac"))
        n_frames = math.ceil(mix_pcm.duration_s / _LOCKED_HOP_S) if mix_pcm.duration_s > 0 else 0
        if n_frames == 0:
            continue

        truth_frames_by_role: dict[str, list[bool]] = {}
        for role, rel in truth.get("roles", {}).items():
            if role not in ROLE_CLASS_MAP:
                continue
            stem_pcm = decode(str(fixture_dir / rel))
            truth_frames_by_role[role] = stem_truth_frames(stem_pcm, n_frames)
        if not truth_frames_by_role:
            continue
        n_fixtures += 1

        for head in heads:
            timeline = build_timeline(mix_pcm, taggers[head])
            scored_roles = mapped_roles(truth_frames_by_role, timeline.classes)
            for role in scored_roles:
                pred = predicted_frames(timeline, ROLE_CLASS_MAP[role])
                tp, fp, fn = frame_counts(pred, truth_frames_by_role[role])
                _add_counts(counts[head], role, tp, fp, fn)
                _add_counts(counts[head], "__overall__", tp, fp, fn)
                roles_seen.add(role)

    f1_table = {
        head: {role: f1_from_counts(*c) for role, c in per_role.items()}
        for head, per_role in counts.items()
    }
    overall_f1 = {head: f1_table[head].get("__overall__", 0.0) for head in heads}
    winner = pick_winner(overall_f1, candidates=heads)

    return ScoreReport(
        f1_table=f1_table,
        counts=counts,
        overall_f1=overall_f1,
        winner=winner,
        n_fixtures=n_fixtures,
        roles_scored=sorted(roles_seen),
    )


# --------------------------------------------------------------------------- #
# Reporting: markdown table + registry/pin-report writers
# --------------------------------------------------------------------------- #


def format_table(report: ScoreReport, heads: Sequence[str] = CANDIDATE_HEADS) -> str:
    roles = report.roles_scored
    header = ["head", *[f"{r} F1" for r in roles], "overall F1 (micro)", "tp/fp/fn (overall)"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for head in heads:
        per_role = report.f1_table.get(head, {})
        cells = [head]
        for r in roles:
            cells.append(f"{per_role[r]:.3f}" if r in per_role else "n/a")
        cells.append(f"{report.overall_f1.get(head, 0.0):.3f}")
        tp, fp, fn = report.counts.get(head, {}).get("__overall__", [0, 0, 0])
        cells.append(f"{tp}/{fp}/{fn}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_registry_head(winner: str, *, registry_path: Path | None = None) -> None:
    """Set ``"head": <winner>`` on the ``tagger/panns-sed`` registry entry."""
    from orpheus_mcp.analysis import model_registry

    path = registry_path if registry_path is not None else model_registry.registry_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for item in raw:
        if item.get("id") == _TAGGER_ENTRY_ID:
            item["head"] = winner
            break
    else:
        raise RegistryError(f"{_TAGGER_ENTRY_ID} entry not found in the registry at {path}")
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _strip_section(text: str, header: str) -> str:
    """Drop a previously-written ``header`` markdown section (through the next ``## ``
    heading or EOF) so re-running this script never accumulates duplicate sections."""
    out: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.strip() == header:
            skipping = True
            continue
        if skipping and line.startswith("## ") and line.strip() != header:
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def append_pin_run_d_section(
    report: ScoreReport,
    fixture_set_hash: str,
    *,
    vocab_note: str = "",
    pin_report_path: Path = _PIN_REPORT,
) -> None:
    base = (
        pin_report_path.read_text(encoding="utf-8")
        if pin_report_path.is_file()
        else "# Model Pins — Reference Analysis Engine\n"
    )
    base = _strip_section(base, _PIN_RUN_D_HEADER)

    tie = len({report.overall_f1[h] for h in CANDIDATE_HEADS if h in report.overall_f1}) == 1
    winner_note = (
        f"**Winner: `{report.winner}`**"
        + (" (three-way tie on overall F1 -- §15.6 breaks to `DecisionLevelAtt`)" if tie else "")
    )

    lines = [
        _PIN_RUN_D_HEADER,
        "",
        "Generated by `scripts/pick_sed_head.py` on the reference RTX 3080, scored against the "
        f"desktop fixture set (`~/.orpheus_fixtures`, {report.n_fixtures} fixtures scored, "
        f"`fixture_set_hash` `{fixture_set_hash}`). Frame-F1 is micro-averaged (TP/FP/FN pooled "
        "across every 1 s frame of every scored fixture) per role and overall; truth-activity "
        "is `stem RMS > -50 dBFS` per 1 s frame (matching the Timeline hop). Only roles with a "
        f"registry `tagger_classes` mapping are scored: {', '.join(report.roles_scored)}.",
        "",
        "Known scope limit: the installed `panns-inference` package loads only the "
        "`Cnn14_DecisionLevelMax` architecture regardless of the `head` string passed to "
        "`PannsTagger` (the Avg/Att checkpoints are not part of that PyPI distribution) -- all "
        "three candidate heads therefore run the identical weights/forward pass in this run, "
        "producing a genuine three-way tie rather than a fabricated split decision.",
        "",
        format_table(report),
        "",
        winner_note,
    ]
    if vocab_note:
        lines += ["", vocab_note]
    lines.append("")

    pin_report_path.parent.mkdir(parents=True, exist_ok=True)
    new_text = base.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n"
    pin_report_path.write_text(new_text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path.home() / ".orpheus_fixtures",
        help="Desktop fixture set root (default: ~/.orpheus_fixtures).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and print the table only; do not write the registry or pin report.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = verify_manifest_integrity(args.fixtures_dir)
    except FixtureSetStale as exc:
        print(f"BLOCKED: {exc}")
        return 2

    report = score_fixture_set(args.fixtures_dir)
    print(format_table(report))
    print(f"\nWinner: {report.winner}")

    if not args.dry_run:
        write_registry_head(report.winner)
        append_pin_run_d_section(report, manifest["fixture_set_hash"])
        print(
            f"Wrote head={report.winner!r} to the registry and appended pin run (d) to "
            f"{_PIN_REPORT}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
