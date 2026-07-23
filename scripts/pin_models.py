"""``scripts/pin_models.py`` — model registry checkpoint pin run (Task 0.3, pin run a).

Two jobs, both landing in ``data/model_registry.json`` + ``docs/pins/2026-07-model-pins.md``:

1. **Hash pinning** (``pin_all_hashes``): download every registry checkpoint and record its
   sha256 into the registry JSON. Resilient and per-entry — a 404 / dead mirror on one slot is
   recorded as a failure and the run continues (Deviations D6), it never aborts the others. A
   bag-of-models entry (list ``checkpoint_url``) records a positional list of hashes.

2. **Separation pin run** (``separation_pin_run``): for every separation / dereverb registry
   entry, run it on a test mix and record load time, separation time, and rt_factor (wall /
   audio-duration) on the reference 3080 into the JSON + the markdown pin report. ``msst``
   entries run through the vendored SCNet path (``_vendor/msst/inference.py``); every
   ``audio-separator`` entry runs through the isolated separator env's CLI
   (``scripts/setup_separator_env.py`` — D9), never imported into this env.

No owned mp3 is required: ``--synthetic`` generates a 60 s stereo 44.1 kHz test mix
(drums-like bursts + chord pad + bass sine + a synthetic vocal formant sweep) and runs the
timing/smoke on it. Every rt_factor row is then flagged ``synthetic-audio`` and the ear-check
notes column is ``PENDING (human)`` — an owned-mp3 re-run + ear-check is a tracked deferred
step, not this run's.

Usage:
    uv run python scripts/pin_models.py --hashes-only          # just pin sha256s
    uv run --extra separation python scripts/pin_models.py --synthetic
    uv run --extra separation python scripts/pin_models.py --audio <owned.mp3>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from orpheus_mcp.analysis.fetch_models import (
    _filename_for_url,
    _sha256_file,
    download_checkpoint,
    require_ffmpeg,
)
from orpheus_mcp.analysis.model_registry import (
    ModelEntry,
    checkpoint_dir,
    load_registry,
    registry_path,
)

# Tasks whose entries are run in the separation pin run (everything that produces stems).
SEPARATION_TASKS = {"multi-stem", "single-target", "drumkit-split", "dereverb"}

# fetch_models() statuses that are not real hash digests.
_NON_HASH_STATUSES = {"skipped (no checkpoint_url)"}

_PIN_REPORT = Path("docs/pins/2026-07-model-pins.md")
_AUDIO_OUT = Path("docs/pins/audio_out")

# --- pin report table contract (the CI parser test depends on these exact headers) ---
_TABLE_COLUMNS = [
    "id",
    "engine",
    "url_resolved",
    "sha256",
    "load_s",
    "sep_s",
    "rt_factor",
    "audio",
    "status",
    "ear_check_notes",
]


# ---------------------------------------------------------------------------
# 1. Hash pinning (resilient, per-entry)
# ---------------------------------------------------------------------------
def _urls_of(entry: ModelEntry) -> list[str]:
    """The entry's checkpoint URL(s) as a flat list (empty if it has none)."""
    if entry.checkpoint_url is None:
        return []
    if isinstance(entry.checkpoint_url, list):
        return entry.checkpoint_url
    return [entry.checkpoint_url]


def pin_entry_hash(entry: ModelEntry) -> str | list[str]:
    """Download an entry's checkpoint(s) and return its sha256 (str, or list for a bag).

    Raises on any download/hash failure so the caller can record the slot as UNRESOLVED.
    """
    urls = _urls_of(entry)
    digests: list[str] = []
    for url in urls:
        dest = checkpoint_dir(entry.id) / _filename_for_url(url, entry.id)
        download_checkpoint(url, dest)
        digests.append(_sha256_file(dest))
    return digests if isinstance(entry.checkpoint_url, list) else digests[0]


def pin_all_hashes() -> tuple[dict[str, str | list[str]], dict[str, str]]:
    """Pin the sha256 of every downloadable registry checkpoint, writing back into the JSON.

    Returns ``(pinned, failed)`` — ``pinned`` maps id -> hash(es) actually written, ``failed``
    maps id -> error string for slots whose download failed (their sha256 stays None; D6).
    """
    entries = load_registry(allow_unpinned=True)
    raw = json.loads(registry_path().read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in raw}

    pinned: dict[str, str | list[str]] = {}
    failed: dict[str, str] = {}

    for entry in entries:
        if not _urls_of(entry):
            continue
        try:
            digest = pin_entry_hash(entry)
        except Exception as exc:  # noqa: BLE001 — a dead mirror must not abort the whole run
            failed[entry.id] = f"{type(exc).__name__}: {exc}"
            print(f"UNRESOLVED {entry.id}: {failed[entry.id]}")
            continue
        by_id[entry.id]["sha256"] = digest
        pinned[entry.id] = digest
        # Persist incrementally so a long, resumable background run never loses progress.
        registry_path().write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"pinned {entry.id}")

    return pinned, failed


# ---------------------------------------------------------------------------
# 2. Synthetic test mix
# ---------------------------------------------------------------------------
def generate_synthetic_mix(path: Path, duration_s: float = 60.0, sr: int = 44100) -> None:
    """Write a simple 60 s stereo 44.1 kHz test mix for the separation smoke run.

    Four crude "stems" summed: drum-like filtered noise bursts on the beat, a chord pad
    (C-major triad partials), a bass sine on the root, and a synthetic 'vocal' — an AM-voiced
    carrier with a slow formant sweep. Deliberately not musical; it just gives every separator
    real broadband/tonal/transient content to chew on so timing + non-crash can be measured.
    """
    rng = np.random.default_rng(0)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    bpm = 120.0
    beat = 60.0 / bpm

    # Drums: short exponentially-decaying noise bursts every beat.
    drums = np.zeros(n)
    for k in range(int(duration_s / beat)):
        start = int(k * beat * sr)
        length = int(0.12 * sr)
        env = np.exp(-np.linspace(0, 8, length))
        burst = rng.standard_normal(length) * env
        drums[start : start + length] += burst[: max(0, min(length, n - start))][: n - start]

    # Bass: root sine walking C2->G2 per bar.
    roots = [65.41, 65.41, 87.31, 98.00]  # C2 C2 F2 G2
    bass = np.zeros(n)
    bar = beat * 4
    for k in range(int(duration_s / bar) + 1):
        f = roots[k % len(roots)]
        start, end = int(k * bar * sr), min(n, int((k + 1) * bar * sr))
        bass[start:end] = 0.6 * np.sin(2 * np.pi * f * t[start:end])

    # Chord pad: C-major triad partials, gently tremoloed.
    pad = np.zeros(n)
    for f in (261.63, 329.63, 392.00):
        pad += np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    pad *= 0.15 * (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t))

    # Vocal: AM-voiced carrier with a slow formant sweep (200 -> 400 Hz center).
    formant = 200 + 200 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    voice = np.sin(2 * np.pi * np.cumsum(formant) / sr)
    voice *= 0.4 * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 5 * t)))  # 5 Hz voicing

    mono = drums + bass + pad + voice
    mono /= np.max(np.abs(mono)) + 1e-9
    mono *= 0.9
    stereo = np.stack([mono, np.roll(mono, 32)], axis=1).astype(np.float32)  # slight width
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sr)


# ---------------------------------------------------------------------------
# 3. Separation pin run
# ---------------------------------------------------------------------------
@dataclass
class PinRow:
    id: str
    engine: str
    url_resolved: str = "no"
    sha256: str = "-"
    load_s: str = "-"
    sep_s: str = "-"
    rt_factor: str = "-"
    audio: str = "-"
    status: str = "UNRESOLVED"
    ear_check_notes: str = "PENDING (human)"


def _run_msst(entry: ModelEntry, wav_in: Path, out_dir: Path) -> tuple[float, float]:
    """Run the vendored SCNet path; returns (load_s, sep_s). Writes FLAC stems to out_dir."""
    import torch

    from orpheus_mcp._vendor.msst.inference import load_scnet, separate_array

    urls = _urls_of(entry)
    ckpt = checkpoint_dir(entry.id) / _filename_for_url(urls[0], entry.id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    model, config = load_scnet(ckpt, None, device)
    load_s = time.perf_counter() - t0

    data, sr = sf.read(str(wav_in), dtype="float32", always_2d=True)
    mix = data.T
    if mix.shape[0] == 1:
        mix = np.repeat(mix, int(config.model.audio_channels), axis=0)

    t1 = time.perf_counter()
    stems = separate_array(model, config, mix, device)
    sep_s = time.perf_counter() - t1

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, wav in stems.items():
        sf.write(str(out_dir / f"{name}.flac"), wav.T, sr, subtype="PCM_16")
    return load_s, sep_s


def _run_audio_separator(entry: ModelEntry, wav_in: Path, out_dir: Path) -> float:
    """Run an audio-separator entry through the isolated env CLI; returns wall seconds."""
    try:  # importable both as `scripts.pin_models` (tests) and as a bare script (`__main__`)
        from scripts.setup_separator_env import separator_env_cli
    except ModuleNotFoundError:
        from setup_separator_env import separator_env_cli

    cli = separator_env_cli()
    if not cli.exists():
        raise FileNotFoundError(
            f"audio-separator env CLI not found at {cli}. Run "
            "`python scripts/setup_separator_env.py` first (D9)."
        )
    urls = _urls_of(entry)
    model_filename = _filename_for_url(urls[0], entry.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli),
        str(wav_in),
        "--model_filename",
        model_filename,
        "--model_file_dir",
        str(checkpoint_dir(entry.id)),
        "--output_dir",
        str(out_dir),
        "--output_format",
        "flac",
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return time.perf_counter() - t0


def separation_pin_run(
    audio_path: Path,
    *,
    synthetic: bool,
    pinned: dict[str, str | list[str]],
    run_separators: bool = False,
) -> list[PinRow]:
    """Run every separation/dereverb entry on ``audio_path`` and collect a PinRow per slot.

    ``msst`` entries always run (the vendored SCNet path, this task's engine — GPU, fast).
    ``audio-separator`` entries only run when ``run_separators=True``: their subprocess adapter
    is Phase 4 (D9) work and the isolated env here is CPU-only, so by default they are recorded
    HASH-PINNED with a deferral note rather than timed on CPU (amendment 4: no CPU timings).
    """
    info = sf.info(str(audio_path))
    duration_s = info.frames / info.samplerate
    audio_label = "synthetic-audio" if synthetic else audio_path.name

    entries = [e for e in load_registry(allow_unpinned=True) if e.task in SEPARATION_TASKS]
    raw = json.loads(registry_path().read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in raw}

    rows: list[PinRow] = []
    for entry in entries:
        digest = pinned.get(entry.id)
        sha_short = "-"
        if isinstance(digest, str):
            sha_short = digest[:12]
        elif isinstance(digest, list):
            sha_short = f"[{len(digest)} files]"
        row = PinRow(
            id=entry.id,
            engine=entry.engine,
            url_resolved="yes" if entry.id in pinned else "no",
            sha256=sha_short,
            audio=audio_label,
        )
        out_dir = _AUDIO_OUT / entry.id.replace("/", "_")
        hash_pinned = entry.id in pinned

        if entry.engine != "msst" and not run_separators:
            row.status = "HASH-PINNED" if hash_pinned else "UNRESOLVED"
            row.ear_check_notes = "sep deferred: audio-separator adapter is Phase 4 (D9)"
            rows.append(row)
            continue

        try:
            if entry.engine == "msst":
                load_s, sep_s = _run_msst(entry, audio_path, out_dir)
                row.load_s = f"{load_s:.2f}"
            else:
                sep_s = _run_audio_separator(entry, audio_path, out_dir)
            row.sep_s = f"{sep_s:.2f}"
            rt = sep_s / duration_s
            row.rt_factor = f"{rt:.3f}"
            row.status = "PINNED"  # hash pinned AND separation ran + timed
            by_id[entry.id]["rt_factor"] = round(rt, 3)
        except Exception as exc:  # noqa: BLE001 — record and move to the next slot (D6)
            # Distinguish a dead download (nothing pinned) from a hash-pinned checkpoint whose
            # separation could not be timed automatically here (e.g. the audio-separator CLI
            # rejects out-of-registry checkpoint filenames — that adapter is Task 4.1's job).
            row.status = "HASH-PINNED" if hash_pinned else "UNRESOLVED"
            row.ear_check_notes = f"sep not run here: {type(exc).__name__}: {str(exc)[:110]}"
            print(f"{row.status} (engine) {entry.id}: {exc}")
        rows.append(row)

    registry_path().write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return rows


# ---------------------------------------------------------------------------
# 4. Pin report read/write (the CI-safe contract)
# ---------------------------------------------------------------------------
def _fmt_row(row: PinRow) -> str:
    cells = [getattr(row, col) for col in _TABLE_COLUMNS]
    return "| " + " | ".join(str(c) for c in cells) + " |"


def write_pin_report(rows: list[PinRow], failed_hashes: dict[str, str]) -> None:
    """Write ``docs/pins/2026-07-model-pins.md`` — the pin run (a) record."""
    header = "| " + " | ".join(_TABLE_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _TABLE_COLUMNS) + " |"
    lines = [
        "# Model Pins — Reference Analysis Engine",
        "",
        "## Pin run (a): separation + dereverb engines",
        "",
        "Generated by `scripts/pin_models.py` on the reference RTX 3080. All timing rows below",
        "were produced on **synthetic audio** (a generated 60 s stereo mix — no owned mp3 was",
        "available at pin time); rt_factor is therefore indicative, not a quality signal. The",
        "`ear_check_notes` column is `PENDING (human)`: an owned-mp3 re-run + per-stem ear-check",
        "is a tracked, deferred follow-up (§12 step 3), NOT part of this automated run.",
        "",
        header,
        sep,
    ]
    lines.extend(_fmt_row(r) for r in rows)
    lines.append("")
    if failed_hashes:
        lines.append("## Unresolved checkpoint downloads (hash not pinned — D6)")
        lines.append("")
        for model_id, err in sorted(failed_hashes.items()):
            lines.append(f"- `{model_id}`: {err}")
        lines.append("")
    _PIN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    _PIN_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_pin_report(text: str) -> list[dict[str, str]]:
    """Parse the pin report's engine table into a list of row dicts (CI-safe, stdlib-only).

    Returns one dict per table row keyed by ``_TABLE_COLUMNS``. Robust to the surrounding
    prose — it locks onto the single table whose header row is exactly ``_TABLE_COLUMNS``.
    """
    rows: list[dict[str, str]] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells == _TABLE_COLUMNS:
            in_table = True
            continue
        if in_table:
            if all(set(c) <= {"-"} for c in cells):  # the |---|---| separator line
                continue
            if len(cells) == len(_TABLE_COLUMNS):
                rows.append(dict(zip(_TABLE_COLUMNS, cells, strict=True)))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class _Args:
    audio: Path | None = None
    synthetic: bool = False
    hashes_only: bool = False
    skip_hashes: bool = False
    extras: list[str] = field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=None, help="Owned audio file to pin on.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate + use a 60 s synthetic test mix instead of an owned file.",
    )
    parser.add_argument(
        "--hashes-only",
        action="store_true",
        help="Only pin sha256 hashes; skip the separation timing run.",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip hash pinning (assume already pinned); only run the separation timing.",
    )
    parser.add_argument(
        "--run-separators",
        action="store_true",
        help="Also invoke the audio-separator env CLI per slot (Phase-4 adapter; needs a GPU "
        "separator env). Off by default — only the vendored MSST/SCNet engine runs.",
    )
    args = parser.parse_args()

    require_ffmpeg()

    pinned: dict[str, str | list[str]] = {}
    failed: dict[str, str] = {}
    if not args.skip_hashes:
        pinned, failed = pin_all_hashes()
        print(f"Pinned {len(pinned)} checkpoint hashes; {len(failed)} unresolved.")
    else:
        # Reconstruct the already-pinned map from the registry for the report.
        for entry in load_registry(allow_unpinned=True):
            if entry.sha256 is not None:
                pinned[entry.id] = entry.sha256

    if args.hashes_only:
        return

    if args.synthetic:
        audio_path = _AUDIO_OUT / "_synthetic_mix.flac"
        print(f"Generating synthetic test mix -> {audio_path}")
        generate_synthetic_mix(audio_path)
    elif args.audio is not None:
        audio_path = args.audio
    else:
        raise SystemExit("Provide --audio <file>, --synthetic, or --hashes-only.")

    rows = separation_pin_run(
        audio_path,
        synthetic=args.synthetic,
        pinned=pinned,
        run_separators=args.run_separators,
    )
    write_pin_report(rows, failed)
    print(f"Wrote pin report: {_PIN_REPORT} ({len(rows)} separation slots).")
    for r in rows:
        print(f"  {r.id}: {r.status} (sep_s={r.sep_s}, rt={r.rt_factor})")


if __name__ == "__main__":
    main()
