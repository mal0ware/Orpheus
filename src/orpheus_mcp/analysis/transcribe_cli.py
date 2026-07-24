"""T5 lane: isolated basic-pitch subprocess transcription (§9 D9, Task 0.5).

basic-pitch is the **last resort** tier (§3.1/§9 T5) — instrument-agnostic polyphonic note
transcription, opt-in only. It pins Python <=3.11 and is NEVER imported in the main env;
instead it lives in its own uv venv (``~/.orpheus_transcribe_env``, created by
``scripts/setup_transcribe_env.py``) and is driven purely through its CLI as a subprocess —
the same isolation pattern as audio-separator (Task 0.3).

The CLI writes a ``<audio-stem>_basic_pitch.csv`` note-events file into the output directory
(``basic_pitch.inference.save_note_events``, verified against upstream source) with header
``start_time_s,end_time_s,pitch_midi,velocity,pitch_bend`` — ``pitch_bend`` is a variable
number of trailing columns, present only when the note carries pitch-bend data.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "NoteEvent",
    "TranscribeEnvMissing",
    "transcribe",
    "transcribe_env_dir",
    "transcribe_env_cli",
]


def transcribe_env_dir() -> Path:
    """Location of the isolated basic-pitch env: ``~/.orpheus_transcribe_env``."""
    return Path.home() / ".orpheus_transcribe_env"


def transcribe_env_cli() -> Path:
    """Path to the ``basic-pitch`` console script inside the isolated env.

    Windows lays console scripts under ``Scripts/`` with a ``.exe`` suffix; POSIX uses
    ``bin/``. Returns the expected path whether or not it exists yet (callers check).
    """
    root = transcribe_env_dir()
    if sys.platform == "win32":
        return root / "Scripts" / "basic-pitch.exe"
    return root / "bin" / "basic-pitch"


class TranscribeEnvMissing(Exception):
    """The isolated basic-pitch env hasn't been created yet."""


@dataclass(frozen=True)
class NoteEvent:
    """One row of basic-pitch's note-events CSV output.

    Field names match the CSV header verbatim (``start_time_s,end_time_s,pitch_midi,
    velocity,pitch_bend``); ``pitch_bend`` is ``None`` when the row carried no trailing
    pitch-bend columns.
    """

    start_time_s: float
    end_time_s: float
    pitch_midi: int
    velocity: int
    pitch_bend: list[int] | None = None


def _parse_note_events_csv(text: str) -> list[NoteEvent]:
    """Parse basic-pitch's note-events CSV text, skipping malformed rows defensively.

    Raises ``ValueError`` (naming the count) only if every data row was malformed — a
    partially-malformed file still yields whatever parsed cleanly.
    """
    reader = csv.reader(text.splitlines())
    next(reader, None)  # header: start_time_s,end_time_s,pitch_midi,velocity,pitch_bend

    events: list[NoteEvent] = []
    total = 0
    malformed = 0
    for row in reader:
        if not row:
            continue
        total += 1
        try:
            if len(row) < 4:
                raise ValueError("expected at least 4 columns")
            start_time_s = float(row[0])
            end_time_s = float(row[1])
            pitch_midi = int(row[2])
            velocity = int(row[3])
            pitch_bend = [int(v) for v in row[4:]] if len(row) > 4 else None
        except ValueError:
            malformed += 1
            continue
        events.append(NoteEvent(start_time_s, end_time_s, pitch_midi, velocity, pitch_bend))

    if total and malformed == total:
        raise ValueError(
            f"basic-pitch note-events CSV: all {malformed} data row(s) were malformed; "
            "could not parse any note events"
        )
    return events


def transcribe(wav_path: Path, out_dir: Path) -> list[NoteEvent]:
    """Run basic-pitch (subprocess, isolated env) on ``wav_path``, returning parsed note events.

    ``out_dir`` is a caller-provided temp/output directory the CLI writes into. Raises
    :class:`TranscribeEnvMissing` naming the setup command if the isolated env doesn't exist
    yet.
    """
    cli = transcribe_env_cli()
    if not cli.is_file():
        raise TranscribeEnvMissing(
            "The isolated basic-pitch env has not been set up. Run "
            "`uv run python scripts/setup_transcribe_env.py` first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # basic-pitch's CLI prints a sparkle-emoji banner to stdout; on Windows the default
    # console codepage (cp1252) can't encode it and the subprocess crashes with a
    # UnicodeEncodeError (verified live on this box). Force UTF-8 stdio for the child only.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run(
        [str(cli), str(out_dir), str(wav_path), "--save-note-events"],
        check=True,
        env=env,
    )

    csv_path = out_dir / f"{wav_path.stem}_basic_pitch.csv"
    return _parse_note_events_csv(csv_path.read_text(encoding="utf-8"))
