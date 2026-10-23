"""Create the isolated ``basic-pitch`` uv env pinned to Python 3.11 (Task 0.5, §9 D9).

basic-pitch (Spotify) pins its own dependency stack to Python <=3.11, while the main env's
floor is 3.11 but may run on 3.12+. It cannot share the main env and lives in its own uv venv,
driven only through its CLI as a subprocess (never ``import basic_pitch`` in the main env) —
the same isolation pattern the plan uses for audio-separator (Task 0.3,
``scripts/setup_separator_env.py``).

    uv venv ~/.orpheus_transcribe_env --python 3.11
    uv pip install --python ~/.orpheus_transcribe_env basic-pitch

Usage: ``uv run python scripts/setup_transcribe_env.py``

The env is created ONCE at install time. ``analysis/transcribe_cli.py``'s ``transcribe()``
(T5 lane, last resort) invokes ``basic-pitch`` out of this env via ``transcribe_env_cli()``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from orpheus_mcp.analysis.transcribe_cli import transcribe_env_cli, transcribe_env_dir


def setup() -> Path:
    """Create the env pinned to Python 3.11 and install basic-pitch into it.

    Returns the CLI path.
    """
    root = transcribe_env_dir()
    subprocess.run(["uv", "venv", "--python", "3.11", str(root)], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(root), "basic-pitch"],
        check=True,
    )
    # basic-pitch's note_creation.py -> resampy still imports the legacy `pkg_resources` API
    # at module load time. Newer setuptools (>=81) dropped it outright (verified live on this
    # box: setuptools 83.0.0 has no pkg_resources, breaking every CLI invocation with
    # `ModuleNotFoundError: No module named 'pkg_resources'`); pin below that floor so the
    # module still ships. Harmless deprecation warning only, from setuptools<81.
    subprocess.run(
        ["uv", "pip", "install", "--python", str(root), "setuptools<81"],
        check=True,
    )
    cli = transcribe_env_cli()
    print(f"basic-pitch env ready: {cli}")
    return cli


def main() -> None:
    setup()


if __name__ == "__main__":
    main()
