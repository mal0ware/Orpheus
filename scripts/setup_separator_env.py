"""Create the isolated ``audio-separator`` uv env (Deviations D9).

Every ``audio-separator`` release transitively caps ``fastmcp<3.4`` (Task 0.1 finding), so it
cannot share the main env with the MCP server. It lives in its own uv venv and is driven only
through its CLI as a subprocess (never ``import audio_separator`` in the main env) — the same
isolation pattern the plan uses for basic-pitch (Task 0.5).

    uv venv ~/.orpheus_separator_env          # any modern Python
    uv pip install "audio-separator[gpu]"     # into that env

Usage:  ``python scripts/setup_separator_env.py``  (add ``--cpu`` for the CPU wheel).

The env is created ONCE at install time. ``analysis/separation.py``'s ``AudioSeparatorEngine``
(Phase 4) and ``scripts/pin_models.py``'s separation pin run invoke ``audio-separator`` out of
this env via ``separator_env_cli()``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def separator_env_dir() -> Path:
    """Location of the isolated audio-separator env: ``~/.orpheus_separator_env``."""
    return Path.home() / ".orpheus_separator_env"


def separator_env_cli() -> Path:
    """Path to the ``audio-separator`` console script inside the isolated env.

    Windows lays console scripts under ``Scripts/`` with a ``.exe`` suffix; POSIX uses
    ``bin/``. Returns the expected path whether or not it exists yet (callers check).
    """
    root = separator_env_dir()
    if sys.platform == "win32":
        return root / "Scripts" / "audio-separator.exe"
    return root / "bin" / "audio-separator"


def setup(*, gpu: bool = True) -> Path:
    """Create the env and install audio-separator into it. Returns the CLI path."""
    root = separator_env_dir()
    subprocess.run(["uv", "venv", str(root)], check=True)
    extra = "gpu" if gpu else "cpu"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(root), f"audio-separator[{extra}]"],
        check=True,
    )
    cli = separator_env_cli()
    print(f"audio-separator env ready: {cli}")
    return cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install the CPU wheel instead of the CUDA (gpu) build.",
    )
    args = parser.parse_args()
    setup(gpu=not args.cpu)


if __name__ == "__main__":
    main()
