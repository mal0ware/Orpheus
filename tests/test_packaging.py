"""Packaging invariants — extras exist, markers registered, no heavy import at core import time."""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def _pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text("utf-8"))

def test_extras_declared():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert {"analysis", "separation", "transcribe", "dev"} <= set(extras)
    assert any(d.startswith("librosa") for d in extras["analysis"])
    assert any(d.startswith("torch") for d in extras["analysis"])
    # audio-separator lives in an ISOLATED env (D9) - its deps cap fastmcp<3.4
    assert not any(d.startswith("audio-separator") for d in extras["separation"])
    assert any(d.startswith("faster-whisper") for d in extras["separation"])
    # ctranslate2 (a faster-whisper dep) still does `import pkg_resources` at import time;
    # setuptools 81 dropped it (83 removed it outright) - pinned until ctranslate2 drops it.
    assert any(d.startswith("setuptools<81") for d in extras["separation"])

def test_markers_registered():
    ini = _pyproject()["tool"]["pytest"]["ini_options"]
    assert any(m.startswith("desktop") for m in ini["markers"])
    assert any(m.startswith("gpu") for m in ini["markers"])

def test_core_import_stays_lean():
    # subprocess, not importlib.reload — reload of a package with submodules is unreliable
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import orpheus_mcp, sys;"
                "bad=[m for m in ('torch','librosa','audio_separator',"
                "'faster_whisper') if m in sys.modules];"
                "sys.exit(1 if bad else 0)"
            ),
        ],
        capture_output=True,
    )
    assert out.returncode == 0  # analysis extras load lazily, inside functions


def test_core_dependencies_have_no_gpl_matchering():
    # MIT package: GPL-3.0 Matchering may only ever be an optional extra (spec 2026-10-23 §3).
    deps = _pyproject()["project"]["dependencies"]
    assert not any(d.lower().startswith("matchering") for d in deps)
