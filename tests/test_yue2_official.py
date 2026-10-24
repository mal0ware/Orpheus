"""Opt-in: Orpheus's YuE2 checker and the official ``abc_tools.py`` must give the same verdict
(spec 2026-10-24 §2 "Validation"). Run with ``ORPHEUS_OFFICIAL_ABC_CHECK=1``.

The official checker (Apache-2.0, stdlib-only) is fetched once from the YuE repository at a pinned
commit into ``<cache>/third_party/`` and its sha256 is checked; it is never vendored into the repo
and runs isolated (``python -I``) in a temp folder. Known, deliberate difference: Orpheus rejects
CR line endings (yue2.cpp would pass them to the model); the official script tolerates them.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pytest

from orpheus_mcp.analysis.cachefs import _default_cache_root
from orpheus_mcp.notation.leadsheet import resolve_lead_sheet
from orpheus_mcp.notation.yue2_abc import export_yue2
from orpheus_mcp.notation.yue2_check import check_yue2_abc

pytestmark = pytest.mark.skipif(
    os.environ.get("ORPHEUS_OFFICIAL_ABC_CHECK") != "1",
    reason="opt-in: set ORPHEUS_OFFICIAL_ABC_CHECK=1 (fetches YuE's abc_tools.py once)",
)

COMMIT = "e76a8753dcd23358726e7b59f4a014498e8895c6"  # YuE main, 2026-09-09
URL = ("https://raw.githubusercontent.com/multimodal-art-projection/YuE/"
       f"{COMMIT}/skills/yue2-music/scripts/abc_tools.py")
SHA256 = "ea04b922dacebec7ad257a2f8d83bdb5dfecb7a23110c1a3121c5c41c313930e"


@pytest.fixture(scope="module")
def official_tool() -> Path:
    path = _default_cache_root() / "third_party" / f"abc_tools-{COMMIT[:12]}.py"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as resp:  # noqa: S310 (pinned https)
            data = resp.read()
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == SHA256, f"abc_tools.py at {path} does not match the pinned hash"
    return path


def _official_ok(tool: Path, text: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        score = Path(d) / "score.abc"
        score.write_text(text, encoding="utf-8", newline="\n")
        proc = subprocess.run([sys.executable, "-I", str(tool), "inspect", str(score)],
                              capture_output=True, encoding="utf-8", errors="replace", cwd=d)
    return proc.returncode == 0


def _cases() -> list:
    sys.path.insert(0, str(Path(__file__).parent))
    import test_yue2_check as checks
    import test_yue2_import as imports

    out = [pytest.param(t, id=f"good-{i}")
           for i, t in enumerate([checks.SCORE_ABC, checks.MELODY_ABC, checks.EXAMPLE_INS_ABC,
                                  checks.MODEL_STYLE_ABC])]
    out += [pytest.param(export_yue2(resolve_lead_sheet(s)[0]).abc, id=f"export-{i}")
            for i, s in enumerate(imports.SHEETS)]
    out += [pytest.param(p.values[0], id=f"bad-{p.id}") for p in checks.BAD_CASES
            if p.id != "crlf"]
    return out


@pytest.mark.parametrize("text", _cases())
def test_same_verdict_as_the_official_checker(official_tool, text):
    assert (not check_yue2_abc(text)) == _official_ok(official_tool, text)
