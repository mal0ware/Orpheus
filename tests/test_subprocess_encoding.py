"""Subprocess output must never be decoded with the Windows locale (cp1252).

ffmpeg prints a file's tags and path to stderr as UTF-8. On Windows ``subprocess.run`` reads
pipes on helper threads; with ``text=True`` and no encoding those threads decode as cp1252,
which has no character for bytes like 0x9D (the tail of UTF-8 "”"). The reader thread then dies
with UnicodeDecodeError (seen on the first real-music run, 2026-07-25) and the captured stream
comes back as None.
"""

from __future__ import annotations

import shutil
import subprocess
import threading

import pytest

from orpheus_mcp.analysis import cachefs

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

UTF8_ONLY = "Help ”Her” self — café"  # "” — é": 0x9D is not cp1252


@pytest.fixture
def thread_errors(monkeypatch):
    seen: list[BaseException] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: seen.append(args.exc_value))
    return seen


def _tagged_mp3(path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-metadata", f"title={UTF8_ONLY}", "-metadata", f"artist={UTF8_ONLY}", str(path)],
        check=True,
    )


def test_decoding_a_file_with_utf8_tags_has_no_reader_thread_errors(tmp_path, thread_errors):
    mp3 = tmp_path / "tagged.mp3"
    _tagged_mp3(mp3)
    data, sr, _ = cachefs._decode_via_ffmpeg(mp3)
    assert sr > 0 and len(data) > 0
    assert thread_errors == []


def test_ffmpeg_failure_on_a_utf8_path_reports_cleanly(tmp_path, thread_errors):
    bad = tmp_path / f"{UTF8_ONLY}.mp3"
    bad.write_bytes(b"not audio at all" * 64)
    with pytest.raises(cachefs.DecodeError, match="ffmpeg failed to decode"):
        cachefs._decode_via_ffmpeg(bad)
    assert thread_errors == []
