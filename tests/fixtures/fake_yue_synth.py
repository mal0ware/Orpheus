"""A stand-in for yue2.cpp's ``yue-synth`` CLI (spec 2026-10-24 sketch §4), for tests only.

Takes the same arguments, prints the log lines the real binary prints (copied from the
2026-10-23 spike logs), writes a short 48 kHz 24-bit stereo WAV at ``--out`` and, like the
real one, a replay JSON at ``<out-stem>.json``. ``FAKE_YUE_MODE`` picks the run:
``ok`` (default), ``truncate`` (the semantic stage hits its budget), ``fatal`` (exit 1, no
audio) or ``fatal_after_wav`` (exit 1 after the WAV is written).
It also records the arguments and the request it read in ``<out-dir>/fake_cli_call.json``.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import wave
from pathlib import Path


def main(argv: list[str]) -> int:
    args = dict(zip(argv[::2], argv[1::2], strict=False))
    req_path, out = Path(args["--request"]), Path(args["--out"])
    request = json.loads(req_path.read_text(encoding="utf-8"))
    mode = os.environ.get("FAKE_YUE_MODE", "ok")
    log = sys.stderr.write
    log(f"[Request] Parsed {req_path.name}\n")
    log("[Store] Created (policy=STRICT)\n")
    (out.parent / "fake_cli_call.json").write_text(
        json.dumps({"argv": argv, "request": request, "path": os.environ.get("PATH", "")}),
        encoding="utf-8")
    if mode == "fatal":
        log("[Synth] FATAL: unknown output format wav99\n")
        return 1
    frames = int(25 * request["duration"])
    if mode == "truncate":
        log(f"[AR] Frame budget clamped to {frames} by the requested duration "
            f"({request['duration']:.1f} s)\n")
        log(f"[AR] Semantic song 0: {frames} tokens (truncated)\n")
    else:
        frames = max(1, frames - 100)
        log(f"[AR] Semantic song 0: end token at step {frames}\n")
        log(f"[AR] Semantic song 0: {frames} tokens\n")
    seconds = 0.5
    n = int(48000 * seconds)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(48000)
        sample = struct.pack("<i", 1 << 16)[:3]
        w.writeframes(sample * 2 * n)
    log(f"[WAV] Wrote {out.name}: {n} samples, 48000 Hz, stereo\n")
    if mode == "fatal_after_wav":  # the real binary exits 1 when the replay JSON can't be written
        log(f"[Synth] FATAL: cannot write {out.with_suffix('.json')}\n")
        return 1
    replay = dict(request, semantic_tokens="1,2,3")
    out.with_suffix(".json").write_text(json.dumps(replay), encoding="utf-8")
    log(f"[Synth] Done: {seconds:.1f} s of audio, seeds {request['lm_seed']} and "
        f"{request['seed']} -> {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
