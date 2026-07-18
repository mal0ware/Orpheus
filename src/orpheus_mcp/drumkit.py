# src/orpheus_mcp/drumkit.py
"""Provide the 3 stock-kit one-shots. If bundled CC0 samples exist in data/drumkit/ they
are used; otherwise tiny synthesized WAVs are generated (license-clean, no download)."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

_SR = 44100
_VOICES = ("kick", "snare", "hat")


def _synth(voice: str) -> list[int]:
    """Return int16 PCM samples for a short one-shot."""
    import random  # local: only for noise voices

    rng = random.Random({"kick": 1, "snare": 2, "hat": 3}[voice])
    dur = {"kick": 0.18, "snare": 0.16, "hat": 0.05}[voice]
    n = int(_SR * dur)
    out: list[int] = []
    for i in range(n):
        t = i / _SR
        env = math.exp(-t * {"kick": 22, "snare": 30, "hat": 90}[voice])
        if voice == "kick":
            freq = 120 * math.exp(-t * 8) + 45  # pitch drop
            s = math.sin(2 * math.pi * freq * t)
        elif voice == "snare":
            s = 0.5 * math.sin(2 * math.pi * 180 * t) + 0.5 * (rng.uniform(-1, 1))
        else:  # hat: filtered noise
            s = rng.uniform(-1, 1)
        out.append(int(max(-1.0, min(1.0, s * env)) * 30000))
    return out


def ensure_drum_samples(dest_dir: Path) -> dict[str, str]:
    """Ensure kick/snare/hat WAVs exist in dest_dir; return {voice: absolute_path}."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    bundled = Path(__file__).resolve().parent.parent.parent / "data" / "drumkit"
    paths: dict[str, str] = {}
    for voice in _VOICES:
        src = bundled / f"{voice}.wav"
        target = dest_dir / f"{voice}.wav"
        if src.exists():
            if not target.exists():
                target.write_bytes(src.read_bytes())
        elif not target.exists():
            samples = _synth(voice)
            with wave.open(str(target), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(_SR)
                w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
        paths[voice] = str(target)
    return paths


def load_drumkit(bridge, track: str) -> dict:
    """Ensure the stock 3-voice kit samples exist and load them on `track` via `bridge`.
    Returns the bridge's add_instrument result ({track, loaded, already_present})."""
    import tempfile
    from pathlib import Path

    samples = ensure_drum_samples(Path(tempfile.gettempdir()) / "orpheus_drumkit")
    return bridge.call("add_instrument", track=track, kind="drumkit", samples=samples)
