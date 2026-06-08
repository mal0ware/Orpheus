"""Audio feature extraction — the post-FX 'sonic signature'.

numpy + soundfile + pyloudnorm are required; librosa (the [analysis] extra) sharpens
spectral features. Each upgrade is opt-in with graceful degradation — the core pipeline
runs without librosa, just coarser. Pure functions over WAV paths; no REAPER.
"""

from __future__ import annotations

from orpheus_mcp.models import AudioCharacter


def analyze_audio_character(wav_path: str) -> AudioCharacter:
    """Compute 3-band energy, spectral centroid, LUFS, true peak, crest factor, stereo width.

    Mirrors dschuler36's four objective analyses (Level / Frequency / Stereo / Dynamics),
    made into one fingerprint vector. Implementation (M2): soundfile read → numpy band split
    → pyloudnorm LUFS → optional librosa centroid/rolloff/MFCC/chroma.
    """
    raise NotImplementedError("M2 — see docs/roadmap.md")


def has_librosa() -> bool:
    """Whether the optional high-fidelity spectral path is available."""
    try:
        import librosa  # noqa: F401

        return True
    except ImportError:
        return False
