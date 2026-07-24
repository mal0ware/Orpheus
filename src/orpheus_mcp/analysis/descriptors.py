"""``analysis/descriptors.py`` (Task 6.1, §7): the per-source ``SoundCharacter`` descriptor
engine — spectral / dynamics / harmonics / stereo / time-fx / pitch, measured POST-FX on an
already-decoded source.

Two entry points feed ``characterize`` (``tools/reference.py``):

- :func:`resolve_source` turns a stem handle (or a literal path) into an :class:`AudioSource`
  (decoded ``Pcm`` + honest :class:`SourceMeta`). It is the one place that translates the §6
  ladder's ``StemResult.notes`` signalling into typed meta: a ``"complement=True"`` note ⇒
  ``SourceMeta.complement=True`` (§6 R3), and the ACTUAL audio's channel count ⇒
  ``SourceMeta.channels`` — a dual-mono stem (two identical channels) reports ``1`` so the §7
  mono policy fires (stereo descriptors stay ``None``, never fabricated).
- :func:`compute_character` runs the DSP and returns a :class:`SoundCharacter`. Every block's
  KEY NAMES are the locked §15.3 contract (Task 10.1 cross-checks them against
  ``data/intent_rules.json`` at load time) — including the six rule bands computed HERE
  precisely because the log-spaced sketch bands do not align with the rule table's musical
  bands.

librosa is imported lazily inside the functions that need it (yin/hpss/stft) — the module
imports without the [analysis] extra so ``resolve_source`` and the numpy-only paths still work.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from orpheus_mcp.analysis import cachefs, model_registry
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import SoundCharacter, SourceMeta, StemHandle

if TYPE_CHECKING:
    from orpheus_mcp.analysis.cachefs import CacheFS, Pcm
    from orpheus_mcp.analysis.separation import SeparatorEngine

__all__ = [
    "AudioSource",
    "MELODIC_STEM_TYPES",
    "resolve_source",
    "compute_character",
]

_EPS = 1e-12
_BAND_FLOOR_DB = -120.0

# Melodic stem types get a pitch block; everything else reports ``pitch=None`` (§7). Kept as a
# module constant so the "which stems have a fundamental worth tracking" decision lives in ONE
# place, not scattered ``in (...)`` checks.
MELODIC_STEM_TYPES = frozenset(
    {"vocals", "vocal", "guitar", "bass", "lead", "synth-lead", "keys", "piano", "strings"}
)

# 8 log-spaced sketch bands, 60 Hz–16 kHz — the SAME edges recon.py uses, but stamped with the
# §15.3 underscore key format (``band_60_120_db``) rather than recon's ``"60-120"`` dict keys.
_SKETCH_EDGES_HZ = tuple(float(e) for e in np.geomspace(60.0, 16000.0, 9))

# The six §15.3 rule bands (rules 4/2/1/3 + rule-13 dynamic Δ). Musical bands that do NOT align
# with the log-spaced sketch, so they are measured here directly against these exact edges.
_RULE_BANDS_HZ: tuple[tuple[str, float, float], ...] = (
    ("band_100_250_db", 100.0, 250.0),
    ("band_200_400_db", 200.0, 400.0),
    ("band_2000_5000_db", 2000.0, 5000.0),
    ("band_10000_16000_db", 10000.0, 16000.0),
)
_RULE_DYN_BAND_HZ = (5000.0, 9000.0)  # band_5000_9000_dyn_db (rule 13's per-frame std)


# --------------------------------------------------------------------------- #
# AudioSource + resolve_source
# --------------------------------------------------------------------------- #


@dataclass
class AudioSource:
    """A decoded source ready for §7 characterization: the audio (``pcm``) plus what it IS
    (``meta``). Plain dataclass — it carries an ndarray and never crosses a tool boundary."""

    pcm: Pcm
    meta: SourceMeta


def _effective_channels(samples: np.ndarray) -> int:
    """The HONEST channel count for the §7 mono policy: a true-mono buffer is 1; a 2-channel
    buffer whose channels are identical within ``CFG.DUAL_MONO_TOLERANCE`` (a mono source fed
    dual-mono to a stereo-only separator, §7) is ALSO 1 — its "stereo" carries no width, so
    stereo descriptors must stay N/A rather than be fabricated from a fake stereo field. Any
    genuine L≠R buffer is its real channel count."""
    if samples.ndim < 2 or samples.shape[1] < 2:
        return 1
    if samples.shape[1] == 2 and np.allclose(
        samples[:, 0], samples[:, 1], atol=CFG.DUAL_MONO_TOLERANCE, rtol=0.0
    ):
        return 1
    return samples.shape[1]


def resolve_source(handle: StemHandle | str, cache: CacheFS) -> AudioSource:  # noqa: ARG001
    """Resolve a stem handle (or a literal audio path) to an :class:`AudioSource`.

    - ``str`` ⇒ a literal path: decoded verbatim, ``stem_type="mix-region"``,
      ``produced_by="file"``, ``complement=False``.
    - :class:`StemHandle` ⇒ a cached/produced stem: decoded from ``handle.stem_path``, with
      ``stem_type``/``produced_by``/``region`` taken from the handle and ``complement`` set
      from the §6 R3 ``"complement=True"`` note (``StemResult.notes`` → typed meta).

    ``channels`` is always the ACTUAL audio's effective channel count (dual-mono ⇒ 1, §7).
    ``cache`` is accepted for signature parity / future descriptor-cache reuse; decoding goes
    through ``cachefs.decode`` (its own file-bytes memo already avoids re-decodes)."""
    if isinstance(handle, str):
        pcm = cachefs.decode(handle)
        meta = SourceMeta(
            stem_type="mix-region",
            produced_by="file",
            region=None,
            sample_rate=pcm.sample_rate,
            channels=_effective_channels(pcm.samples),
            complement=False,
        )
        return AudioSource(pcm=pcm, meta=meta)

    pcm = cachefs.decode(handle.stem_path)
    complement = any("complement=True" in n for n in handle.notes)
    meta = SourceMeta(
        stem_type=handle.stem_type,
        produced_by=handle.produced_by,
        region=handle.region,
        sample_rate=pcm.sample_rate,
        channels=_effective_channels(pcm.samples),
        complement=complement,
    )
    return AudioSource(pcm=pcm, meta=meta)


# --------------------------------------------------------------------------- #
# compute_character — the orchestrator
# --------------------------------------------------------------------------- #


def compute_character(
    src: AudioSource, *, dereverb: SeparatorEngine | None = None
) -> SoundCharacter:
    """Measure the §7 ``SoundCharacter`` of ``src``. ``dereverb`` (a ``SeparatorEngine``), when
    supplied AND ``src.meta.stem_type == "vocals"``, drives the wet−dry reverb measurement
    (``reverb_approx=False``); otherwise the reverb decay-slope heuristic runs
    (``reverb_approx=True``). Also STAMPS ``src.meta.bandwidth_hz`` with the measured spectral
    cliff so it rides on ``SoundCharacter.source`` for §8."""
    samples = src.pcm.samples
    if samples.ndim < 2:
        samples = samples.reshape(-1, 1)
    sr = src.pcm.sample_rate
    mono = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)

    notes: list[str] = []

    bandwidth = _bandwidth_hz(mono, sr)
    src.meta.bandwidth_hz = bandwidth

    spectral = _spectral_descriptors(mono, sr)
    dynamics = _dynamics_descriptors(samples, mono, sr)
    harmonics = _harmonics_descriptors(mono, sr)

    if src.meta.channels == 1:
        stereo: dict[str, float | None] = {
            "width_low": None,
            "width_mid": None,
            "width_high": None,
            "corr": None,
        }
    else:
        stereo = _stereo_descriptors(samples, sr)

    time_fx = _time_fx_descriptors(mono, sr, src.meta, dereverb, notes)

    if src.meta.stem_type in MELODIC_STEM_TYPES:
        pitch: dict[str, float | str | None] | None = _pitch_descriptors(mono, sr)
    else:
        pitch = None

    if src.meta.complement:
        notes.append(
            "complement residual (§6 R3): inherits subtracted stems' tails — tail-sensitive "
            "descriptors (reverb/decay) are discounted downstream (§8)."
        )

    return SoundCharacter(
        source=src.meta,
        spectral=spectral,
        dynamics=dynamics,
        harmonics=harmonics,
        stereo=stereo,
        time_fx=time_fx,
        pitch=pitch,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Shared spectral helpers
# --------------------------------------------------------------------------- #


def _band_db(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """Per-band RMS in dB (Parseval: RMS² == total spectral power, ×2 for the folded negative
    frequencies), floored — mirrors recon.py / audio.py so band numbers are comparable."""
    band_power = float(power[(freqs >= lo) & (freqs < hi)].sum())
    linear_rms = math.sqrt(2.0 * band_power)
    if linear_rms <= _EPS:
        return _BAND_FLOOR_DB
    return max(20.0 * math.log10(linear_rms), _BAND_FLOOR_DB)


def _spectral_descriptors(mono: np.ndarray, sr: int) -> dict[str, float | None]:
    spectrum = np.abs(np.fft.rfft(mono)) / max(len(mono), 1)
    power = spectrum**2
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)

    out: dict[str, float | None] = {}

    total_power = float(power.sum())
    out["centroid_hz"] = (
        float((freqs * power).sum() / total_power) if total_power > _EPS else None
    )
    out["rolloff_hz"] = _rolloff_hz(power, freqs)
    out["tilt_db_per_oct"] = _spectral_tilt(mono, sr)

    for lo, hi in zip(_SKETCH_EDGES_HZ, _SKETCH_EDGES_HZ[1:], strict=False):
        out[f"band_{int(lo)}_{int(hi)}_db"] = _band_db(power, freqs, lo, hi)

    for name, lo, hi in _RULE_BANDS_HZ:
        out[name] = _band_db(power, freqs, lo, hi)
    out["band_5000_9000_dyn_db"] = _band_dynamics_db(mono, sr, *_RULE_DYN_BAND_HZ)

    out["hp_ratio"] = _harmonic_percussive_ratio(mono)
    return out


def _rolloff_hz(power: np.ndarray, freqs: np.ndarray, roll: float = 0.85) -> float | None:
    total = float(power.sum())
    if total <= _EPS:
        return None
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, roll * total))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _spectral_tilt(mono: np.ndarray, sr: int) -> float | None:
    """Spectral slope in dB per octave: mean POWER DENSITY per log-spaced band (10·log10 of the
    per-bin mean, so wider HF bands don't bias the fit), linearly regressed against log2(center
    frequency). A −6 dB/oct-shaped source reads ≈ −6."""
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
    edges = np.geomspace(
        CFG.SPECTRAL_TILT_LO_HZ, min(CFG.SPECTRAL_TILT_HI_HZ, sr / 2.0), CFG.SPECTRAL_TILT_BANDS + 1
    )
    centers: list[float] = []
    band_db: list[float] = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue
        mean_power = float(power[mask].mean())
        if mean_power <= _EPS:
            continue
        centers.append(math.sqrt(lo * hi))
        band_db.append(10.0 * math.log10(mean_power))
    if len(centers) < 3:
        return None
    x = np.log2(np.asarray(centers))
    slope = float(np.polyfit(x, np.asarray(band_db), 1)[0])
    return slope


def _band_dynamics_db(mono: np.ndarray, sr: int, lo: float, hi: float) -> float | None:
    """Rule-13 "dynamic Δ": the std-dev (in dB) of per-frame energy in ``[lo, hi]`` — a steady
    band reads ~0, a pumping/gated band reads high. STFT magnitude → per-frame band power →
    dB → std."""
    try:
        import librosa
    except ImportError:
        return None
    n_fft = min(CFG.DESCRIPTOR_STFT_N, _pow2_le(len(mono)))
    if n_fft < 64:
        return None
    stft = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=n_fft // 4)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return None
    frame_power = stft[mask, :].sum(axis=0)
    frame_power = frame_power[frame_power > _EPS]
    if frame_power.size < 2:
        return None
    frame_db = 10.0 * np.log10(frame_power)
    return float(np.std(frame_db))


def _harmonic_percussive_ratio(mono: np.ndarray) -> float | None:
    """Harmonic energy fraction: ``harm_power / (harm_power + perc_power)`` from librosa HPSS —
    1.0 = purely tonal, 0.0 = purely percussive. Reported under ``spectral`` as the timbre
    tonal/transient balance."""
    try:
        import librosa
    except ImportError:
        return None
    if len(mono) < 512:
        return None
    harm, perc = librosa.effects.hpss(mono)
    hp = float(np.mean(harm.astype(np.float64) ** 2))
    pp = float(np.mean(perc.astype(np.float64) ** 2))
    if hp + pp <= _EPS:
        return None
    return hp / (hp + pp)


def _pow2_le(n: int) -> int:
    return 1 << (n.bit_length() - 1) if n >= 1 else 0


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #


def _dynamics_descriptors(
    samples: np.ndarray, mono: np.ndarray, sr: int
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    out["lufs_i"] = _integrated_lufs(samples, sr)
    out["lufs_s_max"] = _short_term_lufs_max(samples, sr)
    out["crest_db"] = _crest_db(mono)

    attack_p50, attack_p90 = _attack_percentiles(mono, sr)
    out["attack_ms_p50"] = attack_p50
    out["attack_ms_p90"] = attack_p90
    out["env_kurtosis"] = _env_kurtosis(mono, sr)
    return out


def _integrated_lufs(samples: np.ndarray, sr: int) -> float | None:
    import pyloudnorm

    data = samples.astype(np.float64)
    try:
        lufs = float(pyloudnorm.Meter(sr).integrated_loudness(data))
    except (ValueError, IndexError):
        return None
    return lufs if math.isfinite(lufs) else None


def _short_term_lufs_max(samples: np.ndarray, sr: int, window_s: float = 3.0) -> float | None:
    """Max short-term (3 s) loudness: integrated loudness over sliding 3 s windows (1 s hop).
    Falls back to the whole-signal integrated value when the source is shorter than one
    window."""
    import pyloudnorm

    data = samples.astype(np.float64)
    n = data.shape[0]
    win = int(window_s * sr)
    if n < win:
        return _integrated_lufs(samples, sr)
    meter = pyloudnorm.Meter(sr)
    hop = max(1, int(sr))
    best = -math.inf
    for start in range(0, n - win + 1, hop):
        try:
            val = float(meter.integrated_loudness(data[start : start + win]))
        except (ValueError, IndexError):
            continue
        if math.isfinite(val) and val > best:
            best = val
    return best if math.isfinite(best) else None


def _crest_db(mono: np.ndarray) -> float | None:
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
    if peak <= _EPS or rms <= _EPS:
        return None
    return 20.0 * math.log10(peak / rms)


def _attack_percentiles(mono: np.ndarray, sr: int) -> tuple[float | None, float | None]:
    """Per-onset attack time (onset → next local envelope peak), p50/p90 in ms. Empty when no
    onsets are found (a steady tone has no attacks)."""
    try:
        import librosa
    except ImportError:
        return None, None
    if len(mono) < 2048:
        return None, None
    hop = 512
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames"
    )
    if len(onsets) == 0:
        return None, None
    attacks: list[float] = []
    for o in onsets:
        end = min(o + 20, len(onset_env))
        if end <= o + 1:
            continue
        peak_rel = int(np.argmax(onset_env[o:end]))
        attacks.append(peak_rel * hop / sr * 1000.0)
    if not attacks:
        return None, None
    arr = np.asarray(attacks)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 90))


def _env_kurtosis(mono: np.ndarray, sr: int) -> float | None:
    """Excess kurtosis of the amplitude envelope — spiky/transient material reads high, a
    steady drone reads near a normal-distribution baseline."""
    hop = max(1, int(sr * CFG.DESCRIPTOR_ENV_HOP_MS / 1000.0))
    env = _peak_envelope(mono, hop)
    if env.size < 4:
        return None
    x = env.astype(np.float64)
    mean = x.mean()
    var = x.var()
    if var <= _EPS:
        return None
    return float(np.mean((x - mean) ** 4) / (var**2) - 3.0)


# --------------------------------------------------------------------------- #
# Harmonics — even/odd flavor on sustained (onset-free) segments
# --------------------------------------------------------------------------- #


def _harmonics_descriptors(mono: np.ndarray, sr: int) -> dict[str, float | str | None]:
    seg = _sustained_window(mono, sr)
    out: dict[str, float | str | None] = {
        "even_db": None,
        "odd_db": None,
        "inharmonicity": None,
        "flavor": "clean",
    }
    if seg is None or seg.size < 1024:
        return out

    f0 = _estimate_f0(seg, sr)
    if f0 is None or f0 <= 0.0:
        return out

    windowed = seg.astype(np.float64) * np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    power = spectrum**2
    tol = CFG.HARMONIC_TOLERANCE_HZ

    def harmonic_power(k: int) -> float:
        fc = k * f0
        if fc >= sr / 2.0:
            return 0.0
        mask = (freqs >= fc - tol) & (freqs <= fc + tol)
        return float(power[mask].sum())

    even = sum(harmonic_power(k) for k in (2, 4, 6, 8))
    odd = sum(harmonic_power(k) for k in (3, 5, 7, 9))
    even_db = 10.0 * math.log10(even) if even > _EPS else _BAND_FLOOR_DB
    odd_db = 10.0 * math.log10(odd) if odd > _EPS else _BAND_FLOOR_DB
    out["even_db"] = even_db
    out["odd_db"] = odd_db
    out["inharmonicity"] = _inharmonicity(power, freqs, f0, sr)

    delta = even_db - odd_db
    if delta > CFG.FLAVOR_EVEN_ODD_DB:
        out["flavor"] = "warm-even"
    elif delta < -CFG.FLAVOR_EVEN_ODD_DB:
        out["flavor"] = "gritty-odd"
    else:
        out["flavor"] = "clean"
    return out


def _inharmonicity(power: np.ndarray, freqs: np.ndarray, f0: float, sr: int) -> float | None:
    """Mean relative deviation of the strongest partial in each harmonic slot from an exact
    integer multiple of ``f0`` — 0 for a perfectly harmonic tone, growing with stiffness."""
    tol = CFG.HARMONIC_TOLERANCE_HZ
    devs: list[float] = []
    for k in range(1, 9):
        fc = k * f0
        if fc >= sr / 2.0:
            break
        mask = (freqs >= fc - tol) & (freqs <= fc + tol)
        if not mask.any() or power[mask].sum() <= _EPS:
            continue
        peak_freq = float(freqs[mask][int(np.argmax(power[mask]))])
        devs.append(abs(peak_freq - fc) / fc)
    return float(np.mean(devs)) if devs else None


def _sustained_window(mono: np.ndarray, sr: int) -> np.ndarray | None:
    """The longest onset-free window ≥ ``CFG.SUSTAIN_MIN_WINDOW_S`` (a signal with no onsets is
    wholly sustained). Returns ``None`` only when even the whole signal is shorter than the
    floor."""
    min_len = int(CFG.SUSTAIN_MIN_WINDOW_S * sr)
    if mono.size < min_len:
        return None
    try:
        import librosa
    except ImportError:
        return mono
    hop = 512
    if len(mono) < 2048:
        return mono
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, units="samples"
    )
    if len(onsets) == 0:
        return mono
    bounds = [0, *[int(o) for o in onsets], len(mono)]
    best: np.ndarray | None = None
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        seg = mono[lo:hi]
        if best is None or seg.size > best.size:
            best = seg
    # A real note-onset stream yields a long sustained gap; a steady tone can still trip
    # spurious spectral-flux onsets that leave no ≥ floor gap — fall back to the whole signal
    # (it IS sustained) rather than abandoning the harmonic read.
    if best is not None and best.size >= min_len:
        return best
    return mono


def _estimate_f0(seg: np.ndarray, sr: int) -> float | None:
    try:
        import librosa
    except ImportError:
        return None
    try:
        f0 = librosa.yin(
            seg.astype(np.float32), fmin=50.0, fmax=min(2000.0, sr / 2.0 - 1.0), sr=sr
        )
    except Exception:  # noqa: BLE001 - yin can raise on degenerate/too-short input
        return None
    f0 = f0[np.isfinite(f0)]
    if f0.size == 0:
        return None
    return float(np.median(f0))


# --------------------------------------------------------------------------- #
# Stereo — per-band M/S width + correlation (only reached when channels >= 2)
# --------------------------------------------------------------------------- #


def _stereo_descriptors(samples: np.ndarray, sr: int) -> dict[str, float | None]:
    left = samples[:, 0].astype(np.float64)
    right = samples[:, 1].astype(np.float64)
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    def band_width(lo: float, hi: float) -> float | None:
        m = _bandlimit(mid, sr, lo, hi)
        s = _bandlimit(side, sr, lo, hi)
        m_rms = math.sqrt(float(np.mean(m**2)))
        s_rms = math.sqrt(float(np.mean(s**2)))
        total = m_rms + s_rms
        return s_rms / total if total > _EPS else None

    corr: float | None = None
    if np.std(left) > _EPS and np.std(right) > _EPS:
        corr = float(np.corrcoef(left, right)[0, 1])

    return {
        "width_low": band_width(0.0, 250.0),
        "width_mid": band_width(250.0, 4000.0),
        "width_high": band_width(4000.0, sr / 2.0),
        "corr": corr,
    }


def _bandlimit(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """FFT band-pass (zero bins outside ``[lo, hi]``) — cheap and dependency-free; stereo width
    only needs the per-band RMS, not a phase-linear filter."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    spectrum[(freqs < lo) | (freqs >= hi)] = 0.0
    return np.fft.irfft(spectrum, len(x))


# --------------------------------------------------------------------------- #
# time_fx — delay / modulation / reverb
# --------------------------------------------------------------------------- #


def _time_fx_descriptors(
    mono: np.ndarray,
    sr: int,
    meta: SourceMeta,
    dereverb: SeparatorEngine | None,
    notes: list[str],
) -> dict[str, float | str | bool | None]:
    out: dict[str, float | str | bool | None] = {}

    delay_ms, feedback, conf = _delay(mono, sr)
    out["delay_ms"] = delay_ms
    out["delay_feedback"] = feedback
    out["delay_conf"] = conf
    out["delay_approx"] = True  # §7: tempo-sync confound is never resolvable from audio alone

    rate, depth, mod_class = _modulation(mono, sr)
    out["mod_rate_hz"] = rate
    out["mod_depth"] = depth
    out["mod_class"] = mod_class

    reverb = _reverb(mono, sr, meta, dereverb, notes)
    out.update(reverb)
    return out


def _peak_envelope(mono: np.ndarray, hop: int) -> np.ndarray:
    n_frames = len(mono) // hop
    if n_frames < 1:
        return np.abs(mono)
    return np.abs(mono[: n_frames * hop]).reshape(n_frames, hop).max(axis=1)


def _rms_envelope(mono: np.ndarray, hop: int) -> np.ndarray:
    n_frames = len(mono) // hop
    if n_frames < 1:
        return np.asarray([math.sqrt(float(np.mean(mono**2)))])
    frames = mono[: n_frames * hop].reshape(n_frames, hop).astype(np.float64)
    return np.sqrt((frames**2).mean(axis=1))


def _delay(mono: np.ndarray, sr: int) -> tuple[float | None, float | None, float | None]:
    """Delay time via amplitude-envelope autocorrelation: a discrete echo makes every onset
    recur one delay-time later, so the envelope autocorrelation peaks at the delay lag.
    Returns ``(delay_ms, feedback, confidence)`` — ``feedback`` is the normalized peak height
    (echo-to-source amplitude proxy), ``confidence`` its prominence over the local baseline."""
    hop = max(1, int(sr * CFG.DESCRIPTOR_ENV_HOP_MS / 1000.0))
    env = _peak_envelope(mono, hop)
    if env.size < 8:
        return None, None, None
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[env.size - 1 :]
    zero = ac[0]
    if zero <= _EPS:
        return None, None, None
    ms_per_lag = hop / sr * 1000.0
    lo = max(1, int(CFG.DELAY_MIN_MS / ms_per_lag))
    hi = min(ac.size, int(CFG.DELAY_MAX_MS / ms_per_lag))
    if hi <= lo + 1:
        return None, None, None
    seg = ac[lo:hi]
    k = lo + int(np.argmax(seg))
    peak = ac[k]
    if peak <= 0.0:
        return None, None, None
    feedback = float(np.clip(peak / zero, 0.0, 1.0))
    baseline = float(np.median(np.abs(seg)))
    conf = float(np.clip((peak - baseline) / (peak + baseline + _EPS), 0.0, 1.0))
    return float(k * ms_per_lag), feedback, conf


def _modulation(mono: np.ndarray, sr: int) -> tuple[float | None, float | None, str | None]:
    """Periodic modulation via envelope spectral analysis. The amplitude envelope's dominant
    frequency in ``[MOD_RATE_MIN_HZ, MOD_RATE_MAX_HZ]`` gives an AM (tremolo) candidate; an f0
    track's modulation gives an FM (vibrato) candidate. The domain that modulates more (above
    ``MOD_DETECT_MIN_DEPTH``) wins; comb/phaser-vs-chorus is left ``None`` when ambiguous
    (honest, per §7)."""
    hop = max(1, int(sr * CFG.MOD_ENV_HOP_MS / 1000.0))
    env = _rms_envelope(mono, hop)
    env_sr = sr / hop
    amp_rate, amp_depth = _dominant_periodicity(env, env_sr)

    pitch_rate, pitch_depth = _pitch_modulation(mono, sr)

    amp_ok = amp_depth is not None and amp_depth >= CFG.MOD_DETECT_MIN_DEPTH
    pitch_ok = pitch_depth is not None and pitch_depth >= CFG.MOD_DETECT_MIN_DEPTH

    if amp_ok and (not pitch_ok or (amp_depth or 0.0) >= (pitch_depth or 0.0)):
        return amp_rate, amp_depth, "tremolo"
    if pitch_ok:
        return pitch_rate, pitch_depth, "vibrato"
    return None, None, None


def _dominant_periodicity(
    env: np.ndarray, env_sr: float
) -> tuple[float | None, float | None]:
    """Dominant frequency (in the modulation band) of a 1-D signal + its normalized depth
    (single-sided amplitude at the peak / mean level)."""
    if env.size < 8:
        return None, None
    mean_level = float(np.mean(env))
    centered = env - mean_level
    window = np.hanning(len(centered))
    spectrum = np.abs(np.fft.rfft(centered * window))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / env_sr)
    band = (freqs >= CFG.MOD_RATE_MIN_HZ) & (freqs <= CFG.MOD_RATE_MAX_HZ)
    if not band.any():
        return None, None
    idx = np.where(band)[0]
    k = idx[int(np.argmax(spectrum[idx]))]
    # Coherent-gain-corrected single-sided amplitude of the peak component.
    amp = 2.0 * spectrum[k] / float(window.sum())
    depth = amp / mean_level if mean_level > _EPS else None
    return float(freqs[k]), (float(depth) if depth is not None else None)


def _pitch_modulation(mono: np.ndarray, sr: int) -> tuple[float | None, float | None]:
    try:
        import librosa
    except ImportError:
        return None, None
    if len(mono) < 4096:
        return None, None
    try:
        f0 = librosa.yin(mono, fmin=50.0, fmax=min(2000.0, sr / 2.0 - 1.0), sr=sr)
    except Exception:  # noqa: BLE001
        return None, None
    f0 = f0[np.isfinite(f0)]
    if f0.size < 8:
        return None, None
    median = float(np.median(f0))
    if median <= _EPS:
        return None, None
    cents = 1200.0 * np.log2(np.clip(f0, _EPS, None) / median)
    f0_sr = sr / 512.0  # yin's default frame hop
    rate, _depth = _dominant_periodicity(cents, f0_sr)
    # Depth for pitch modulation is reported as a fraction of a semitone (100 cents) so it is
    # comparable to the amplitude depth's [0, 1] scale for the winner-picks decision.
    swing_cents = float(np.percentile(np.abs(cents - np.mean(cents)), 90))
    depth = swing_cents / 100.0
    return rate, depth


def _reverb(
    mono: np.ndarray,
    sr: int,
    meta: SourceMeta,
    dereverb: SeparatorEngine | None,
    notes: list[str],
) -> dict[str, float | str | bool | None]:
    """§7 reverb split: a vocal stem WITH an injected dereverb engine gets a wet−dry
    measurement (``reverb_approx=False``); everything else gets the decay-slope heuristic
    (``reverb_approx=True``)."""
    if meta.stem_type == "vocals" and dereverb is not None:
        measured = _reverb_wet_dry(mono, sr, dereverb)
        if measured is not None:
            return measured
        notes.append(
            "dereverb path unavailable (no registry entry or engine failure) — reverb fell "
            "back to the decay-slope heuristic (reverb_approx=True)."
        )
    return _reverb_heuristic(mono, sr)


def _reverb_wet_dry(
    mono: np.ndarray, sr: int, dereverb: SeparatorEngine
) -> dict[str, float | str | bool | None] | None:
    entry = model_registry.resolve_dereverb_entry()
    if entry is None:
        return None
    try:
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="orpheus_dereverb_") as td:
            wav = Path(td) / "wet.wav"
            sf.write(str(wav), mono.reshape(-1, 1), sr, subtype="FLOAT")
            out_dir = Path(td) / "out"
            produced = dereverb.separate(str(wav), entry, out_dir)
            if not produced:
                return None
            dry_path = produced.get("vocals") or next(iter(produced.values()))
            dry, _dry_sr = sf.read(str(dry_path), always_2d=True, dtype="float32")
    except Exception:  # noqa: BLE001 - a dereverb engine failure degrades gracefully
        return None

    dry_mono = dry.mean(axis=1).astype(np.float64)
    n = min(len(mono), len(dry_mono))
    wet = mono[:n].astype(np.float64)
    dry_mono = dry_mono[:n]
    residual = wet - dry_mono  # the reverberant tail the dereverb removed

    dry_rms = math.sqrt(float(np.mean(dry_mono**2)))
    res_rms = math.sqrt(float(np.mean(residual**2)))
    wet_dry = res_rms / dry_rms if dry_rms > _EPS else None

    early_late = _early_late_ratio(residual, sr)
    rt60_class = _rt60_class(residual, sr)
    return {
        "reverb_wet_dry": wet_dry,
        "reverb_rt60_class": rt60_class,
        "reverb_early_late": early_late,
        "reverb_approx": False,
    }


def _reverb_heuristic(mono: np.ndarray, sr: int) -> dict[str, float | str | bool | None]:
    """Decay-slope heuristic (§7): estimate an RT60 class from the tail decay of the amplitude
    envelope. Honest and coarse — ``reverb_wet_dry``/``reverb_early_late`` stay ``None`` (they
    are not measurable without a dry reference)."""
    return {
        "reverb_wet_dry": None,
        "reverb_rt60_class": _rt60_class(mono, sr),
        "reverb_early_late": None,
        "reverb_approx": True,
    }


def _rt60_class(sig: np.ndarray, sr: int) -> str | None:
    """Coarse RT60 bucket from the decay slope of the envelope in dB: fit a line to the falling
    portion, extrapolate to −60 dB. ``None`` when there is no measurable decay."""
    hop = max(1, int(sr * 0.01))
    env = _rms_envelope(np.asarray(sig, dtype=np.float32), hop)
    if env.size < 8:
        return None
    env_db = 20.0 * np.log10(np.clip(env, _EPS, None))
    peak_idx = int(np.argmax(env_db))
    tail = env_db[peak_idx:]
    if tail.size < 4:
        return None
    x = np.arange(tail.size) * hop / sr
    slope = float(np.polyfit(x, tail, 1)[0])  # dB per second
    if slope >= -1.0:
        return None  # essentially no decay (steady tone / noise)
    rt60_s = 60.0 / abs(slope)
    if rt60_s < 0.4:
        return "short"
    if rt60_s < 1.2:
        return "medium"
    return "long"


def _early_late_ratio(residual: np.ndarray, sr: int, split_s: float = 0.08) -> float | None:
    split = int(split_s * sr)
    if residual.size <= split:
        return None
    early = float(np.sum(residual[:split] ** 2))
    late = float(np.sum(residual[split:] ** 2))
    if early + late <= _EPS:
        return None
    return early / (early + late)


# --------------------------------------------------------------------------- #
# Pitch — melodic stems only
# --------------------------------------------------------------------------- #


def _pitch_descriptors(mono: np.ndarray, sr: int) -> dict[str, float | str | None]:
    out: dict[str, float | str | None] = {
        "f0_median_hz": None,
        "vibrato_rate_hz": None,
        "vibrato_depth_cents": None,
        "portamento": None,
    }
    try:
        import librosa
    except ImportError:
        return out
    if len(mono) < 4096:
        return out
    try:
        f0 = librosa.yin(mono, fmin=50.0, fmax=min(2000.0, sr / 2.0 - 1.0), sr=sr)
    except Exception:  # noqa: BLE001
        return out
    finite = f0[np.isfinite(f0)]
    if finite.size == 0:
        return out
    median = float(np.median(finite))
    out["f0_median_hz"] = median
    if median <= _EPS:
        return out

    cents = 1200.0 * np.log2(np.clip(finite, _EPS, None) / median)
    f0_sr = sr / 512.0
    rate, _depth = _dominant_periodicity(cents, f0_sr)
    out["vibrato_rate_hz"] = rate
    out["vibrato_depth_cents"] = float(np.percentile(np.abs(cents - np.mean(cents)), 90))
    # Portamento: the largest smooth pitch glide (cents traversed over a monotone run).
    out["portamento"] = _portamento_cents(cents)
    return out


def _portamento_cents(cents: np.ndarray) -> float | None:
    if cents.size < 4:
        return None
    diffs = np.diff(cents)
    best = 0.0
    run = 0.0
    sign = 0
    for d in diffs:
        s = 1 if d > 0 else (-1 if d < 0 else 0)
        if s != 0 and s == sign:
            run += d
        else:
            run = d
            sign = s
        if abs(run) > abs(best):
            best = run
    return float(abs(best))


def _bandwidth_hz(mono: np.ndarray, sr: int) -> float | None:
    """§7 spectral cliff: the upper edge of the highest fixed-width band whose mean power is
    within ``CFG.BANDWIDTH_CLIFF_DROP_DB`` of the peak band. A brickwall-limited source's
    band above the cutoff drops below the floor and is excluded, so this reads back the cutoff."""
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
    bw = CFG.BANDWIDTH_BAND_HZ
    edges = np.arange(0.0, sr / 2.0 + bw, bw)
    band_db: list[float] = []
    uppers: list[float] = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            continue
        mean_power = float(power[mask].mean())
        band_db.append(10.0 * math.log10(mean_power) if mean_power > _EPS else _BAND_FLOOR_DB)
        uppers.append(float(hi))
    if not band_db:
        return None
    db_arr = np.asarray(band_db)
    peak = float(db_arr.max())
    qualifying = np.where(db_arr >= peak - CFG.BANDWIDTH_CLIFF_DROP_DB)[0]
    if qualifying.size == 0:
        return None
    return uppers[int(qualifying.max())]
