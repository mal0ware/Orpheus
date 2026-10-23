"""``analysis/separation.py`` (Task 4.1, §3.1d/§5/§7/§15.5): the separator-engine adapters
and ``separate_region`` — the region-cropped, VRAM-guarded, length-aligned core every stem in
the reference-analysis pipeline flows through.

Two real engines sit behind one structural ``SeparatorEngine`` Protocol:

- **``MSSTEngine``** — the vendored SCNet path (``_vendor/msst/inference.py``), run **in-process**
  behind lazy imports (torch only loads when it runs). GPU-verified at Task 0.3.
- **``AudioSeparatorEngine``** — a **subprocess** invocation of the isolated
  ``~/.orpheus_separator_env`` ``audio-separator`` CLI (list argv, no shell, UTF-8 child stdio,
  the same isolation pattern as ``transcribe_cli``). ``audio_separator`` is NEVER imported into
  the main env (D9: every ``audio-separator`` release transitively caps ``fastmcp<3.4``).

**The audio-separator custom-checkpoint contract (Task 4.1 carried problem):** audio-separator's
CLI only accepts model filenames present in its OWN model registry — it rejects arbitrary
checkpoint paths ("Model file … not found in supported model files"). Our pinned checkpoints for
five of the audio-separator entries (htdemucs-ft / htdemucs-6s / BS-Roformer-SW / DrumSep /
dereverb-anvuew) are byte-identical to what audio-separator would download for those models
(same upstream URLs), so ``model_registry.json`` records, per entry, the exact
``--model_filename`` audio-separator expects (``ModelEntry.audio_separator_model``). The adapter
points ``--model_file_dir`` at our verified ``checkpoint_dir(entry.id)`` and passes that filename:
audio-separator finds our weights already present (matching name) and reuses them, fetching only
its small architecture-config sidecar. Entries with no compatible audio-separator model
(ZFTurbo's 4-stem BS-RoFormer ep_17, the MelBand vocal / becruily guitar community checkpoints,
LarsNet) carry ``audio_separator_model = None`` and this adapter refuses them with an honest error
rather than silently substituting a different model.

``separate_region`` locked behavior (each is a named test): crop ``region ±
CFG.SEPARATION_CONTEXT_PAD_S`` clamped to ``[0, duration]`` (§3.1d) → temp wav → engine under the
process-wide ``GPU_SEMAPHORE`` with the §15.5 VRAM guard (entry that does not fit
``VRAM_FIT_FRACTION × free`` falls to the next ``resolve()`` candidate producing the same stem set,
CPU-capable last; the fallback + reason is recorded in ``StemResult.notes``) → trim outputs back to
the exact requested region (§3.1d) → resample + length-align to the input (§5 gate precondition) →
mono input is fed dual-mono to stereo-only engines, with a §7 note that stereo descriptors stay
N/A downstream. When the chosen entry declares a verified ``ModelEntry.native_sr`` (Task 4.1
follow-up) that differs from the input rate, the cropped feed is resampled UP to ``native_sr``
before the engine runs — off-rate input can hit slow/degraded code paths (SCNet-XL fed 22.05 kHz
hit torch's experimental ComplexHalf istft path: 185.8 s for 17 s of audio) — and the existing
output resample step brings results back down to the input rate.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import soundfile as sf

from orpheus_mcp.analysis import model_registry
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.fetch_models import _filename_for_url
from orpheus_mcp.models import StemResult, Window

if TYPE_CHECKING:
    from orpheus_mcp.analysis.cachefs import Pcm
    from orpheus_mcp.analysis.model_registry import ModelEntry

__all__ = [
    "SeparatorEngine",
    "MSSTEngine",
    "AudioSeparatorEngine",
    "SeparationError",
    "SeparatorEnvMissing",
    "separate_region",
]


class SeparationError(Exception):
    """A separation could not be performed (no compatible engine/model, no VRAM candidate, …)."""


class SeparatorEnvMissing(SeparationError):
    """The isolated ``audio-separator`` env has not been created yet."""


class SeparatorEngine(Protocol):
    """Structural contract every separator engine satisfies — ``separate_region`` only ever
    calls through this. ``separate`` writes one FLAC per produced target into ``out_dir`` and
    returns ``{target: flac_path}``. An engine may advertise ``requires_stereo`` (a plain
    attribute) when it cannot accept mono input; ``separate_region`` reads it via ``getattr``,
    so declaring it is optional."""

    def separate(self, wav_path: str, entry: ModelEntry, out_dir: Path) -> dict[str, str]:
        """Separate ``wav_path`` with ``entry``'s checkpoint, one FLAC per target in ``out_dir``."""
        ...


# --------------------------------------------------------------------------- #
# MSSTEngine — vendored SCNet, in-process
# --------------------------------------------------------------------------- #


class MSSTEngine:
    """The vendored MSST/SCNet engine (``4stem-b/scnet-xl``), run in-process behind lazy
    imports. ``requires_stereo`` is True (SCNet is a stereo model); ``separate_file`` also
    duplicates mono internally, so feeding it dual-mono is a no-op safety."""

    requires_stereo = True

    def separate(self, wav_path: str, entry: ModelEntry, out_dir: Path) -> dict[str, str]:
        from orpheus_mcp._vendor.msst.inference import separate_file

        urls = entry.checkpoint_url if isinstance(entry.checkpoint_url, list) else [
            entry.checkpoint_url
        ]
        if not urls or urls[0] is None:
            raise SeparationError(f"{entry.id}: msst engine needs a checkpoint_url")
        ckpt = model_registry.checkpoint_dir(entry.id) / _filename_for_url(urls[0], entry.id)
        if not ckpt.is_file():
            raise SeparationError(
                f"{entry.id}: SCNet checkpoint not found at {ckpt}. Run `orpheus-mcp "
                "fetch-models` first."
            )
        written = separate_file(ckpt, wav_path, out_dir, output_format="FLAC")
        # SCNet source names (drums/bass/other/vocals) already match our target vocabulary.
        return {name: str(path) for name, path in written.items() if name in entry.targets}


# --------------------------------------------------------------------------- #
# AudioSeparatorEngine — isolated-env CLI subprocess
# --------------------------------------------------------------------------- #


def _separator_env_cli() -> Path:
    """Path to the ``audio-separator`` console script in the isolated env (``~/.orpheus_
    separator_env``) — replicated here (not imported from ``scripts/``) so ``src`` never depends
    on the repo-root scripts package, the same way ``transcribe_cli`` owns its env path."""
    root = Path.home() / ".orpheus_separator_env"
    if sys.platform == "win32":
        return root / "Scripts" / "audio-separator.exe"
    return root / "bin" / "audio-separator"


def _map_audio_separator_outputs(out_dir: Path, targets: list[str]) -> dict[str, str]:
    """Map audio-separator's output FLACs (named ``<in>_(<StemLabel>)_<model>.flac``) to our
    target vocabulary. Two-pass: an exact parenthesized-label match first, then a substring
    fallback for stray labels — first writer wins, so a target is never double-assigned."""
    files = sorted(out_dir.glob("*.flac"))
    labels: list[tuple[Path, str, str]] = []
    for f in files:
        paren = re.findall(r"\(([^)]+)\)", f.stem)
        label = (paren[-1] if paren else f.stem).lower()
        labels.append((f, label, f.stem.lower()))

    result: dict[str, str] = {}
    for target in targets:
        t = target.lower()
        for f, label, _full in labels:
            if label == t and target not in result:
                result[target] = str(f)
                break
    for target in targets:
        if target in result:
            continue
        t = target.lower()
        for f, label, full in labels:
            if str(f) in result.values():
                continue
            if t in label or t in full:
                result[target] = str(f)
                break
    return result


class AudioSeparatorEngine:
    """Drives the isolated-env ``audio-separator`` CLI as a subprocess (D9). Model resolution is
    data-driven from ``ModelEntry.audio_separator_model`` + our verified ``checkpoint_dir`` (see
    the module docstring). ``requires_stereo`` is True (all supported models are stereo)."""

    requires_stereo = True

    def separate(self, wav_path: str, entry: ModelEntry, out_dir: Path) -> dict[str, str]:
        model_filename = entry.audio_separator_model
        if not model_filename:
            raise SeparationError(
                f"{entry.id}: audio-separator has no registry-compatible model for this "
                "checkpoint (audio_separator_model is None) — it cannot be run through the "
                "audio-separator CLI. Use a different registry entry for this target."
            )
        cli = _separator_env_cli()
        if not cli.is_file():
            raise SeparatorEnvMissing(
                "The isolated audio-separator env has not been set up. Run "
                "`python scripts/setup_separator_env.py` first (D9)."
            )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            str(cli),
            str(wav_path),
            "--model_filename",
            model_filename,
            "--model_file_dir",
            str(model_registry.checkpoint_dir(entry.id)),
            "--output_dir",
            str(out_dir),
            "--output_format",
            "flac",
        ]
        # audio-separator prints a rich banner; force UTF-8 child stdio so the Windows cp1252
        # console codepage can't crash the subprocess (same fix as transcribe_cli).
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(  # UTF-8 + replace: never the cp1252 locale (see cachefs)
            argv, capture_output=True, encoding="utf-8", errors="replace", env=env, check=False
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
            raise SeparationError(
                f"{entry.id}: audio-separator CLI failed (exit {proc.returncode}):\n{tail}"
            )
        produced = _map_audio_separator_outputs(out_dir, list(entry.targets))
        if not produced:
            raise SeparationError(
                f"{entry.id}: audio-separator produced no FLAC matching targets {entry.targets} "
                f"in {out_dir} (files: {[p.name for p in out_dir.glob('*.flac')]})."
            )
        return produced


def _default_engine(entry: ModelEntry) -> SeparatorEngine:
    if entry.engine == "msst":
        return MSSTEngine()
    if entry.engine == "audio-separator":
        return AudioSeparatorEngine()
    raise SeparationError(
        f"{entry.id}: engine {entry.engine!r} is not a separator engine (separate_region only "
        "runs msst / audio-separator entries)."
    )


# --------------------------------------------------------------------------- #
# separate_region — crop ± pad, VRAM guard, engine, trim back, length-align
# --------------------------------------------------------------------------- #


def _apply_vram_guard(entry: ModelEntry) -> tuple[ModelEntry, list[str]]:
    """§15.5: keep ``entry`` if it fits ``VRAM_FIT_FRACTION × free`` VRAM, else fall to the
    next ``resolve()`` candidate producing the same stem set (CPU-capable models last). Returns
    ``(chosen_entry, notes)`` — ``notes`` carries the degradation reason string when a fallback
    happened, empty otherwise."""
    free = model_registry.free_vram_gb()
    frac = CFG.VRAM_FIT_FRACTION
    fits = entry.vram_gb <= 0.0 if free is None else entry.vram_gb <= frac * free
    if fits:
        return entry, []

    target0 = entry.targets[0] if entry.targets else ""
    req_set = set(entry.targets)
    candidates = model_registry.resolve(entry.task, target0, vram_free_gb=free)
    # Prefer a candidate whose stem set EXACTLY matches the request (a 4-stem job should not
    # silently fall to a 6-stem model), then by rank; a superset match is the second choice.
    viable = [c for c in candidates if c.id != entry.id and set(c.targets) >= req_set]
    viable.sort(key=lambda c: (0 if set(c.targets) == req_set else 1, c.rank))
    if not viable:
        raise SeparationError(
            f"{entry.id} needs {entry.vram_gb:.1f} GB VRAM which does not fit, and no lower-VRAM "
            f"registry candidate for {entry.task}/{target0} could be found."
        )
    chosen = viable[0]
    free_desc = "no CUDA available" if free is None else f"{free:.1f} GB free"
    chosen_desc = "CPU" if chosen.vram_gb <= 0.0 else f"{chosen.vram_gb:.1f} GB VRAM"
    note = (
        f"VRAM fallback (§15.5): {entry.id} needs {entry.vram_gb:.1f} GB but only {free_desc} "
        f"(< {frac:g}× fit) — fell back to {chosen.id} ({chosen_desc})."
    )
    return chosen, [note]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _resample_to(stem: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample ``(frames, channels)`` to ``target_sr`` (§5 alignment precondition). Lazy
    librosa import — only pays when an engine actually changed the sample rate."""
    if orig_sr == target_sr:
        return stem
    import librosa

    resampled = librosa.resample(stem.T, orig_sr=orig_sr, target_sr=target_sr)
    return np.ascontiguousarray(resampled.T, dtype=np.float32)


def _align_length(stem: np.ndarray, target_len: int) -> np.ndarray:
    """Truncate or zero-pad ``(frames, channels)`` to exactly ``target_len`` frames — the
    separator pad/overrun (§5: an unaligned ``mix − Σstems`` residual is spuriously huge)."""
    frames = stem.shape[0]
    if frames == target_len:
        return stem
    if frames > target_len:
        return stem[:target_len]
    pad = np.zeros((target_len - frames, stem.shape[1]), dtype=stem.dtype)
    return np.concatenate([stem, pad], axis=0)


def separate_region(
    pcm: Pcm,
    entry: ModelEntry,
    region: Window | None,
    engine: SeparatorEngine | None = None,
) -> dict[str, StemResult]:
    """Separate ``pcm`` (optionally cropped to ``region``) with ``entry``, returning one
    :class:`StemResult` per produced target. See the module docstring for the locked behavior.

    ``engine=None`` selects the engine from the (possibly VRAM-downgraded) entry; tests inject a
    stub. ``gates`` on each result is left ``None`` — the §5 gate results are stamped by Task
    4.2's tool wrapper. Final stems are written to a fresh temp directory whose paths ride on the
    results (the Task 4.2 tool relocates them into the §9 cache)."""
    sr = pcm.sample_rate
    samples = pcm.samples if pcm.samples.ndim == 2 else pcm.samples.reshape(-1, 1)
    n = samples.shape[0]
    dur = n / sr if sr else 0.0

    notes: list[str] = []

    # -- region clamp (§3.1d) --------------------------------------------------- #
    if region is None:
        req_start_s, req_end_s = 0.0, dur
    else:
        req_start_s, req_end_s = region.start_s, region.end_s
    r_start_s = _clamp(req_start_s, 0.0, dur)
    r_end_s = max(_clamp(req_end_s, 0.0, dur), r_start_s)
    if region is not None and (req_start_s < 0.0 or req_end_s > dur or req_start_s > dur):
        notes.append(
            f"region clamped to [{r_start_s:.3f}, {r_end_s:.3f}] s "
            f"(requested [{req_start_s:.3f}, {req_end_s:.3f}], song is {dur:.3f} s)."
        )

    r_start_i = int(round(r_start_s * sr))
    r_end_i = min(int(round(r_end_s * sr)), n)
    r_start_i = min(r_start_i, r_end_i)
    expected_len = r_end_i - r_start_i

    # -- crop ± context pad, clamped ------------------------------------------- #
    pad = CFG.SEPARATION_CONTEXT_PAD_S
    c_start_i = max(0, int(round(_clamp(r_start_s - pad, 0.0, dur) * sr)))
    c_end_i = min(n, int(round(_clamp(r_end_s + pad, 0.0, dur) * sr)))
    crop = samples[c_start_i:c_end_i]

    # -- VRAM guard + engine selection ----------------------------------------- #
    chosen, vram_notes = _apply_vram_guard(entry)
    notes.extend(vram_notes)
    active_engine = engine if engine is not None else _default_engine(chosen)

    # -- mono → dual-mono for stereo-only engines (§7) ------------------------- #
    feed = crop
    if crop.shape[1] == 1 and getattr(active_engine, "requires_stereo", False):
        feed = np.repeat(crop, 2, axis=1)
        notes.append(
            "mono source fed dual-mono to a stereo-only engine — stereo descriptors "
            "(width/correlation) stay N/A downstream (§7)."
        )

    # -- native-SR resample (Task 4.1 follow-up) -------------------------------- #
    # A model with a verified native_sr fed off-rate audio can hit slow/degraded code paths
    # (SCNet-XL + 22.05 kHz hit torch's experimental ComplexHalf istft: 185.8 s for 17 s of
    # audio). Resample the CROPPED feed up/down to the model's native rate before the engine;
    # the existing raw→sr resample below (§5 alignment precondition) already brings the output
    # back to the input rate, so no change is needed there.
    engine_sr = sr
    if chosen.native_sr and chosen.native_sr != sr:
        feed = _resample_to(feed, sr, chosen.native_sr)
        engine_sr = chosen.native_sr
        notes.append(
            f"resampled {sr} Hz → {chosen.native_sr} Hz (native_sr) for {chosen.id}; output "
            f"resampled back to {sr} Hz."
        )

    finals_dir = Path(tempfile.mkdtemp(prefix="orpheus_stems_"))
    results: dict[str, StemResult] = {}
    with tempfile.TemporaryDirectory(prefix="orpheus_sep_") as work:
        work_dir = Path(work)
        in_wav = work_dir / "region.wav"
        sf.write(str(in_wav), feed, engine_sr, subtype="FLOAT")

        raw_out = work_dir / "raw"
        raw_out.mkdir(parents=True, exist_ok=True)
        with model_registry.GPU_SEMAPHORE:  # §15.5: serialize ALL GPU work
            raw_map = active_engine.separate(str(in_wav), chosen, raw_out)

        for target, raw_path in raw_map.items():
            if target not in chosen.targets:
                continue
            raw, raw_sr = sf.read(str(raw_path), always_2d=True, dtype="float32")
            raw = _resample_to(raw, int(raw_sr), sr)
            # Trim the ±pad context back to the requested region (§3.1d), by offset within crop.
            trim_start = r_start_i - c_start_i
            trim_end = r_end_i - c_start_i
            trimmed = raw[max(0, trim_start):max(0, trim_end)]
            trimmed = _align_length(trimmed, expected_len)  # §5 length-align precondition

            dest = finals_dir / f"{target.replace('/', '_')}.flac"
            sf.write(str(dest), trimmed, sr, subtype="PCM_24")
            results[target] = StemResult(
                stem_path=str(dest),
                model_id=chosen.id,
                target=target,
                region_trimmed=Window(start_s=r_start_i / sr, end_s=r_end_i / sr),
                sample_rate=sr,
                gates=None,
                notes=list(notes),
            )

    if not results:
        raise SeparationError(
            f"{chosen.id}: engine produced no stems matching targets {chosen.targets}."
        )
    return results
