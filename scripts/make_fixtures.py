"""``scripts/make_fixtures.py`` — synthetic-truth fixture forge (Task 0.4, §10 / §15.1, D4/D8).

Renders deterministic, ground-truth music fixtures headlessly — no REAPER, no DAW — so the
whole reference-analysis pipeline (recon, timeline, chords, separation gates, calibration) can
be tested and calibrated against material whose bpm/key/progression/roles are KNOWN.

Pipeline per role: compose-stack helpers (``resolve_progression`` / ``voice_lead`` /
``bassline_notes`` / ``parse_drum_grid``) build note dicts in beats -> a tiny local Standard-MIDI
writer emits an SMF (``mido`` is NOT a dependency) -> ``sfizz_render`` (pinned Windows release)
renders each SMF through an authored SFZ that maps pinned VSCO2-CE Community-Edition samples
(CC0) -> stems are peak-normalized to -6 dBFS and summed into ``mix.flac``. ``truth.json`` records
role->stem map, bpm, key, and progression.

Assets (sfizz binary + VSCO2-CE samples) are pinned by url + sha256 below and downloaded ONCE
into ``~/.orpheus_fixture_assets/`` (never the repo). Drums use the repo's own deterministic,
seeded synth kit (``orpheus_mcp.drumkit``) — CC0-by-construction — because VSCO2-CE ships no
clean hi-hat one-shot and its multi-velocity percussion naming resists a stable single-file pin;
they are still rendered through ``sfizz_render`` so every role goes through one uniform path.

The committed CI subset (``tests/fixtures/audio/{ci_sparse,ci_vocal}``) is generated here too:
``ci_sparse`` is the instrumental ``no_vocals`` fixture; ``ci_vocal`` adds a vocal lane =
Windows SAPI (``System.Speech``) reading 6 original plain lyric lines (D8), placed on the bar
grid and mixed at -12 dBFS. The SAPI WAV is committed as FLAC so no other box ever needs SAPI;
tests never re-render it (SAPI is not bit-stable). The instrument stems ARE deterministic and the
``@pytest.mark.desktop`` determinism test re-renders ``ci_sparse`` and compares hashes.

Usage:
    uv run python scripts/make_fixtures.py --out ~/.orpheus_fixtures   # full 24 + CI subset
    uv run python scripts/make_fixtures.py --ci-only                   # just the 2 CI fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf

from orpheus_mcp.drumkit import ensure_drum_samples
from orpheus_mcp.theory.chords import resolve_progression, voice_lead
from orpheus_mcp.theory.music_theory_data import NOTE_TO_PC, scale_notes
from orpheus_mcp.theory.patterns import DRUM_PATTERNS, bassline_notes, parse_drum_grid

# --------------------------------------------------------------------------- #
# Pinned assets (url + sha256). Downloaded to ~/.orpheus_fixture_assets, never the repo.
# --------------------------------------------------------------------------- #
ASSETS_DIR = Path.home() / ".orpheus_fixture_assets"

SFIZZ_VERSION = "1.2.3"
SFIZZ_ZIP_URL = (
    "https://github.com/sfztools/sfizz/releases/download/1.2.3/sfizz-1.2.3-win64.zip"
)
SFIZZ_ZIP_SHA256 = "7431cbe9b7f2e71a9ff4b12fbbe82c766dbf75b1eedecca627e7506cb06a9fcc"
SFIZZ_RENDER_MEMBER = "bin/Release/sfizz_render.exe"
SFIZZ_DLL_MEMBER = "bin/Release/sfizz.dll"

# VSCO-2 Community Edition (CC0-1.0), pinned to an immutable commit for reproducible raw fetch.
VSCO2_COMMIT = "440300901dfe9275fd84e0b7763af1f8443ae62e"
VSCO2_RAW_BASE = f"https://raw.githubusercontent.com/sgossner/VSCO-2-CE/{VSCO2_COMMIT}"


@dataclass(frozen=True)
class SamplePin:
    """A single pinned VSCO2-CE sample: repo path, sha256, and its true MIDI keycenter."""

    repo_path: str
    sha256: str
    keycenter: int


SAMPLE_PINS: dict[str, SamplePin] = {
    # Upright Piano, two samples split at key 65 for a less-abused repitch across the chord range.
    "piano_lo": SamplePin(
        "Keys/Upright Piano/Player_dyn1_rr1_016.wav",
        "f1a3a542cf70a13aa865ce130d50d72cd735449e605d1c5a45dc06b5a60b27fd",
        53,
    ),
    "piano_hi": SamplePin(
        "Keys/Upright Piano/Player_dyn1_rr1_028.wav",
        "a9fbd1af0703b294d3267f8f88fca4b4366e3b772f7d780acf2de1cbf2056355",
        77,
    ),
    # Solo Contrabass pizzicato E2 for the bass role.
    "contrabass": SamplePin(
        "Strings/Solo Contrabass/Pizz/BKCtbss_Pizz_E2_v1_rr1.wav",
        "b5c3dd4943ea9a3f1aa23ed07c827ec8851e28a7254bab7811b706c09fceb231",
        40,
    ),
    # Solo Flute sustained A4 for the lead role.
    "flute": SamplePin(
        "Woodwinds/Flute/susNV/LDFlute_susNV_A4_v1_1.wav",
        "c8fd15bb561dbdac6d35ffd9c5e49413131d96f8948d5bdf03d4a03fa344c5c0",
        69,
    ),
}

# --------------------------------------------------------------------------- #
# Fixture render constants
# --------------------------------------------------------------------------- #
FIXTURE_SR = 22050          # §10: 22.05 kHz stereo FLAC
RENDER_SR = 22050
BARS = 8                    # 8 bars (in the §10 8-16 range; the amended CI size floor)
BEATS_PER_BAR = 4
BARS_PER_CHORD = 2          # 4-chord progressions cover 8 bars once through
TPQN = 480
STEM_DBFS = -6.0            # §10 stem peak-normalize target
VOCAL_DBFS = -12.0          # D8 vocal-lane mix level
TAIL_S = 0.4               # short release tail kept after the last bar
MIX_CEILING = 0.9909        # ~-0.08 dBFS safety ceiling when summed stems would clip

# The 6 original, plain, non-copyrighted lyric lines the SAPI vocal lane reads (D8).
LYRIC_LINES: list[str] = [
    "Morning light comes over the hill",
    "I count the hours slow and still",
    "Every step a steady sound",
    "Walking on the open ground",
    "Hold the line and carry through",
    "Simple words will see us new",
]


# --------------------------------------------------------------------------- #
# Genre / density recipes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Genre:
    """A fixture genre recipe. Progressions carry >=1 maj7 AND >=1 slash chord (the §13 bar)."""

    name: str
    bpm: int
    key_root: str
    key_mode: str
    progression: str      # comma-separated absolute chord symbols; a '/' marks a slash bass


# 8 genres. classical/hiphop/lofi mirror theory/genre_profiles.py; the rest are fixture recipes
# (amendment 2 — recipes, not product features). Each progression has a maj7 and a slash chord.
GENRES: list[Genre] = [
    Genre("classical", 72, "C", "major", "Cmaj7,F/A,G7,C"),
    Genre("hiphop", 90, "A", "minor", "Am7,Dm7,Fmaj7,E/G#"),
    Genre("lofi", 75, "D", "minor", "Dm7,Gmaj7,C/E,A7"),
    Genre("house", 124, "A", "minor", "Am7,Fmaj7,C/G,G"),
    Genre("rock", 120, "E", "minor", "Em,Cmaj7,G/B,D"),
    Genre("jazz", 132, "C", "major", "Cmaj7,A7,Dm7,G7/B"),
    Genre("pop", 116, "G", "major", "Gmaj7,C/E,D,Em7"),
    Genre("cinematic", 80, "C", "major", "Cmaj7,Fmaj7,G/B,Am7"),
]

# Density -> (drum pattern name, bass style, has_lead, lead subdivision in beats).
DENSITIES: dict[str, tuple[str, str, bool, float]] = {
    "sparse": ("halftime", "root", False, 0.0),
    "medium": ("backbeat", "root_fifth", True, 1.0),
    "dense": ("fourfloor", "octave", True, 0.5),
}

ROLE_SFZ = {"drums": "drums.sfz", "bass": "bass.sfz", "chords": "piano.sfz", "lead": "lead.sfz"}


# --------------------------------------------------------------------------- #
# Minimal Standard MIDI File writer (no mido)
# --------------------------------------------------------------------------- #
def _vlq(value: int) -> bytes:
    """Variable-length quantity encoding for MIDI delta times."""
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(out)


def write_smf(notes: list[dict], path: Path, bpm: int, tpqn: int = TPQN) -> None:
    """Write a format-0 Standard MIDI File from note dicts (pitch/start/duration/velocity).

    A tiny local writer so the forge needs no MIDI dependency. Note-on/off pairs are sorted by
    tick (offs before ons at the same tick) and delta-encoded; a tempo meta and an EOT close it.
    """
    events: list[tuple[int, int, bytes]] = []
    for n in notes:
        pitch = int(n["pitch"])
        vel = int(n["velocity"])
        on = int(round(float(n["start_beat"]) * tpqn))
        off = int(round((float(n["start_beat"]) + float(n["duration_beats"])) * tpqn))
        events.append((on, 1, bytes([0x90, pitch & 0x7F, vel & 0x7F])))
        events.append((off, 0, bytes([0x80, pitch & 0x7F, 0])))
    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    mpqn = int(round(60_000_000 / bpm))
    track += b"\x00\xff\x51\x03" + struct.pack(">I", mpqn)[1:]
    last = 0
    for tick, _order, msg in events:
        track += _vlq(tick - last) + msg
        last = tick
    track += b"\x00\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpqn)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    path.write_bytes(header + chunk)


# --------------------------------------------------------------------------- #
# Note builders (through the compose-stack helpers)
# --------------------------------------------------------------------------- #
def _split_slash(token: str) -> tuple[str, int | None]:
    """'G7/B' -> ('G7', 11); 'Cmaj7' -> ('Cmaj7', None). Bass returned as a pitch class."""
    if "/" in token:
        chord_part, bass_name = token.split("/", 1)
        return chord_part.strip(), NOTE_TO_PC[bass_name.strip()]
    return token.strip(), None


def resolve_recipe_chords(progression: str) -> tuple[list[list[int]], list[int | None]]:
    """Voice-led chord pitch-lists + per-chord slash bass pitch classes.

    Slash tails are stripped before voicing (they only recolor the bass), then re-applied to the
    bass lane. Uses ``resolve_progression`` (comma path) + ``voice_lead`` from the compose stack.
    """
    tokens = [t.strip() for t in progression.split(",") if t.strip()]
    chord_syms = [_split_slash(t)[0] for t in tokens]
    slash_pcs = [_split_slash(t)[1] for t in tokens]
    voiced = voice_lead(resolve_progression(",".join(chord_syms), octave=4))
    return voiced, slash_pcs


def build_chord_notes(voiced: list[list[int]]) -> list[dict]:
    """Sustained block chords, one 2-bar block per chord."""
    span = BEATS_PER_BAR * BARS_PER_CHORD
    notes: list[dict] = []
    for i, chord in enumerate(voiced):
        start = i * span
        for pitch in chord:
            notes.append(
                {"pitch": pitch, "start_beat": start, "duration_beats": span, "velocity": 78}
            )
    return notes


def build_bass_notes(
    voiced: list[list[int]], slash_pcs: list[int | None], style: str
) -> list[dict]:
    """Bass line via ``bassline_notes`` in the bass register (36-47), slash bass applied."""
    bass_chords: list[list[int]] = []
    for chord, slash in zip(voiced, slash_pcs, strict=True):
        pc = slash if slash is not None else chord[0] % 12
        bass_chords.append([36 + pc])
    return bassline_notes(bass_chords, style=style, bars_per_chord=BARS_PER_CHORD)


def build_lead_notes(
    voiced: list[list[int]], key_root: str, key_mode: str, subdivision: float
) -> list[dict]:
    """Deterministic scalar arpeggio in the lead octave (chord tones, scale-anchored)."""
    span = BEATS_PER_BAR * BARS_PER_CHORD
    scale = [p % 12 for p in scale_notes(key_root, key_mode, octave=5)]
    notes: list[dict] = []
    for i, chord in enumerate(voiced):
        tones = sorted({p % 12 for p in chord})
        start = i * span
        steps = int(round(span / subdivision))
        for s in range(steps):
            pc = tones[s % len(tones)] if s % 2 == 0 else scale[(s // 2) % len(scale)]
            pitch = 72 + pc  # lead octave (C5-based)
            notes.append(
                {
                    "pitch": pitch,
                    "start_beat": start + s * subdivision,
                    "duration_beats": subdivision * 0.9,
                    "velocity": 70,
                }
            )
    return notes


def build_drum_notes(pattern_name: str) -> list[dict]:
    """Repeat a one-bar drum grid across all bars."""
    one_bar = parse_drum_grid(DRUM_PATTERNS[pattern_name])
    notes: list[dict] = []
    for bar in range(BARS):
        offset = bar * BEATS_PER_BAR
        for n in one_bar:
            notes.append({**n, "start_beat": n["start_beat"] + offset})
    return notes


# --------------------------------------------------------------------------- #
# Asset provisioning
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urlopen(url) as resp, tmp.open("wb") as fh:  # noqa: S310 (pinned fixture asset fetch)
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)


def ensure_sfizz() -> Path:
    """Download + verify the pinned sfizz release; extract sfizz_render.exe. Returns its path."""
    render = ASSETS_DIR / "sfizz" / SFIZZ_RENDER_MEMBER
    if render.exists():
        return render
    zip_path = ASSETS_DIR / f"sfizz-{SFIZZ_VERSION}-win64.zip"
    if not zip_path.exists() or _sha256(zip_path) != SFIZZ_ZIP_SHA256:
        print(f"Downloading sfizz {SFIZZ_VERSION} ...")
        _download(SFIZZ_ZIP_URL, zip_path)
    got = _sha256(zip_path)
    if got != SFIZZ_ZIP_SHA256:
        raise RuntimeError(f"sfizz zip sha256 mismatch: expected {SFIZZ_ZIP_SHA256}, got {got}")
    with zipfile.ZipFile(zip_path) as zf:
        for member in (SFIZZ_RENDER_MEMBER, SFIZZ_DLL_MEMBER):
            zf.extract(member, ASSETS_DIR / "sfizz")
    return render


def ensure_samples() -> Path:
    """Download + verify each pinned VSCO2-CE sample into the shared SFZ dir. Returns that dir."""
    sfz_dir = ASSETS_DIR / "sfz"
    sfz_dir.mkdir(parents=True, exist_ok=True)
    for name, pin in SAMPLE_PINS.items():
        dest = sfz_dir / f"{name}.wav"
        if dest.exists() and _sha256(dest) == pin.sha256:
            continue
        url = f"{VSCO2_RAW_BASE}/{urllib.parse.quote(pin.repo_path)}"
        print(f"Downloading VSCO2-CE sample {name} ...")
        _download(url, dest)
        got = _sha256(dest)
        if got != pin.sha256:
            raise RuntimeError(f"{name} sha256 mismatch: expected {pin.sha256}, got {got}")
    return sfz_dir


def author_sfz(sfz_dir: Path) -> None:
    """Write the four authored SFZ instruments + the deterministic synth drum one-shots."""
    (sfz_dir / "piano.sfz").write_text(
        f"<region> sample=piano_lo.wav pitch_keycenter={SAMPLE_PINS['piano_lo'].keycenter}"
        " lokey=0 hikey=64\n"
        f"<region> sample=piano_hi.wav pitch_keycenter={SAMPLE_PINS['piano_hi'].keycenter}"
        " lokey=65 hikey=127\n",
        encoding="utf-8",
    )
    (sfz_dir / "bass.sfz").write_text(
        f"<region> sample=contrabass.wav pitch_keycenter={SAMPLE_PINS['contrabass'].keycenter}"
        " lokey=0 hikey=127\n",
        encoding="utf-8",
    )
    (sfz_dir / "lead.sfz").write_text(
        f"<region> sample=flute.wav pitch_keycenter={SAMPLE_PINS['flute'].keycenter}"
        " lokey=0 hikey=127\n",
        encoding="utf-8",
    )
    # Drums: repo's seeded synth kit (CC0-by-construction), keyed to GM notes, no repitch.
    ensure_drum_samples(sfz_dir)
    (sfz_dir / "drums.sfz").write_text(
        "<region> key=36 sample=kick.wav loop_mode=one_shot\n"
        "<region> key=38 sample=snare.wav loop_mode=one_shot\n"
        "<region> key=42 sample=hat.wav loop_mode=one_shot\n",
        encoding="utf-8",
    )


@dataclass
class RenderCtx:
    """Everything a render needs: the sfizz binary, the SFZ dir, and a scratch dir."""

    sfizz_render: Path
    sfz_dir: Path
    scratch: Path


def make_render_ctx(scratch: Path) -> RenderCtx:
    sfizz_render = ensure_sfizz()
    sfz_dir = ensure_samples()
    author_sfz(sfz_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    return RenderCtx(sfizz_render=sfizz_render, sfz_dir=sfz_dir, scratch=scratch)


# --------------------------------------------------------------------------- #
# Rendering + mixing
# --------------------------------------------------------------------------- #
def _peak_normalize(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-9:
        return audio
    target = 10.0 ** (target_dbfs / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad a (frames, 2) array to exactly ``n`` frames."""
    if audio.shape[0] >= n:
        return audio[:n]
    pad = np.zeros((n - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=0)


def render_role(role: str, notes: list[dict], bpm: int, ctx: RenderCtx) -> np.ndarray:
    """Render one role's notes through its SFZ; return a normalized (frames, 2) float32 stem."""
    mid = ctx.scratch / f"{role}.mid"
    wav = ctx.scratch / f"{role}.wav"
    write_smf(notes, mid, bpm)
    cmd = [
        str(ctx.sfizz_render),
        "--sfz", str(ctx.sfz_dir / ROLE_SFZ[role]),
        "--midi", str(mid),
        "--wav", str(wav),
        "-s", str(RENDER_SR),
        "--use-eot",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    audio, _sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return _peak_normalize(audio, STEM_DBFS)


def grid_frames(bpm: int) -> int:
    grid_s = BARS * BEATS_PER_BAR * (60.0 / bpm)
    return int(round((grid_s + TAIL_S) * FIXTURE_SR))


# --------------------------------------------------------------------------- #
# SAPI vocal lane (D8) — generated once, committed, never re-rendered
# --------------------------------------------------------------------------- #
def _sapi_line_to_wav(text: str, dest: Path) -> None:
    """Speak one line to a WAV via PowerShell System.Speech (Windows-only)."""
    escaped_text = text.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{dest}'); "
        f"$s.Speak('{escaped_text}'); $s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        check=True, capture_output=True, text=True,
    )


def _resample_stereo(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src_sr != dst_sr:
        n_dst = int(round(len(audio) * dst_sr / src_sr))
        x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        xi = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
        audio = np.interp(xi, x, audio).astype(np.float32)
    return np.stack([audio, audio], axis=1).astype(np.float32)


def build_vocal_stem(bpm: int, ctx: RenderCtx) -> tuple[np.ndarray, list[dict]]:
    """Speak the 6 lyric lines, place them on the bar grid, normalize to -12 dBFS.

    Returns the (frames, 2) stem and per-line placements (text/start_s/end_s) for truth.json.
    """
    n = grid_frames(bpm)
    stem = np.zeros((n, 2), dtype=np.float32)
    total_s = n / FIXTURE_SR
    slot = total_s / len(LYRIC_LINES)
    placements: list[dict] = []
    for i, line in enumerate(LYRIC_LINES):
        raw = ctx.scratch / f"vocal_{i}.wav"
        _sapi_line_to_wav(line, raw)
        audio, sr = sf.read(str(raw), dtype="float32", always_2d=False)
        clip = _resample_stereo(audio, sr, FIXTURE_SR)
        start = int(round(i * slot * FIXTURE_SR))
        end = min(n, start + clip.shape[0])
        stem[start:end] += clip[: end - start]
        placements.append(
            {
                "text": line,
                "start_s": round(start / FIXTURE_SR, 3),
                "end_s": round(end / FIXTURE_SR, 3),
            }
        )
    return _peak_normalize(stem, VOCAL_DBFS), placements


# --------------------------------------------------------------------------- #
# Fixture assembly
# --------------------------------------------------------------------------- #
def _mix_stems(stems: dict[str, np.ndarray]) -> np.ndarray:
    """Sum stems; apply a safety ceiling only if the sum would clip (keeps mix ~= sum of stems)."""
    n = max(s.shape[0] for s in stems.values())
    mix = np.zeros((n, 2), dtype=np.float32)
    for s in stems.values():
        mix += _fit_length(s, n)
    peak = float(np.max(np.abs(mix)))
    if peak > MIX_CEILING:
        mix *= MIX_CEILING / peak
    return mix.astype(np.float32)


def _write_flac(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, FIXTURE_SR, format="FLAC", subtype="PCM_16")


def build_fixture(
    name: str,
    genre: Genre,
    density: str,
    out_dir: Path,
    ctx: RenderCtx,
    *,
    with_vocal: bool = False,
) -> Path:
    """Render one fixture directory: stems/<role>.flac, mix.flac, truth.json. Returns its path."""
    pattern, bass_style, has_lead, subdivision = DENSITIES[density]
    voiced, slash_pcs = resolve_recipe_chords(genre.progression)
    n = grid_frames(genre.bpm)

    role_notes: dict[str, list[dict]] = {
        "drums": build_drum_notes(pattern),
        "bass": build_bass_notes(voiced, slash_pcs, bass_style),
        "chords": build_chord_notes(voiced),
    }
    if has_lead:
        role_notes["lead"] = build_lead_notes(voiced, genre.key_root, genre.key_mode, subdivision)

    fixture_dir = out_dir / name
    stems_dir = fixture_dir / "stems"
    stems: dict[str, np.ndarray] = {}
    for role, notes in role_notes.items():
        stem = _fit_length(render_role(role, notes, genre.bpm, ctx), n)
        stems[role] = stem
        _write_flac(stems_dir / f"{role}.flac", stem)

    truth: dict = {
        "name": name,
        "genre": genre.name,
        "density": density,
        "bpm": genre.bpm,
        "bars": BARS,
        "sample_rate": FIXTURE_SR,
        "key": {"root": genre.key_root, "mode": genre.key_mode},
        "progression": genre.progression,
        "chords": [_split_slash(t)[0] for t in genre.progression.split(",")],
        "roles": {role: f"stems/{role}.flac" for role in role_notes},
    }

    if with_vocal:
        vocal_stem, placements = build_vocal_stem(genre.bpm, ctx)
        stems["vocals"] = vocal_stem
        _write_flac(stems_dir / "vocals.flac", vocal_stem)
        truth["roles"]["vocals"] = "stems/vocals.flac"
        truth["lyrics"] = {"lines": list(LYRIC_LINES), "placements": placements}

    _write_flac(fixture_dir / "mix.flac", _mix_stems(stems))
    (fixture_dir / "truth.json").write_text(
        json.dumps(truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return fixture_dir


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def _iter_fixture_files(fixture_dir: Path) -> list[Path]:
    return sorted(p for p in fixture_dir.rglob("*") if p.is_file())


def build_manifest(fixture_dirs: list[Path], root: Path) -> dict:
    """{files: {relpath: sha256}, fixture_set_hash: sha256(sorted file hashes)} — §15.1 identity."""
    files: dict[str, str] = {}
    for fixture_dir in fixture_dirs:
        for path in _iter_fixture_files(fixture_dir):
            if path.name == "manifest.json":
                continue
            rel = path.relative_to(root).as_posix()
            files[rel] = _sha256(path)
    set_hash = hashlib.sha256("".join(sorted(files.values())).encode("utf-8")).hexdigest()
    return {"files": dict(sorted(files.items())), "fixture_set_hash": set_hash}


def write_manifest(fixture_dirs: list[Path], root: Path) -> Path:
    manifest = build_manifest(fixture_dirs, root)
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CI subset — the committed fixtures
# --------------------------------------------------------------------------- #
CI_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "audio"
# ci_sparse is the instrumental no_vocals fixture. The lead (sustained flute) stem is dropped
# here to hold the committed subset under the 6 MB budget (amendment 5) — drums/bass/chords, the
# 3-track instrumental bed; the desktop set keeps the full 4-track medium/dense recipes.
CI_GENRE = GENRES[6]        # pop
CI_DENSITY = "sparse"       # drums/bass/chords (no lead — budget)


def build_ci_instrumental(ctx: RenderCtx, out_root: Path) -> Path:
    """Build ci_sparse (deterministic instrument stems only)."""
    return build_fixture("ci_sparse", CI_GENRE, CI_DENSITY, out_root, ctx, with_vocal=False)


def build_ci_subset(ctx: RenderCtx) -> Path:
    """Build both committed CI fixtures + their manifest into tests/fixtures/audio/."""
    CI_ROOT.mkdir(parents=True, exist_ok=True)
    sparse = build_ci_instrumental(ctx, CI_ROOT)
    vocal = build_fixture("ci_vocal", CI_GENRE, CI_DENSITY, CI_ROOT, ctx, with_vocal=True)
    write_manifest([sparse, vocal], CI_ROOT)
    return CI_ROOT


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@dataclass
class _Args:
    out: Path | None = None
    ci_only: bool = False
    scratch: Path = field(default_factory=lambda: ASSETS_DIR / "scratch")


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Desktop fixture set output dir.")
    parser.add_argument(
        "--ci-only", action="store_true", help="Build only the 2 committed CI fixtures + manifest."
    )
    ns = parser.parse_args(argv)

    scratch = ASSETS_DIR / "scratch"
    ctx = make_render_ctx(scratch)

    # Always (re)build the committed CI subset — it is the deterministic, testable core.
    ci_root = build_ci_subset(ctx)
    ci_kb = _dir_size_bytes(ci_root) / 1024.0
    print(f"CI subset -> {ci_root}  ({ci_kb:.0f} KB)")

    if ns.ci_only or ns.out is None:
        return

    out = ns.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    fixture_dirs: list[Path] = []
    for genre in GENRES:
        for density in DENSITIES:
            name = f"{genre.name}_{density}"
            print(f"Rendering {name} ...")
            fixture_dirs.append(build_fixture(name, genre, density, out, ctx))
    write_manifest(fixture_dirs, out)
    total_mb = _dir_size_bytes(out) / 1e6
    print(f"Desktop set -> {out}  ({len(fixture_dirs)} fixtures, {total_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
