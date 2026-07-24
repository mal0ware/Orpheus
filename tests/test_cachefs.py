"""``analysis/cachefs.py`` (Task 1.2, §9): decode, canonical-PCM hashing, artifact store.

CI-safe: soundfile-native formats (wav/flac) never touch ffmpeg; mp3-dependent tests are
skipped cleanly when ffmpeg is absent from PATH (CI has it installed, so they run there).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orpheus_mcp.analysis import cachefs
from orpheus_mcp.analysis.cachefs import CacheFS, DecodeError, decode, pcm_hash
from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.models import Window

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not on PATH")

FIXTURE_MIX = Path(__file__).resolve().parent / "fixtures" / "audio" / "ci_sparse" / "mix.flac"


# --------------------------------------------------------------------------- #
# Test helpers (brief §Step 1: make_sine, transcode, retag)
# --------------------------------------------------------------------------- #


def make_sine(path: Path, *, freq: float = 440.0, seconds: float = 1.0, sr: int = 22050) -> Path:
    """Write a deterministic mono sine-wave WAV — a minimal, reproducible audio fixture."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sine = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sine, sr, subtype="PCM_16")
    return path


def transcode(src: Path, dst: Path) -> Path:
    """Re-encode ``src`` to ``dst`` via ffmpeg (container/codec change, no other edit)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


def retag(src: Path, dst: Path, *, title: str) -> Path:
    """Copy ``src`` to ``dst`` with only an ID3 title tag changed — audio bytes untouched
    (``-c copy`` is a stream copy: no re-encode, so this isolates tag-only edits)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-metadata", f"title={title}", "-c", "copy", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


@pytest.fixture(scope="session")
def ci_mp3(tmp_path_factory) -> Path:
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not on PATH")
    out_dir = tmp_path_factory.mktemp("ci_mp3")
    dst = out_dir / "mix.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(FIXTURE_MIX), str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


# --------------------------------------------------------------------------- #
# Load-bearing tests — verbatim from the Task 1.2 brief
# --------------------------------------------------------------------------- #


@requires_ffmpeg
def test_same_audio_two_containers_share_hash(tmp_path):
    wav = make_sine(tmp_path / "a.wav")
    flac = transcode(wav, tmp_path / "a.flac")
    assert pcm_hash(str(wav)) == pcm_hash(str(flac))


@requires_ffmpeg
def test_id3_edit_does_not_change_hash(tmp_path, ci_mp3):
    tagged = retag(ci_mp3, tmp_path / "tagged.mp3", title="X")
    assert pcm_hash(str(ci_mp3)) == pcm_hash(str(tagged))


def test_meta_mismatch_is_miss(tmp_path):
    fs = CacheFS(tmp_path)
    h = "ab" * 32
    fs.write_json(h, "recon", {"x": 1}, meta={"schema_version": 1, "registry_version": "r1"})
    assert fs.read_json(h, "recon", meta={"schema_version": 2, "registry_version": "r1"}) is None
    assert fs.read_json(h, "recon", meta={"schema_version": 1, "registry_version": "r1"}) == {
        "x": 1
    }


def test_lru_skips_pinned(tmp_path, monkeypatch):
    # Cap forced tiny (bytes, not GB) so any real write trips eviction.
    monkeypatch.setattr(CFG, "CACHE_SIZE_CAP_GB", 1e-6)
    fs = CacheFS(tmp_path)
    h_pinned = "aa" * 32
    h_evictee = "bb" * 32
    meta = {"schema_version": 1}
    fs.write_json(h_pinned, "recon", {"data": "x" * 5000}, meta=meta)
    fs.pin(h_pinned)
    fs.write_json(h_evictee, "recon", {"data": "y" * 5000}, meta=meta)

    freed = fs.evict_lru()

    assert freed > 0
    assert fs.read_json(h_pinned, "recon", meta=meta) == {"data": "x" * 5000}
    assert fs.read_json(h_evictee, "recon", meta=meta) is None


def test_drm_rejected():
    with pytest.raises(DecodeError, match="(?i)drm"):
        decode("nonexistent_but_never_opened.m4p")


def test_truncated_file_raises(tmp_path):
    wav = make_sine(tmp_path / "full.wav")
    raw = wav.read_bytes()
    truncated = tmp_path / "half.wav"
    truncated.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(DecodeError):
        decode(str(truncated))


def test_memo_prevents_second_decode(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path))
    wav = make_sine(tmp_path / "src" / "a.wav")

    calls = {"n": 0}
    real_decode = cachefs.decode

    def counted(path: str):
        calls["n"] += 1
        return real_decode(path)

    monkeypatch.setattr(cachefs, "decode", counted)

    h1 = cachefs.pcm_hash(str(wav))
    h2 = cachefs.pcm_hash(str(wav))

    assert h1 == h2
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Supplementary coverage (not verbatim in the brief, but exercising the same
# locked decisions from other angles)
# --------------------------------------------------------------------------- #


def test_decode_native_wav_reports_soundfile_decoder(tmp_path):
    wav = make_sine(tmp_path / "a.wav")
    pcm = decode(str(wav))
    assert pcm.decoder == "soundfile"
    assert pcm.sample_rate == 22050
    assert pcm.samples.dtype == np.float32
    assert pcm.samples.ndim == 2
    assert pcm.duration_s == pytest.approx(1.0, abs=0.01)
    assert len(pcm.pcm_hash) == 64  # hex sha256


@requires_ffmpeg
def test_decode_mp3_reports_ffmpeg_decoder(ci_mp3):
    pcm = decode(str(ci_mp3))
    assert pcm.decoder.startswith("ffmpeg ")


def test_ffmpeg_nonzero_exit_raises_decode_error_with_stderr_tail(tmp_path):
    fake_mp3 = tmp_path / "garbage.mp3"
    fake_mp3.write_bytes(b"this is not an mp3 file at all")
    with pytest.raises(DecodeError):
        decode(str(fake_mp3))


def test_unsupported_extension_raises_decode_error(tmp_path):
    junk = tmp_path / "song.xyz"
    junk.write_bytes(b"not audio")
    with pytest.raises(DecodeError):
        decode(str(junk))


def test_missing_file_raises_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        decode(str(tmp_path / "does_not_exist.wav"))


def test_write_json_then_read_json_roundtrip(tmp_path):
    fs = CacheFS(tmp_path)
    h = "cc" * 32
    meta = {"schema_version": 1, "model_id": "m1"}
    path = fs.write_json(h, "timeline", {"hop_s": 1.0}, meta=meta)
    assert path.is_file()
    assert fs.read_json(h, "timeline", meta=meta) == {"hop_s": 1.0}


def test_read_json_miss_when_never_written(tmp_path):
    fs = CacheFS(tmp_path)
    assert fs.read_json("dd" * 32, "recon", meta={"schema_version": 1}) is None


def test_artifact_dir_is_keyed_by_pcm_hash(tmp_path):
    fs = CacheFS(tmp_path)
    h = "ee" * 32
    d = fs.artifact_dir(h)
    assert d == Path(tmp_path) / h
    assert d.is_dir()


def test_stem_dir_layout(tmp_path):
    fs = CacheFS(tmp_path)
    h = "ff" * 32
    region = Window(start_s=1.5, end_s=3.25)
    d = fs.stem_dir(h, "4stem-a/bsroformer", region)
    assert d == Path(tmp_path) / h / "stems" / "4stem-a" / "bsroformer" / "1500-3250"
    assert d.is_dir()


def test_clear_removes_all_artifacts(tmp_path):
    fs = CacheFS(tmp_path)
    h = "11" * 32
    meta = {"schema_version": 1}
    fs.write_json(h, "recon", {"x": 1}, meta=meta)
    fs.clear()
    assert fs.read_json(h, "recon", meta=meta) is None
    assert not any(Path(tmp_path).iterdir())


def test_pin_unpin_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(CFG, "CACHE_SIZE_CAP_GB", 1e-6)
    fs = CacheFS(tmp_path)
    h = "22" * 32
    meta = {"schema_version": 1}
    fs.write_json(h, "recon", {"data": "z" * 5000}, meta=meta)
    fs.pin(h)
    fs.unpin(h)
    freed = fs.evict_lru()
    assert freed > 0
    assert fs.read_json(h, "recon", meta=meta) is None


def test_hash_memo_file_written_under_root(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path))
    wav = make_sine(tmp_path / "src" / "b.wav")
    cachefs.pcm_hash(str(wav))
    memo_path = Path(tmp_path) / "hash_memo.json"
    assert memo_path.is_file()
    memo = json.loads(memo_path.read_text("utf-8"))
    file_hash = hashlib.sha256(wav.read_bytes()).hexdigest()
    assert file_hash in memo


def test_verify_config_hash_changes_when_cal_field_changes(monkeypatch):
    before = cachefs.verify_config_hash()
    monkeypatch.setattr(CFG, "GATE_BLEED_PASS", CFG.GATE_BLEED_PASS + 0.01)
    after = cachefs.verify_config_hash()
    assert before != after


def test_verify_config_hash_ignores_non_cal_field(monkeypatch):
    before = cachefs.verify_config_hash()
    monkeypatch.setattr(CFG, "KEY_WINDOW_BARS", CFG.KEY_WINDOW_BARS + 1)
    after = cachefs.verify_config_hash()
    assert before == after


def test_registry_version_matches_registry_file_hash():
    from orpheus_mcp.analysis.model_registry import registry_path

    expected = hashlib.sha256(registry_path().read_bytes()).hexdigest()
    assert cachefs.registry_version() == expected


# --------------------------------------------------------------------------- #
# checkpoints/ reservation (review fix, Task 12.2 wave 1): clear/LRU must never touch
# model_registry.checkpoint_dir()'s downloaded weights, which live under the SAME cache
# root as the analysis artifacts CacheFS itself owns.
# --------------------------------------------------------------------------- #


def test_entry_dirs_excludes_checkpoints(tmp_path):
    fs = CacheFS(tmp_path)
    h = "33" * 32
    meta = {"schema_version": 1}
    fs.write_json(h, "recon", {"x": 1}, meta=meta)

    ckpt_dir = tmp_path / "checkpoints" / "some-model"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "weights.bin").write_bytes(b"x" * 1000)

    assert [d.name for d in fs.entry_dirs()] == [h]
    assert fs.entry_count() == 1


def test_size_bytes_excludes_checkpoints(tmp_path):
    fs = CacheFS(tmp_path)
    h = "44" * 32
    meta = {"schema_version": 1}
    fs.write_json(h, "recon", {"data": "y" * 100}, meta=meta)
    size_before_checkpoint = fs.size_bytes()

    ckpt_dir = tmp_path / "checkpoints" / "some-model"
    ckpt_dir.mkdir(parents=True)
    # Deliberately much bigger than the analysis entry above -- if size_bytes() ever
    # regresses to counting checkpoints/, this assertion catches it immediately.
    (ckpt_dir / "weights.bin").write_bytes(b"x" * 5_000_000)

    assert fs.size_bytes() == size_before_checkpoint


def test_clear_never_touches_checkpoints(tmp_path):
    fs = CacheFS(tmp_path)
    h = "55" * 32
    meta = {"schema_version": 1}
    fs.write_json(h, "recon", {"x": 1}, meta=meta)

    ckpt_dir = tmp_path / "checkpoints" / "some-model"
    ckpt_dir.mkdir(parents=True)
    weights = ckpt_dir / "weights.bin"
    payload = b"x" * 1000
    weights.write_bytes(payload)

    fs.clear()

    assert fs.read_json(h, "recon", meta=meta) is None  # the analysis entry IS gone
    assert weights.is_file()  # the checkpoint survives, byte-for-byte
    assert weights.read_bytes() == payload


def test_evict_lru_never_evicts_checkpoints(tmp_path, monkeypatch):
    # Cap forced tiny (bytes, not GB) so eviction triggers even though the only entry that
    # could legitimately be evicted (the "66" recon entry) is small.
    monkeypatch.setattr(CFG, "CACHE_SIZE_CAP_GB", 1e-9)
    fs = CacheFS(tmp_path)

    ckpt_dir = tmp_path / "checkpoints" / "some-model"
    ckpt_dir.mkdir(parents=True)
    weights = ckpt_dir / "weights.bin"
    payload = b"x" * 5000
    weights.write_bytes(payload)

    h = "66" * 32
    meta = {"schema_version": 1}
    fs.write_json(h, "recon", {"data": "z" * 5000}, meta=meta)

    freed = fs.evict_lru()

    assert freed > 0  # the "66" entry WAS evicted (proves the cap actually bit)
    assert fs.read_json(h, "recon", meta=meta) is None
    assert weights.is_file()  # but checkpoints/ was never even a candidate
    assert weights.read_bytes() == payload
