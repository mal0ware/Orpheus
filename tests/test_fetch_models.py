"""`orpheus-mcp fetch-models` (§1.1/§9): the one network-touching install step.

All CI-safe — the downloader is injected, never the real `urllib.request.urlopen`.
"""

from __future__ import annotations

import hashlib

import pytest

from orpheus_mcp.analysis import fetch_models as fm
from orpheus_mcp.analysis.model_registry import ModelEntry


def _entry(**overrides) -> ModelEntry:
    base = dict(
        id="fake/model",
        task="single-target",
        targets=["vocals"],
        engine="audio-separator",
        checkpoint_url="https://example.invalid/fake-model.ckpt",
        sha256=None,
        vram_gb=1.0,
        rt_factor=None,
        benchmark=None,
        license="MIT",
        rank=1,
        tagger_classes=[],
    )
    base.update(overrides)
    return ModelEntry(**base)


def _stub_downloader(content: bytes):
    written: list[tuple[str, object]] = []

    def _download(url: str, dest) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        written.append((url, dest))

    _download.written = written  # type: ignore[attr-defined]
    return _download


def test_require_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr(fm.shutil, "which", lambda _name: None)
    with pytest.raises(fm.FfmpegMissing, match="winget install ffmpeg"):
        fm.require_ffmpeg()


def test_require_ffmpeg_passes_when_present(monkeypatch):
    monkeypatch.setattr(fm.shutil, "which", lambda _name: "C:/ffmpeg/bin/ffmpeg.exe")
    fm.require_ffmpeg()  # must not raise


def test_fetch_models_requires_ffmpeg_before_any_download(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "ffmpeg_on_path", lambda: False)
    downloader = _stub_downloader(b"content")
    monkeypatch.setattr(fm, "load_registry", lambda **_kw: [_entry()])
    monkeypatch.setattr(fm, "checkpoint_dir", lambda model_id=None: tmp_path / (model_id or ""))

    with pytest.raises(fm.FfmpegMissing):
        fm.fetch_models(downloader=downloader)
    assert downloader.written == []  # never reached the download step


def test_fetch_models_skips_entries_without_checkpoint_url(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "ffmpeg_on_path", lambda: True)
    entries = [_entry(id="internal/thing", checkpoint_url=None, sha256=None)]
    monkeypatch.setattr(fm, "load_registry", lambda **_kw: entries)
    monkeypatch.setattr(fm, "checkpoint_dir", lambda model_id=None: tmp_path / (model_id or ""))
    downloader = _stub_downloader(b"unused")

    results = fm.fetch_models(downloader=downloader)
    assert results == {"internal/thing": "skipped (no checkpoint_url)"}
    assert downloader.written == []


def test_fetch_models_verify_mode_requires_pinned_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "ffmpeg_on_path", lambda: True)
    monkeypatch.setattr(fm, "load_registry", lambda **_kw: [_entry(sha256=None)])
    monkeypatch.setattr(fm, "checkpoint_dir", lambda model_id=None: tmp_path / (model_id or ""))
    downloader = _stub_downloader(b"content")

    with pytest.raises(fm.FetchModelsError, match="not pinned|--pin"):
        fm.fetch_models(pin=False, downloader=downloader)
    assert downloader.written == []  # refuses before downloading an unverifiable checkpoint


def test_fetch_models_verify_mode_success(monkeypatch, tmp_path):
    content = b"totally a checkpoint"
    correct_hash = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(fm, "ffmpeg_on_path", lambda: True)
    monkeypatch.setattr(fm, "load_registry", lambda **_kw: [_entry(sha256=correct_hash)])
    monkeypatch.setattr(fm, "checkpoint_dir", lambda model_id=None: tmp_path / (model_id or ""))

    results = fm.fetch_models(pin=False, downloader=_stub_downloader(content))
    assert results == {"fake/model": "verified"}


def test_fetch_models_verify_mode_mismatch_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "ffmpeg_on_path", lambda: True)
    monkeypatch.setattr(
        fm, "load_registry", lambda **_kw: [_entry(sha256="0" * 64)]
    )
    monkeypatch.setattr(fm, "checkpoint_dir", lambda model_id=None: tmp_path / (model_id or ""))

    with pytest.raises(fm.FetchModelsError, match="mismatch"):
        fm.fetch_models(pin=False, downloader=_stub_downloader(b"wrong content"))


def test_checkpoint_filename_strips_query_string(monkeypatch, tmp_path):
    # A raw Google Drive URL (?id=...&export=download) or an HF repo-root URL (no path
    # segment) must not leak "?"/"&" into a Windows filename, or produce an empty name.
    monkeypatch.setattr(fm, "ffmpeg_on_path", lambda: True)
    entries = [
        _entry(
            id="drumkit-split/larsnet",
            checkpoint_url="https://drive.google.com/uc?id=abc123&export=download",
        )
    ]
    monkeypatch.setattr(fm, "load_registry", lambda **_kw: entries)
    monkeypatch.setattr(fm, "checkpoint_dir", lambda model_id=None: tmp_path / (model_id or ""))

    downloader = _stub_downloader(b"content")
    fm.fetch_models(pin=True, downloader=downloader)

    (url, dest) = downloader.written[0]
    assert "?" not in dest.name
    assert "&" not in dest.name
    assert dest.name  # never empty


def test_fetch_models_pin_mode_records_hash_even_when_unpinned(monkeypatch, tmp_path):
    content = b"checkpoint bytes"
    expected = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(fm, "ffmpeg_on_path", lambda: True)
    monkeypatch.setattr(fm, "load_registry", lambda **_kw: [_entry(sha256=None)])
    monkeypatch.setattr(fm, "checkpoint_dir", lambda model_id=None: tmp_path / (model_id or ""))

    results = fm.fetch_models(pin=True, downloader=_stub_downloader(content))
    assert results == {"fake/model": expected}
