"""Task 0.3 CI-safe tests: pin-report parser, fully-pinned registry, url-list schema.

Everything GPU/network-heavy lives in ``scripts/pin_models.py`` and is exercised only on the
desktop. These tests touch no checkpoints and no engines — they validate the committed
artifacts (the registry JSON + the pin report) and the pure parser/schema logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from scripts.pin_models import _TABLE_COLUMNS, SEPARATION_TASKS, parse_pin_report

from orpheus_mcp.analysis.model_registry import ModelEntry, load_registry

_PIN_REPORT = Path("docs/pins/2026-07-model-pins.md")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _entry(**overrides) -> ModelEntry:
    base = dict(
        id="x/y",
        task="single-target",
        targets=["vocals"],
        engine="audio-separator",
        checkpoint_url="https://example.invalid/a.ckpt",
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


# --------------------------------------------------------------------------- parser
def test_parse_pin_report_extracts_rows():
    header = "| " + " | ".join(_TABLE_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _TABLE_COLUMNS) + " |"
    body = (
        "| 4stem-b/scnet-xl | msst | yes | abc123def456 | 3.20 | 12.40 | 0.207 "
        "| synthetic-audio | PINNED | PENDING (human) |"
    )
    text = f"# Title\n\nsome prose\n\n{header}\n{sep}\n{body}\n\nmore prose\n"
    rows = parse_pin_report(text)
    assert len(rows) == 1
    assert rows[0]["id"] == "4stem-b/scnet-xl"
    assert rows[0]["engine"] == "msst"
    assert rows[0]["status"] == "PINNED"
    assert rows[0]["audio"] == "synthetic-audio"
    assert rows[0]["ear_check_notes"] == "PENDING (human)"


def test_parse_pin_report_ignores_non_matching_tables():
    text = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"  # not the pin table header
    assert parse_pin_report(text) == []


# --------------------------------------------------------------------------- url-list schema
def test_model_entry_accepts_single_url_and_hash():
    e = _entry(checkpoint_url="https://h/a.ckpt", sha256="a" * 64)
    assert isinstance(e.checkpoint_url, str)
    assert isinstance(e.sha256, str)


def test_model_entry_accepts_url_list_and_hash_list():
    e = _entry(
        checkpoint_url=["https://h/a.th", "https://h/b.th"],
        sha256=["a" * 64, "b" * 64],
    )
    assert isinstance(e.checkpoint_url, list) and len(e.checkpoint_url) == 2
    assert isinstance(e.sha256, list) and len(e.sha256) == 2


def test_model_entry_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        _entry(checkpoint_url=["https://h/a.th"], sha256="a" * 64)
    with pytest.raises(ValueError, match="same shape"):
        _entry(checkpoint_url="https://h/a.th", sha256=["a" * 64])


def test_model_entry_rejects_length_mismatch():
    with pytest.raises(ValueError, match="match positionally"):
        _entry(
            checkpoint_url=["https://h/a.th", "https://h/b.th"],
            sha256=["a" * 64],
        )


def test_model_entry_unpinned_list_url_is_allowed():
    # A bag-of-models entry may sit unpinned (sha256=None) before the pin run, same as any other.
    e = _entry(checkpoint_url=["https://h/a.th", "https://h/b.th"], sha256=None)
    assert e.sha256 is None


def test_htdemucs_ft_is_a_bag_of_models_in_the_seed():
    entries = {e.id: e for e in load_registry(allow_unpinned=True)}
    ft = entries["4stem-c/htdemucs-ft"]
    assert isinstance(ft.checkpoint_url, list)
    assert len(ft.checkpoint_url) == 4


# --------------------------------------------------------------------------- pin report artifact
@pytest.mark.skipif(not _PIN_REPORT.is_file(), reason="pin report not generated yet (desktop run)")
def test_pin_report_covers_every_separation_slot():
    rows = parse_pin_report(_PIN_REPORT.read_text(encoding="utf-8"))
    reported_ids = {r["id"] for r in rows}
    separation_ids = {
        e.id for e in load_registry(allow_unpinned=True) if e.task in SEPARATION_TASKS
    }
    missing = separation_ids - reported_ids
    assert not missing, f"pin report is missing separation slots: {sorted(missing)}"


# --------------------------------------------------------------------------- real-engine smoke
@pytest.mark.desktop
@pytest.mark.gpu
def test_vendored_scnet_separates_synthetic(tmp_path):
    """SCNet-XL (vendored MSST path) produces 4 finite stems from a synthetic mix on the GPU.

    Deselected in CI (`-m "not desktop and not gpu"`); it needs the pinned SCNet checkpoint and
    CUDA. This is the Task 0.3 Step 1 engine-half proof.
    """
    import numpy as np
    import soundfile as sf

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")

    from scripts.pin_models import _filename_for_url, _urls_of, generate_synthetic_mix

    from orpheus_mcp._vendor.msst.inference import separate_file
    from orpheus_mcp.analysis.model_registry import checkpoint_dir

    entry = {e.id: e for e in load_registry(allow_unpinned=True)}["4stem-b/scnet-xl"]
    ckpt = checkpoint_dir(entry.id) / _filename_for_url(_urls_of(entry)[0], entry.id)
    if not ckpt.is_file():
        pytest.skip("SCNet checkpoint not fetched (run scripts/pin_models.py)")

    mix = tmp_path / "mix.flac"
    generate_synthetic_mix(mix, duration_s=20.0)
    written = separate_file(ckpt, mix, tmp_path / "stems", device="cuda")

    assert set(written) == {"drums", "bass", "other", "vocals"}
    for path in written.values():
        audio, _sr = sf.read(str(path))
        assert np.isfinite(audio).all()


# --------------------------------------------------------------------------- registry hashes
def test_registry_is_fully_pinned():
    """Task 0.3 Step 4: after the pin run every downloadable checkpoint carries a sha256.

    ``load_registry(allow_unpinned=False)`` raises if any entry with a ``checkpoint_url`` still
    has a ``None`` hash — so this passing IS the "zero None hashes" invariant.
    """
    entries = load_registry(allow_unpinned=False)
    assert entries
    downloadable = [e for e in entries if e.checkpoint_url is not None]
    assert downloadable
    assert all(e.sha256 is not None for e in downloadable)


def test_pinned_hashes_are_valid_sha256():
    """Whatever IS pinned must be a real 64-hex digest (str) or list thereof."""
    for e in load_registry(allow_unpinned=True):
        if e.sha256 is None:
            continue
        digests = e.sha256 if isinstance(e.sha256, list) else [e.sha256]
        for d in digests:
            assert _HEX64.match(d), f"{e.id}: bad sha256 {d!r}"
