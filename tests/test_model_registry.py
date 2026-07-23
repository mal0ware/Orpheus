"""Model registry (§4): data seed + loader + resolve()/free_vram_gb() (§15.5)."""

from __future__ import annotations

import pytest

from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.model_registry import (
    GPU_SEMAPHORE,
    ModelEntry,
    RegistryError,
    checkpoint_dir,
    free_vram_gb,
    load_registry,
    resolve,
)

# Every §4 slot the seed JSON must cover, named exactly as Task 0.2's brief lists them.
EXPECTED_IDS = {
    "4stem-a/bsroformer",
    "4stem-b/scnet-xl",
    "4stem-c/htdemucs-ft",
    "vocals/melroformer",
    "6stem/bsroformer-sw",
    "6stem-fallback/htdemucs-6s",
    "guitar/melroformer-becruily",
    "drumsep",
    "dereverb/anvuew",
    "tagger/panns-sed",
    "chord/btc-large-voca",
    "chord/musicxlab-ismir2019",
    "lyrics/faster-whisper-large-v3",
    "transcribe/basic-pitch",
}


def test_seed_registry_covers_every_slot():
    entries = load_registry(allow_unpinned=True)
    assert len(entries) >= 12
    ids = {e.id for e in entries}
    assert ids >= EXPECTED_IDS


def test_dereverb_entry_is_vocal_only():
    entries = {e.id: e for e in load_registry(allow_unpinned=True)}
    dereverb = entries["dereverb/anvuew"]
    assert dereverb.task == "dereverb"
    assert dereverb.targets == ["vocals"]


def test_chord_model_lane_rank_order():
    entries = {e.id: e for e in load_registry(allow_unpinned=True)}
    assert entries["chord/btc-large-voca"].rank == 1
    assert entries["chord/musicxlab-ismir2019"].rank == 2


def test_load_registry_rejects_unpinned_checkpoints_by_default():
    with pytest.raises(RegistryError):
        load_registry(allow_unpinned=False)


def test_load_registry_allows_unpinned_when_asked():
    entries = load_registry(allow_unpinned=True)
    # The seed JSON is genuinely unpinned pre-0.3 — every downloadable entry has sha256=None.
    assert any(e.sha256 is None for e in entries)


def test_resolve_multi_stem_vocals_filters_over_vram_budget():
    # threshold = CFG.VRAM_FIT_FRACTION * 9.0 = 7.2 GB
    results = resolve("multi-stem", "vocals", vram_free_gb=9.0)
    ids = [e.id for e in results]
    assert "4stem-b/scnet-xl" not in ids  # 8.0 GB > 7.2 GB threshold, filtered
    assert "4stem-a/bsroformer" in ids
    assert "4stem-c/htdemucs-ft" in ids
    # rank order preserved among survivors
    assert ids.index("4stem-a/bsroformer") < ids.index("4stem-c/htdemucs-ft")
    for entry in results:
        assert entry.vram_gb <= CFG.VRAM_FIT_FRACTION * 9.0


def test_resolve_multi_stem_vocals_no_cuda_keeps_only_cpu_capable():
    results = resolve("multi-stem", "vocals", vram_free_gb=None)
    ids = {e.id for e in results}
    # Both htdemucs entries are the CPU-friendly family ("runs anywhere" per §4); the
    # RoFormer/SCNet entries all declare a real vram_gb and are excluded.
    assert ids == {"4stem-c/htdemucs-ft", "6stem-fallback/htdemucs-6s"}
    for entry in results:
        assert entry.vram_gb == 0.0


def test_resolve_chord_no_cuda_keeps_internal_baseline():
    results = resolve("chord", "chord", vram_free_gb=None)
    ids = {e.id for e in results}
    assert "chord/internal-baseline" in ids
    assert "chord/btc-large-voca" not in ids
    assert "chord/musicxlab-ismir2019" not in ids


def test_resolve_excludes_nc_licensed_entries_by_default():
    results = resolve("drumkit-split", "kick", vram_free_gb=100.0)
    ids = [e.id for e in results]
    assert "drumkit-split/larsnet" not in ids
    assert "drumsep" in ids


def test_resolve_includes_nc_licensed_entries_when_opted_in():
    results = resolve("drumkit-split", "kick", vram_free_gb=100.0, allow_nc=True)
    ids = [e.id for e in results]
    assert "drumkit-split/larsnet" in ids


def test_free_vram_gb_returns_none_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert free_vram_gb() is None


def test_free_vram_gb_missing_torch_returns_none(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch in this env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert free_vram_gb() is None


def test_empty_tagger_classes_allowed():
    # Synth pad has no clean AudioSet class (§4) — an entry may declare no tagger mapping,
    # meaning bleed/exposure gates for it return N/A rather than computing garbage.
    entry = ModelEntry(
        id="synth-pad/example",
        task="single-target",
        targets=["synth-pad"],
        engine="internal",
        checkpoint_url=None,
        sha256=None,
        vram_gb=0.0,
        rt_factor=None,
        benchmark=None,
        license="MIT",
        rank=1,
        tagger_classes=[],
    )
    assert entry.tagger_classes == []


def test_gpu_semaphore_is_a_single_slot():
    assert GPU_SEMAPHORE._value == 1
    acquired = GPU_SEMAPHORE.acquire(blocking=False)
    assert acquired
    try:
        assert GPU_SEMAPHORE.acquire(blocking=False) is False
    finally:
        GPU_SEMAPHORE.release()


def test_checkpoint_dir_honors_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG.CACHE_DIR_ENV, str(tmp_path))
    root = checkpoint_dir()
    assert root == tmp_path / "checkpoints"

    per_model = checkpoint_dir("4stem-a/bsroformer")
    assert per_model == tmp_path / "checkpoints" / "4stem-a" / "bsroformer"
