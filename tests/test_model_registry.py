"""Model registry (§4): data seed + loader + resolve()/free_vram_gb() (§15.5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orpheus_mcp.analysis.config import CFG
from orpheus_mcp.analysis.model_registry import (
    GPU_SEMAPHORE,
    ModelEntry,
    RegistryError,
    checkpoint_dir,
    classes_for_target,
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


def test_load_registry_rejects_unpinned_checkpoints_by_default(tmp_path, monkeypatch):
    # Post-0.3 the seed is fully pinned, so exercise the rejection on a controlled unpinned entry.
    import json

    from orpheus_mcp.analysis import model_registry as mr

    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "unpinned/thing",
                    "task": "single-target",
                    "targets": ["vocals"],
                    "engine": "audio-separator",
                    "checkpoint_url": "https://h/a.ckpt",
                    "sha256": None,
                    "vram_gb": 1.0,
                    "rt_factor": None,
                    "benchmark": None,
                    "license": "MIT",
                    "rank": 1,
                    "tagger_classes": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mr, "registry_path", lambda: path)
    with pytest.raises(RegistryError):
        load_registry(allow_unpinned=False)
    # allow_unpinned=True tolerates the same registry.
    assert load_registry(allow_unpinned=True)[0].sha256 is None


def test_seed_registry_is_fully_pinned():
    # Task 0.3 pinned every downloadable checkpoint — the strict load now succeeds.
    entries = load_registry(allow_unpinned=False)
    assert all(e.sha256 is not None for e in entries if e.checkpoint_url is not None)


def test_resolve_multi_stem_vocals_filters_over_vram_budget():
    # threshold = CFG.VRAM_FIT_FRACTION * 9.0 = 7.2 GB
    results = resolve("multi-stem", "vocals", vram_free_gb=9.0)
    ids = [e.id for e in results]
    assert "4stem-b/scnet-xl" not in ids  # 8.0 GB > 7.2 GB threshold, filtered
    # 4stem-a/bsroformer fits the VRAM budget (6.5 GB) but carries engine_status "no-engine"
    # (D12: audio-separator cannot load ZFTurbo's ep_17 checkpoint) — resolve() must skip it.
    assert "4stem-a/bsroformer" not in ids
    assert "4stem-c/htdemucs-ft" in ids
    for entry in results:
        assert entry.vram_gb <= CFG.VRAM_FIT_FRACTION * 9.0


def test_resolve_skips_no_engine_entries():
    # 4stem-a/bsroformer ranks 3 (below scnet-xl and htdemucs-ft) AND is marked "no-engine" —
    # both-sides: it must never appear in resolve() results (regardless of VRAM headroom)...
    results = resolve("multi-stem", "vocals", vram_free_gb=100.0)
    ids = [e.id for e in results]
    assert "4stem-a/bsroformer" not in ids
    # ...while ordinary (non-"no-engine") 4-stem entries for the same slot are unaffected, and
    # keep their new rank order (SCNet-XL rank 1, htdemucs-ft rank 2).
    four_stem_ids = [i for i in ids if i.startswith("4stem-")]
    assert four_stem_ids == ["4stem-b/scnet-xl", "4stem-c/htdemucs-ft"]


def test_engine_status_no_engine_entry_still_loads():
    # The registry loader itself must not reject/hide a "no-engine" entry — its pin/hash stay
    # intact (D12: kept for a future vendored BS-RoFormer path); only resolve() routing skips it.
    entries = {e.id: e for e in load_registry(allow_unpinned=True)}
    bsroformer = entries["4stem-a/bsroformer"]
    assert bsroformer.engine_status == "no-engine"
    assert bsroformer.engine_status_note
    assert bsroformer.checkpoint_url is not None
    assert bsroformer.sha256 is not None
    assert bsroformer.rank == 3


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


# --------------------------------------------------------------------------- #
# D13: tagger_classes_by_target (per-target tagger classes for multi-target entries)
# --------------------------------------------------------------------------- #

_MULTI_TARGET_BASE = dict(
    id="4stem-fake/example",
    task="multi-stem",
    targets=["vocals", "drums", "bass", "other"],
    engine="audio-separator",
    checkpoint_url=None,
    sha256=None,
    vram_gb=0.0,
    rt_factor=None,
    benchmark=None,
    license="MIT",
    rank=1,
    tagger_classes=["Singing", "Speech", "Drum kit", "Bass guitar"],
)


def test_multi_target_entry_requires_tagger_classes_by_target():
    # A multi-target entry (len(targets) > 1) with NO tagger_classes_by_target is exactly
    # the pre-D13 bug shape (one flat union list for every target) -- the loader must reject
    # it rather than silently accept a registry entry that would leave a per-target bleed
    # gate blind to cross-target bleed.
    with pytest.raises(ValidationError, match="tagger_classes_by_target"):
        ModelEntry(**_MULTI_TARGET_BASE)


def test_multi_target_entry_rejects_by_target_keys_outside_targets():
    with pytest.raises(ValidationError, match="tagger_classes_by_target has keys not in targets"):
        ModelEntry(
            **_MULTI_TARGET_BASE,
            tagger_classes_by_target={
                "vocals": ["Singing", "Speech"],
                "drums": ["Drum kit"],
                "bass": ["Bass guitar"],
                "other": [],
                "guitar": ["Guitar"],  # not in targets -- rejected
            },
        )


def test_multi_target_entry_accepts_valid_by_target():
    entry = ModelEntry(
        **_MULTI_TARGET_BASE,
        tagger_classes_by_target={
            "vocals": ["Singing", "Speech"],
            "drums": ["Drum kit"],
            "bass": ["Bass guitar"],
            "other": [],
        },
    )
    assert entry.tagger_classes_by_target is not None
    assert entry.tagger_classes_by_target["other"] == []


def test_multi_target_entry_by_target_keys_may_be_a_strict_subset_of_targets():
    # "keys subset of targets", not "keys == targets" -- an entry may leave some targets
    # uncovered (classes_for_target falls back to the flat union for those).
    entry = ModelEntry(
        **_MULTI_TARGET_BASE,
        tagger_classes_by_target={"vocals": ["Singing", "Speech"]},
    )
    assert set(entry.tagger_classes_by_target) <= set(entry.targets)


def test_single_target_entry_does_not_require_tagger_classes_by_target():
    # Only one target -- the flat union list already IS the per-target list, nothing to
    # disambiguate, so the loader must not demand tagger_classes_by_target here.
    entry = ModelEntry(
        id="vocals-fake/example",
        task="single-target",
        targets=["vocals"],
        engine="audio-separator",
        checkpoint_url=None,
        sha256=None,
        vram_gb=0.0,
        rt_factor=None,
        benchmark=None,
        license="MIT",
        rank=1,
        tagger_classes=["Singing", "Speech"],
    )
    assert entry.tagger_classes_by_target is None


def test_classes_for_target_uses_by_target_when_present():
    entry = ModelEntry(
        **_MULTI_TARGET_BASE,
        tagger_classes_by_target={
            "vocals": ["Singing", "Speech"],
            "drums": ["Drum kit"],
            "bass": ["Bass guitar"],
            "other": [],
        },
    )
    assert classes_for_target(entry, "drums") == ["Drum kit"]
    assert classes_for_target(entry, "vocals") == ["Singing", "Speech"]
    # "other" is present with an explicit empty list -- that IS the honest N/A answer, not a
    # signal to fall back to the flat union (which would silently reintroduce the D13 bug).
    assert classes_for_target(entry, "other") == []


def test_classes_for_target_falls_back_to_flat_list_when_target_key_absent():
    entry = ModelEntry(
        **_MULTI_TARGET_BASE,
        tagger_classes_by_target={"vocals": ["Singing", "Speech"]},
    )
    # "drums" has no by-target entry at all (not even an empty one) -- falls back to the
    # flat union rather than silently returning [] as if it were a confirmed N/A.
    assert classes_for_target(entry, "drums") == entry.tagger_classes


def test_classes_for_target_falls_back_to_flat_list_for_single_target_entry():
    entry = ModelEntry(
        id="vocals-fake/example",
        task="single-target",
        targets=["vocals"],
        engine="audio-separator",
        checkpoint_url=None,
        sha256=None,
        vram_gb=0.0,
        rt_factor=None,
        benchmark=None,
        license="MIT",
        rank=1,
        tagger_classes=["Singing", "Speech"],
    )
    assert classes_for_target(entry, "vocals") == ["Singing", "Speech"]


def test_seed_registry_multi_stem_entries_carry_by_target_classes():
    # The real seed data (D13 fix): every multi-target entry in data/model_registry.json
    # must carry tagger_classes_by_target -- this is exactly what the loader's own
    # validator (see above) mechanically enforces just by parsing the file successfully,
    # but this test also checks the specific per-family split rather than just "not None".
    entries = {e.id: e for e in load_registry(allow_unpinned=True)}
    for four_stem_id in ("4stem-a/bsroformer", "4stem-b/scnet-xl", "4stem-c/htdemucs-ft"):
        by_target = entries[four_stem_id].tagger_classes_by_target
        assert by_target is not None
        assert by_target["vocals"] == ["Singing", "Speech"]
        assert by_target["bass"] == ["Bass guitar"]
        assert "Singing" not in by_target["drums"]  # the D13 bleed case
        assert by_target["other"] == []  # honest N/A -- no clean AudioSet class

    drumsep_by_target = entries["drumsep"].tagger_classes_by_target
    assert drumsep_by_target is not None
    assert drumsep_by_target["kick"] == ["Bass drum"]
    assert drumsep_by_target["snare"] == ["Snare drum"]
    assert "Snare drum" not in drumsep_by_target["kick"]  # the D13 bleed case, drum-component-scale
