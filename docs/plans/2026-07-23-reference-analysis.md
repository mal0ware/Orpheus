# Reference Analysis Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Normative companion:** `docs/specs/2026-07-22-reference-analysis-design.md` (the spec). Every task cites the spec sections it implements. Where this plan and the spec disagree, the plan's **Deviations Log** (below) is the only place a disagreement is allowed to exist; anything else is a plan bug.

**Goal:** Implement the spec's full local-only reference-analysis pipeline — recon → timeline/chords → routed separation → verified stems → descriptors → FX intents → stock-first apply — as MCP tools in the new `reference` category, plus the un-deferred FX verbs and `render_track_region` bridge work.

**Architecture:** A staged, cache-backed pipeline in `src/orpheus_mcp/analysis/` (pure functions + engine adapters, no REAPER), driven by a deterministic router/ladder. Heavy models live behind registry-declared adapters (`python-audio-separator` + vendored MSST/BTC inference); everything model-shaped is stubbed in CI and desktop-verified behind pytest markers. Bridge work (render + FX verbs) follows the established Lua-handler + `FakeReaperBridge` + lupa pattern.

**Tech Stack:** Python 3.11+ (FastMCP, Pydantic, numpy, librosa, soundfile, pyloudnorm, music21), torch (CPU default / cu124 via `[separation]`), python-audio-separator, vendored MSST + BTC, faster-whisper + whisperX, basic-pitch (isolated ≤3.11 uv env), Lua 5.4 ReaScript, pytest + lupa, ffmpeg (system prerequisite).

## Global Constraints

- **Local only** (§1.1): no tool ever performs network I/O at call time. The only network code paths are `orpheus-mcp fetch-models` and the existing consent-gated `install_sound_pack`.
- **Free/open only** (§1.2): every model/library pip- or checkpoint-downloadable without auth. License recorded per registry entry; NC-licensed entries (LarsNet) carry `"license": "CC BY-NC 4.0"` and are excluded from `default` routing unless the user opts in.
- **Owned audio only** (§1.3): input is a local file path. No fetching, no scraping. DRM files rejected (§9 input contract).
- **Transcription is last resort** (§1.4, §6 R6): never fires automatically; requires `max_tier="T5"` or an explicit notes ask.
- **Claude stays the brain** (§1.5): no LLM inside Orpheus; `question_focus` is an enum Claude sets, never NLP inside the server.
- **Hardware** (§1.6): RTX 3080/10 GB is the reference box; every stage has a CPU fallback; degradation is always named in tool meta (`capped_by_missing_extra`, VRAM fallbacks per §15.5).
- **Every tunable lives in `analysis/config.py`** (§16): constants transcribed verbatim from the §16 table; nothing hard-coded at call sites. Constants this plan adds beyond §16 are listed in Task 1.1 and flagged `# plan-added`.
- **Cache** (§9): `~/.orpheus_cache/<pcm-sha256>/`, sidecar `meta.json` per artifact, field-for-field compare on read, LRU with pinning, 20 GB cap.
- **Registry honesty rule** (house): tools register in `default`/`full` only once implemented; stubs only in `full` with `[NOT IMPLEMENTED]` docstrings.
- **Python floor 3.11**; `from __future__ import annotations` in every new module; ruff (`E,F,I,UP,B,SIM`, line 100) + mypy clean at every commit.
- **Test commands** run from repo root: `uv run pytest -q`, `uv run python scripts/run_lua_tests.py`. New markers: `desktop` (needs real checkpoints/GPU box), `gpu` (needs CUDA). CI runs `-m "not desktop and not gpu"`.
- **Commits:** conventional messages, one per task minimum, no attribution trailers. All work on branch `feat/reference-analysis` (worktree per `superpowers:using-git-worktrees` at execution time); merge to `main` only at phase boundaries with the full gate green.
- **Windows is the dev/target box** (paths via `pathlib`, subprocess without shell, cp1252-safe console output — reuse the `install.py` degradation pattern).

## Deviations Log (plan vs spec — all deliberate, argue here or nowhere)

| # | Spec says | Plan does | Why |
|---|---|---|---|
| D1 | §12.2: all four empirical pins before planning/coding | Pin (a) engines = Phase 0. Pin (d) SED-head = end of Phase 2. Pin (b) chords = end of Phase 3/9. Pin (c) lyrics = end of Phase 8 | Pins b–d require slice deliverables (fixtures, §15.6 script, baseline lane, lyrics pipeline) — running them first is impossible as written |
| D2 | §12.3 slice 4 bar says "gated stems" but gates are slice 5 | Gates 1–3 (`verify.py` core) built in Phase 4; ladder + agreement gate + calibration in Phase 5 | Slice 4's own acceptance bar needs gates; gates 1–3 have no second-model dependency |
| D3 | §3 module list has no home for the tagger or intent engine | `analysis/timeline.py` (PANNs + exposure/absent helpers), `analysis/intents.py` (rule engine) | §3's list is a minimum, not exhaustive; these need single-responsibility homes |
| D4 | §10 "render fixtures with the compose stack" (implies REAPER) | Fixtures rendered headlessly: compose stack → MIDI → `sfizz_render` CLI (pinned download) + VSCO2-CE/CC0 samples | Reproducible without a live REAPER; CI-safe subset committable |
| D5 | §8 step 3 FX-loop max-iterations/plateau guard (who enforces?) | Constants exported in `get_fx_params`/`compare_character` docstrings + `analyze_reference` notes; enforcement is Claude-side by design | The loop is Claude-driven tool calls; Orpheus has no session state to count iterations in. Honest, documented boundary |
| D6 | §4 registry slots named by model family | Phase 0 pin run records *actual* resolved checkpoint ids/urls/hashes; any slot that 404s or fails ear-check gets its alternate recorded in `docs/pins/2026-07-model-pins.md` and the registry JSON — data change, no code change | Spec's own design intent |
| D7 | §15.1 "freeze thresholds in verify.py config" | Calibration emits `data/calibrated_thresholds.json` (+ fixture_set_hash); `config.py` loads it as overrides at import. `verify_config_hash` (§9 cache meta) is computed AFTER overrides apply, so a re-calibration correctly busts stale gate artifacts | One home for tunables (§16) + data-not-code updates |
| D8 | §10/§12 slice-8 bar "self-recorded or synthesized vocal" fixture | `ci_vocal`'s vocal lane = Windows SAPI TTS reading known lyric lines (script-generated once, committed as FLAC + truth lyrics in `truth.json`) — deterministic, real words, Whisper-transcribable, rights-clean | Makes the 5/5-quotes bar runnable by machine on any box |
| D9 (exec-discovered) | §4: `python-audio-separator` as an in-env inference engine | audio-separator runs in an **isolated uv env** (same pattern as basic-pitch/§9) invoked via its CLI with `--model_file_dir` = our checkpoint dir; it is NOT in the `separation` extra. `AudioSeparatorEngine` (Phase 4) is a subprocess adapter. Also: `fastmcp` floor raised to `>=3.4` (the version the shipped code targets) | Discovered at Task 0.1: every audio-separator release (0.30–0.44.5) transitively caps fastmcp<3.4 — one env cannot hold both the MCP server and audio-separator. Vendored MSST + ASR stack coexist fine and stay in `[separation]` |
| D10 (exec-discovered) | Plan conflated two confidences into BeatGrid.confidence | BeatGrid gains additive meter_confidence (margin ratio of the {3,4}-group argmax); confidence stays grid-alignment (feeds BEAT_GRID_CONFIDENCE_MIN rubato fallback) | Review of Task 1.3 found the §T1 margin-ratio formula dropped; two signals, two fields |
| D11 (exec-discovered) | §13/§12 slice-3 bar: "baseline ≥90% majmin on fixtures" | Bar enforced on clean synthesized progressions (100%) AND the fixture's rendered chords STEM (≥90%, beat-level). On the dense FULL-MIX render the template baseline reaches ~0.20 (oracle template ceiling ~0.70) — full-mix complex-chord duty is the §13.1 model lane's (Phase 9); two plan-added tunables (CHORD_EMISSION_SHARPNESS, energy-gated N) were required to make Viterbi usable at all | Template matching on dense renders is chroma-smeared; stem-exposed harmonic content is the honest fixture bar |
| D12 (exec-discovered) | §4: BS-RoFormer ep17 "runs in audio-separator" | False in practice — audio-separator refuses ZFTurbo ep_17 checkpoints; ep17 marked engine_status:"no-engine" (rank 3, hash kept). 4-stem ranks: SCNet-XL (msst, rank 1, native-SR resample added) > htdemucs-ft (audio-separator, rank 2). §5.4's cross-architecture agreement pair = msst/SCNet vs audio-separator/htdemucs — independence preserved. Vendoring the BS-RoFormer model def into _vendor/msst is a post-v1 registry TODO | Pin-run evidence; ladder needs working rank-1 |
| D13 (exec-discovered) | §4 registry: one tagger_classes list per entry | Multi-target entries carry tagger_classes_by_target (per-stem §5.2/§5.3/§3.1 checks use classes_for_target); flat list stays as the union for timeline's mapped subset | Union classes made vocal bleed into a drums stem read as "target" — bleed gate blind on multi-stem entries |
| D14 (exec-discovered) | §9 verify_stem contract: {gates: {...}, overall} nested shape | Shipped shape is the flat GateSet model ({mix_consistency, bleed, silence_honesty, agreement, overall}) — used consistently by separate_region AND verify_stem since Phase 4 | One shared pydantic model beats two wire shapes; flat shape predates 5.3 and is uniform across all §9 gate surfaces |
| D15 (exec-discovered) | Plan 12.1: tier ceiling = literal min(QUESTION_TIERS[class], max_tier) | `_class_ceiling` bumps sound_at/compare_track to T4 (QUESTION_TIERS keeps the T3 nominal; `tier_reached` reports the honest rung) — §3.1's own "+T4 only on gate failure" and §6's R1–R5-are-T4 make the ladder itself enforce the semantic; the ceiling merely permits the climb | A literal min() would make every escalation rung (incl. the carried R3-complement duty) unreachable for the exact question classes escalation exists for |
| D16 (exec-discovered) | §9 meta sidecar: one canonical shape {model_id, checkpoint_sha256, registry_version, verify_config_hash, schema_version, budget_cap?} | Shipped metas share registry_version/verify_config_hash/schema_version but the model-identity field is stage-named (tagger_id/produced_by/model_id/asr_model_id) and budget_cap lives in the ladder payload, not meta | Field-for-field compare makes shape-per-stage safe (any mismatch = miss); stage-named identity fields are self-documenting; recorded per the Log-or-nowhere rule |
| D17 (user-requested) | §9 input contract: mp3/wav/flac/m4a(AAC)/ogg/aiff | `.mp4` (music videos) added to the ffmpeg decode set — same AAC codec as m4a, same `-vn` extraction path, DRM gate unchanged (`.m4p` still rejected; `.m4v` deliberately excluded — frequently DRM'd) | The user's owned reference songs include music-video files; the decoder already handled the container |

---

## Module Map (created/modified — lock these names)

**Created — `src/orpheus_mcp/analysis/` (pure/adapters, no REAPER):**
- `config.py` — every §16 constant + plan-added constants; `dataclass AnalysisConfig` + module-level `CFG` singleton (tests monkeypatch fields).
- `cachefs.py` — decode + hashing + artifact store: `DecodeError`, `decode(path) -> Pcm`, `pcm_hash(path) -> str`, `class CacheFS` (`artifact_path`, `read_json`, `write_json`, `read_stem`, `write_stem`, `pin`, `unpin`, `evict_lru`, `clear`, `size_bytes`).
- `recon.py` — `recon(pcm: Pcm) -> Recon` (T1: LUFS, tempo/beats/downbeats/meter + grid confidence, tuning offset, key + alternates, sections, band sketch).
- `timeline.py` — `class TaggerEngine(Protocol)`, `PannsTagger`, `build_timeline(pcm, tagger) -> Timeline`, `most_exposed_window(tl, classes, min_len_s) -> Window | None`, `target_absent(tl, classes) -> bool`.
- `harmony.py` — `chord_lane_baseline(pcm, recon) -> ChordsLane`, `refine_extensions(lane, chroma) -> ChordsLane`, `detect_slash(lane, bass_f0) -> ChordsLane`, `theory_layer(lane) -> ChordsLane` (keys/roman via music21), `fixed_grid(pcm) -> BeatGrid`.
- `model_registry.py` — `class ModelEntry(BaseModel)` (§4 fields), `load_registry() -> Registry`, `resolve(task, target, *, vram_free_gb) -> list[ModelEntry]` (§15.5 filter), `GPU_SEMAPHORE` (process-wide, 1 slot), `free_vram_gb() -> float | None`.
- `separation.py` — `class SeparatorEngine(Protocol)` (`separate(wav_path, entry, out_dir) -> dict[target, path]`), `AudioSeparatorEngine`, `MSSTEngine` (vendored SCNet path), `separate_region(pcm, entry, region, engine=None) -> dict[str, StemResult]` (crop ±5 s, separate, trim back, align/resample; the TOOL returns the requested target's single `StemResult`, all produced stems are cached).
- `verify.py` — `gate_mix_consistency`, `gate_bleed`, `gate_silence_honesty`, `gate_agreement`, `run_gates(stem, mix, timeline, entry, *, other_stem=None) -> GateSet`.
- `router.py` — `QUESTION_TIERS` table, `route(question_class) -> Tier`, `class Ladder` (R1–R6 state machine, budget, resume-from-rung, log), standing rules a–d as pure helpers.
- `descriptors.py` — `compute_character(source: AudioSource) -> SoundCharacter` (§7 full schema).
- `lyrics.py` — `vad_gate(vocal_pcm) -> list[Window]`, `transcribe(vocal_pcm) -> LyricsResult`, `assemble_lines/sections`, `locate_lyric(lyrics, quote, occurrence=None) -> list[LocateMatch]`.
- `intents.py` — `diff_characters(a, b) -> CharacterDiff` (band-limited per §8; bandwidth is MEASURED in `descriptors.py` and read from `SoundCharacter.source.bandwidth_hz` — one home, no duplicate detector), `fire_rules(diff) -> list[Intent]` (loads `data/intent_rules.json`), `match_inventory(intents, inventory) -> None` (annotates each Intent in place — §8 step 2.1).

**Created — vendored inference (pinned commits, upstream LICENSE files included):**
- `src/orpheus_mcp/_vendor/msst/` (SCNet inference path, Phase 0) · `src/orpheus_mcp/_vendor/btc/` (BTC chord inference, Phase 9).

**Created — data (shipped via existing hatch force-include):**
- `data/model_registry.json` · `data/intent_rules.json` (14 rules, §15.3 verbatim) · `data/vst_catalog.json` (§8.1 verbatim) · `data/fx_param_maps.json` (ReaPlugs + selected JSFX).

**Created — tools & scripts:**
- `src/orpheus_mcp/tools/reference.py` — all §9 tools; registered as category `"reference"` in `registry.py`, added to `default`/`full`.
- `scripts/pin_models.py` · `scripts/make_fixtures.py` · `scripts/pick_sed_head.py` · `scripts/calibrate_gates.py`.
- `fetch-models` subcommand (in `server.py` CLI dispatch, next to `install-bridge`).

**Modified:**
- `src/orpheus_mcp/tools/mix.py` — un-stub `add_fx_by_name`, `set_fx_param`, `get_fx_params` (Phase 11).
- `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua` — `render_track_region` (+ render mutex) and the three FX handlers.
- `src/orpheus_mcp/models.py` — new pydantic models (list under Interfaces below).
- `src/orpheus_mcp/registry.py` — `"reference"` category in `default`/`full` only (§3; `explain` unchanged — matches Task 1.3).
- `tests/fake_reaper.py` — FX param state + render handler on the fake.
- `pyproject.toml` — extras rework (Task 0.1), pytest markers, torch index config.
- `.github/workflows/ci.yml` — `analysis-cpu` job.
- `README.md`, `docs/roadmap.md`, `docs/dev-log.md` — status advance (Phase 12).

**Interfaces other tasks rely on (verbatim — models.py additions):**

```python
class Window(BaseModel):      start_s: float; end_s: float

# NOT in models.py — `Pcm` is a plain @dataclass in analysis/cachefs.py (ndarray payload;
# never crosses a tool boundary): Pcm(samples: np.ndarray float32 (frames, ch),
# sample_rate: int, path: str, pcm_hash: str, duration_s: float)
class BeatGrid(BaseModel):    times_s: list[float]; downbeats_s: list[float]
                              meter: tuple[int, int]; confidence: float; meter_confidence: float
                              kind: Literal["beat","fixed"]
class KeyEstimate(BaseModel): root: str; mode: str; confidence: float
class Recon(BaseModel):
    schema_version: int = 1
    duration_s: float; sample_rate: int; lufs_integrated: float | None
    tempo_bpm: float; grid: BeatGrid; tuning_offset_cents: float
    key: KeyEstimate; key_alternates: list[KeyEstimate]
    sections: list[Window]; band_energy_db: dict[str, float]
class Timeline(BaseModel):
    schema_version: int = 1
    classes: list[str]; hop_s: float; frames: list[list[float]]
    # raw activations, no thresholds baked. hop_s is LOCKED to 1.0: build_timeline
    # max-pools the tagger's native (~10 ms) frames to 1 s — a 4-min song is ~240×|mapped|
    # floats, tool-result-sized. Gate math consumes the same 1 s frames.
class ChordSegment(BaseModel):
    start_beat: float | None; end_beat: float | None; start_s: float; end_s: float
    root: str; bass: str | None; quality: str; extensions: list[str]
    label: str; roman: str | None; key_context: str | None
    confidence: float; produced_by: str
class ChordsLane(BaseModel):  schema_version: int = 1; grid: Literal["beat","fixed"]; segments: list[ChordSegment]
class GateResult(BaseModel):  verdict: Literal["pass","fail","na"]; value: float | None
class GateSet(BaseModel):
    mix_consistency: GateResult; bleed: GateResult
    silence_honesty: GateResult; agreement: GateResult
    overall: Literal["pass","fail","unverifiable"]
class StemResult(BaseModel):
    stem_path: str; model_id: str; target: str
    region_trimmed: Window; sample_rate: int; gates: GateSet | None = None
class StemHandle(BaseModel):  file_pcm_hash: str; model_id: str; region: Window
class SourceMeta(BaseModel):
    stem_type: str            # "vocals"|"guitar"|...|"mix-region"|"complement"|"render"
    produced_by: str; region: Window; sample_rate: int
    bandwidth_hz: float | None = None   # None until descriptors measures + stamps it (Phase 6)
    channels: int; complement: bool = False
class SoundCharacter(BaseModel):
    schema_version: int = 1; source: SourceMeta
    spectral: dict[str, float | None]; dynamics: dict[str, float | None]
    harmonics: dict[str, float | str | None]; stereo: dict[str, float | None]
    time_fx: dict[str, float | str | bool | None]; pitch: dict[str, float | bool | None] | None
class Intent(BaseModel):
    rule_id: int; intent: str; params: dict[str, float | str]
    apply_path: dict[str, str | int | float] | None   # None => recommend-only
    explanation: str; vocal_only_skipped: bool = False
    installed_matches: list[str] = []   # §8 step 2.1 — user's own plugins serving this intent (name-matched)
class CharacterDiff(BaseModel):
    diffs: dict[str, dict[str, float | str | None]]  # descriptor -> {a, b, delta, unit}  # unit added per §9 (review fix)
    intents: list[Intent]; band_limit_hz: float; notes: list[str]
class LyricWord(BaseModel):   word: str; start_s: float; end_s: float; conf: float
class LyricLine(BaseModel):   text: str; start_s: float; end_s: float; word_idx_range: tuple[int, int]
class LyricSection(BaseModel): name: str; index: int; start_s: float; end_s: float
                               kind: Literal["chorus","verse","section"]
class LyricsResult(BaseModel):
    schema_version: int = 1; status: Literal["ok","no_vocals"]; language: str | None
    words: list[LyricWord]; lines: list[LyricLine]; sections: list[LyricSection]
class LocateMatch(BaseModel): start_s: float; end_s: float; matched_text: str
                              score: float; section: str | None; occurrence: int
class AnswerEnvelope(BaseModel):
    schema_version: int = 1
    question_class: str; tier_reached: str; recon_summary: dict
    artifacts: dict            # keys: timeline?, chords?, sections?, stems?, descriptors?
    gates: dict[str, GateSet]; ladder_log: list[str]
    budget_capped: bool = False; capped_by_missing_extra: str | None = None
    notes: list[str]
```

`question_class` values (locked, used by router + envelope + tests):
`"recon" | "timeline" | "chords" | "sound_at" | "compare_track" | "lyric_navigate" | "notes"`.

---

## Phase overview & dependency order

| Phase | Slice (§12.3) | Delivers | Hard deps |
|---|---|---|---|
| 0 | pre-slice | extras/packaging, registry data+loader+`fetch-models`, MSST vendored, **pin (a)**, fixture forge | — |
| 1 | 1 | `config`, `cachefs`, `recon`, `recon_reference` tool | 0 (CI fixtures) |
| 2 | 2 | `timeline.py`, `instrument_timeline` tool, §15.6 script, **pin (d)** | 1 |
| 3 | 3 | `harmony.py` baseline+theory, `get_chords` tool, **pin (b) part 1** | 1 |
| 4 | 4 | engine adapters, `separate_region` + gates 1–3 | 0, 2 |
| 5 | 5 | ladder + agreement gate + ensemble, `verify_stem` tool, calibration script + CI | 4 |
| 6 | 6 | `descriptors.py`, `characterize` tool | 4 (stems), 1 |
| 7 | 7 | `render_track_region` bridge verb + tool | none (bridge-only) |
| 8 | 8 | `lyrics.py`, `transcribe_lyrics` + `locate_lyric` tools, **pin (c)** | 4, 5 (vocal stem via ladder) |
| 9 | 9 | BTC + rank-2 chord lane, bass refinement, **pin (b) part 2** | 3, 4 |
| 10 | 10 | `intents.py`, `compare_character` tool + 14 rules data | 6 |
| 11 | 11 | FX bridge verbs, param maps, apply-loop live smoke | 7, 10 |
| 12 | 12 | `analyze_reference` orchestrator, `suggest_free_vst`, `clear_analysis_cache`, docs | all |

Phases 3, 6→10, and 7 are parallelizable off the critical path (0→1→2→4→5→12).

---

## Phase 0 — Packaging, registry, engines, pin (a), fixture forge

### Task 0.1: Extras rework + markers + torch index plumbing

**Files:** Modify `pyproject.toml`; Modify `.github/workflows/ci.yml`; Test: `tests/test_packaging.py` (new).

**Interfaces produces:** extras `analysis` (CPU torch stack), `separation` (CUDA stack), `transcribe` (unchanged); pytest markers `desktop`, `gpu`.

- [ ] **Step 1: failing test** — `tests/test_packaging.py`:

```python
"""Packaging invariants — extras exist, markers registered, no heavy import at core import time."""
from __future__ import annotations
import subprocess, sys, tomllib
from pathlib import Path

def _pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text("utf-8"))

def test_extras_declared():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert {"analysis", "separation", "transcribe", "dev"} <= set(extras)
    assert any(d.startswith("librosa") for d in extras["analysis"])
    assert any(d.startswith("torch") for d in extras["analysis"])
    assert any(d.startswith("audio-separator") for d in extras["separation"])
    assert any(d.startswith("faster-whisper") for d in extras["separation"])

def test_markers_registered():
    ini = _pyproject()["tool"]["pytest"]["ini_options"]
    assert any(m.startswith("desktop") for m in ini["markers"])
    assert any(m.startswith("gpu") for m in ini["markers"])

def test_core_import_stays_lean():
    # subprocess, not importlib.reload — reload of a package with submodules is unreliable
    out = subprocess.run(
        [sys.executable, "-c",
         "import orpheus_mcp, sys;"
         "bad=[m for m in ('torch','librosa','audio_separator','faster_whisper') if m in sys.modules];"
         "sys.exit(1 if bad else 0)"],
        capture_output=True)
    assert out.returncode == 0  # analysis extras load lazily, inside functions
```

- [ ] **Step 2:** `uv run pytest tests/test_packaging.py -q` → FAIL (no markers/extras).
- [ ] **Step 3:** edit `pyproject.toml`:

```toml
[project.optional-dependencies]
analysis = [
    "librosa>=0.10",
    "torch>=2.4",              # CPU wheels from PyPI on Windows; [separation] overrides index
    "panns-inference>=0.1.1",
]
separation = [
    "orpheus-mcp[analysis]",
    "torch>=2.4", "torchaudio>=2.4",
    "audio-separator>=0.30",
    "faster-whisper>=1.0",
    "whisperx>=3.1",           # NOTE (§14.1): the lib may pull pyannote as a dep — acceptable;
                               # what must NEVER happen is calling diarization or fetching its
                               # HF-gated models. Pin run 0.3 verifies install needs no account.
    "stable-ts>=2.17",         # pinned fallback aligner (§14.1)
    "einops>=0.8", "omegaconf>=2.3",   # vendored MSST runtime deps (trim after 0.3 pin run)
]
transcribe = ["basic-pitch>=0.3"]      # NOTE: installed only in the isolated ≤3.11 env (Task 0.5)

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch      = [{ index = "pytorch-cu124", extra = "separation" }]
torchaudio = [{ index = "pytorch-cu124", extra = "separation" }]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "desktop: needs pinned checkpoints / the desktop box (deselect in CI)",
    "gpu: needs CUDA",
]
```

CI: add job `analysis-cpu` — `sudo apt-get install -y ffmpeg`, `uv sync --extra analysis --extra dev`, then `uv run pytest -q -m "not desktop and not gpu"`; keep the existing lean job as-is. **CI invariant: no checkpoint is ever downloaded in CI** — every real-model test is `desktop`-marked; CI exercises stubs only.
- [ ] **Step 4:** `uv run pytest tests/test_packaging.py -q` → PASS. `uv sync --extra analysis --extra dev` resolves (records torch-cpu in `uv.lock`).
- [ ] **Step 5:** Commit `build: analysis/separation extras with per-extra torch index; desktop/gpu markers`.

### Task 0.2: Model registry — data seed + loader + `fetch-models`

**Files:** Create `data/model_registry.json`, `src/orpheus_mcp/analysis/model_registry.py`, `scripts/pin_models.py` (skeleton); Modify `src/orpheus_mcp/server.py` (CLI dispatch); Test: `tests/test_model_registry.py`.

**Interfaces produces:**
```python
class ModelEntry(BaseModel):
    id: str; task: Literal["multi-stem","single-target","tagger","dereverb",
                           "drumkit-split","transcription","chord","lyrics-asr"]
    targets: list[str]; engine: Literal["audio-separator","msst","panns","btc",
                                        "musicxlab","fasterwhisper","basicpitch","internal"]
    checkpoint_url: str | None; sha256: str | None      # None until pinned; loader REJECTS None outside pin mode
    vram_gb: float; rt_factor: float | None; benchmark: str | None
    license: str; rank: int; tagger_classes: list[str]  # empty => bleed/exposure gates N/A (§4)
def load_registry(*, allow_unpinned: bool = False) -> list[ModelEntry]
def resolve(task: str, target: str, *, vram_free_gb: float | None) -> list[ModelEntry]  # rank order, §15.5 filter
def free_vram_gb() -> float | None      # None => no CUDA (guard per §15.5: is_available() FIRST)
GPU_SEMAPHORE: threading.Semaphore     # 1 slot, module-level
def checkpoint_dir() -> Path            # ~/.orpheus_cache/checkpoints/<id>/
```

- [ ] **Step 1: failing tests** (`tests/test_model_registry.py`) — assert: seed JSON parses into ≥12 entries covering every §4 slot (`4stem-a/bsroformer`, `4stem-b/scnet-xl`, `4stem-c/htdemucs-ft`, `vocals/melroformer`, `6stem/bsroformer-sw`, `6stem-fallback/htdemucs-6s`, `guitar/melroformer-becruily`, `drumsep`, `dereverb/anvuew` (task=`dereverb`, `targets=["vocals"]`), `tagger/panns-sed`, `chord/btc-large-voca` rank 1 + `chord/musicxlab-ismir2019` rank 2, `lyrics/faster-whisper-large-v3`, `transcribe/basic-pitch`); `load_registry()` raises `RegistryError` when any `sha256` is `None` and `allow_unpinned=False`; `resolve("multi-stem","vocals",vram_free_gb=9.0)` returns rank order with entries over `0.8*vram` filtered (§15.5, `VRAM_FIT_FRACTION`); `resolve(...,vram_free_gb=None)` returns only CPU-capable entries (`htdemucs-ft`, internal baseline); `free_vram_gb()` returns `None` (monkeypatched no-CUDA) without raising; empty `tagger_classes` allowed; **NC-licensed entries (LarsNet) excluded from `resolve()` unless `allow_nc=True`** (the Global-Constraints §1.2 rule, mechanically enforced — test both sides).
- [ ] **Step 2:** run → FAIL (module missing).
- [ ] **Step 3:** implement. Seed JSON: every §4 table row, `sha256: null`, `checkpoint_url` = best-known release URL, `tagger_classes` = AudioSet ids per target (vocals→"Singing"/"Speech"; drums→"Drum kit"/"Drum"/"Snare drum"…; guitar→"Guitar"/"Electric guitar"/"Acoustic guitar"; bass→"Bass guitar"; synth-pad→`[]`). `fetch-models` CLI: for each entry with a url — download to `checkpoint_dir()`, sha256-verify (pin mode records instead), check ffmpeg on PATH first, actionable error naming `winget install ffmpeg` if absent. **Scope: every registry checkpoint** — separator ckpts (audio-separator's lazy download-on-first-use disabled via its model-dir setting, §9), PANNs SED weights (pre-fetched to `checkpoint_dir()`, not `~/panns_data`), faster-whisper large-v3 CT2 snapshot (HF, ungated), BTC + musicxlab ckpts. Checkpoints live in the cache dir, never in the repo.
- [ ] **Step 4:** run → PASS. ruff+mypy clean.
- [ ] **Step 5:** Commit `feat(analysis): model registry data + loader + fetch-models CLI (hashes unpinned until 0.3)`.

### Task 0.3: Vendor MSST (SCNet path) + PIN RUN (a) — desktop, human-in-loop

**Files:** Create `src/orpheus_mcp/_vendor/msst/` (+ upstream LICENSE, pinned commit noted in `__init__.py`), finish `scripts/pin_models.py`; Create `docs/pins/2026-07-model-pins.md`; Modify `data/model_registry.json` (record hashes). Test: `tests/test_pin_models.py` (CI-safe: pin-report parser + registry now fully pinned) + `@pytest.mark.desktop` smoke.

- [ ] **Step 1:** vendor the minimal SCNet inference path from ZFTurbo/Music-Source-Separation-Training @ pinned commit (model def + inference loop only; no training code). Record commit hash in `_vendor/msst/__init__.py` docstring.
- [ ] **Step 2:** `scripts/pin_models.py`: `uv run --extra separation python scripts/pin_models.py --audio <owned.mp3>` → for each separation/dereverb registry entry: fetch checkpoint, sha256 → write into JSON; run on the mp3 (full song), write stems to `docs/pins/audio_out/` (gitignored), record wall time + rt_factor on the 3080 into the JSON + a markdown pin report table (model, resolved url, sha256, load s, sep s, notes column for ear-check).
- [ ] **Step 3 (HUMAN):** run it; ear-check each stem set; fill the notes column. Any failed slot (404, broken engine, garbage output) → swap the slot's alternate per §4 and re-run; log the swap in the pin report + Deviations D6.
- [ ] **Step 4:** CI-safe test: `load_registry()` now succeeds with zero `None` hashes; pin report exists and lists every separation entry.
- [ ] **Step 5:** Commit `feat(analysis): vendor MSST SCNet inference; pin all separation checkpoints (pin run a)`.

### Task 0.4: Fixture forge (§10, §15.1 — headless, D4)

**Files:** Create `scripts/make_fixtures.py`, `tests/fixtures/audio/` (2 committed CI fixtures + `manifest.json`); Test: `tests/test_fixture_manifest.py`.

**Interfaces produces:** desktop set `~/.orpheus_fixtures/`: 24 mixes = 8 genre profiles × 3 densities, 8–16 bars, each dir `{mix.flac, stems/<role>.flac, truth.json}`; `manifest.json` with per-file sha256 + `fixture_set_hash` (sha256 of sorted file hashes — the §15.1/§9 calibration identity). Fixture progressions deliberately include ≥ one maj7 and ≥ one slash chord per genre (the §13 acceptance-bar material). Committed CI subset: `ci_sparse` (4-track instrumental, 8 bars — the `no_vocals` fixture) + `ci_vocal` (adds a vocal lane = **Windows SAPI TTS reading 6 known lyric lines** over the arrangement, D8 — `truth.json` carries the exact lines), 22.05 kHz stereo FLAC, ≤ 6 MB total.

- [ ] **Step 1:** implement: for each (genre, density): build sections via the existing compose stack helpers (`resolve_progression`, `voice_lead`, `bassline_notes`, `parse_drum_grid`) → write per-role standard MIDI files (`mido` is NOT a dep — write minimal SMF via existing note dicts + a tiny local writer in the script) → render each role through `sfizz_render` (pinned download url+sha256 in the script, VSCO2-CE + CC0 kit SFZs pinned likewise) → peak-normalize stems to −6 dBFS, sum → mix.flac; truth.json records role→stem map, bpm, key, progression. `ci_vocal`'s vocal lane (D8): generated once on Windows via PowerShell `System.Speech.Synthesis` reading the 6 truth lines, time-stretched onto the bar grid, mixed at −12 dBFS — the WAV is committed (as FLAC) so no other box ever needs SAPI.
- [ ] **Step 2 (HUMAN, desktop):** run `uv run python scripts/make_fixtures.py --out ~/.orpheus_fixtures`; spot-listen 2–3; commit only the CI subset + manifest.
- [ ] **Step 3:** test: manifest hashes match committed files; `fixture_set_hash` stable across runs (re-render `ci_sparse` → identical hash; sfizz_render is deterministic given same seed/sfz).
- [ ] **Step 4:** Commit `feat(test): synthetic-truth fixture forge + committed CI fixture subset`.

### Task 0.5: basic-pitch isolated env (§9 packaging decision)

**Files:** Create `scripts/setup_transcribe_env.py` + `src/orpheus_mcp/analysis/transcribe_cli.py`; Test: `tests/test_transcribe_cli.py` (subprocess contract mocked; `@pytest.mark.desktop` real run).

- [ ] Steps: script creates `~/.orpheus_transcribe_env` via `uv venv --python 3.11` + `uv pip install basic-pitch`; `transcribe_cli.transcribe(wav_path) -> list[NoteEvent]` invokes `uv run --python <env> basic-pitch <out> <wav>` CLI-style, parses the CSV output; defined `TranscribeEnvMissing` error naming the setup command. Test mocks `subprocess.run` asserting exact argv + CSV parse; desktop test runs real env. Commit `feat(analysis): isolated basic-pitch subprocess env (T5 lane)`.

---

## Phase 1 — Slice 1: config + cache + recon (T0/T1)

### Task 1.1: `analysis/config.py` — the §16 constants

**Files:** Create `src/orpheus_mcp/analysis/config.py`; Test: folded into Task 1.2's suite (constants are data; the test asserts the §16 values exist and match).

- [ ] **Step 1:** transcribe §16 verbatim into a frozen dataclass + singleton:

```python
"""ALL analysis tunables (§16). Nothing is hard-coded at a call site.
Rows marked 'cal' are re-fit by scripts/calibrate_gates.py, which records the
fixture_set_hash it fit against (cache invalidation, §9)."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class AnalysisConfig:
    GATE_MIX_RESIDUAL_PASS_DB: float = -20.0        # cal
    GATE_BLEED_PASS: float = 0.20                   # cal
    GATE_SILENCE_PASS_DB: float = -35.0             # cal
    GATE_SILENCE_TAIL_ALLOWANCE_S: float = 3.0      # cal
    GATE_AGREEMENT_PASS: float = 0.90               # cal
    TIMELINE_ACTIVITY_THRESHOLD: float = 0.30       # cal
    EXPOSURE_THRESHOLD: float = 0.15                # cal
    VAD_VOCAL_ACTIVE_DBFS: float = -45.0            # cal
    BEAT_GRID_CONFIDENCE_MIN: float = 0.60
    FIXED_GRID_BARS_TO_SECONDS: float = 2.0
    CHORD_STABILITY_RATIO: float = 0.60
    CHORD_EXTENSION_MIN_BEATS: int = 2
    CHORD_EXTENSION_ENERGY_RATIO: float = 0.15      # cal
    KEY_WINDOW_BARS: int = 16
    KEY_HOP_BARS: int = 4
    KEY_HYSTERESIS_WINDOWS: int = 2
    LYRIC_MATCH_MIN_SCORE: float = 0.60
    FX_LOOP_MAX_ITERATIONS: int = 4
    FX_LOOP_PLATEAU_EPSILON: float = 0.05
    WIDEN_MONO_SAFETY_CORR_MIN: float = 0.50
    DEFAULT_MAX_TIER: str = "T4"
    DEFAULT_MAX_SECONDS: float = 300.0
    VRAM_FIT_FRACTION: float = 0.80
    CACHE_SIZE_CAP_GB: float = 20.0
    # --- plan-added (allowed by §16's normative rule; NOT in the spec table) ---
    SEPARATION_CONTEXT_PAD_S: float = 5.0           # §3.1 T3 crop pad
    CHORD_SELF_TRANSITION: float = 0.90             # §13.1 Viterbi stay-probability
    EXPOSURE_MIN_WINDOW_S: float = 3.0              # shortest usable exposed window (§3.1b)
    ANALYSIS_SR: int = 44100                        # §8 common analysis rate
    CACHE_DIR_ENV: str = "ORPHEUS_CACHE_DIR"        # test override hook
    CALIBRATION_SDR_GOOD: float = 7.0               # §15.1 ROC "good" floor
    CALIBRATION_SDR_BAD: float = 4.0                # §15.1 ROC "bad" ceiling

CFG = AnalysisConfig()
```

- [ ] **Step 2:** Commit with Task 1.2 (`feat(analysis): config constants (§16) + cache filesystem`).

### Task 1.2: `analysis/cachefs.py` — decode, PCM hash, artifact store

**Files:** Create `src/orpheus_mcp/analysis/cachefs.py`; Test: `tests/test_cachefs.py`.

**Interfaces produces:**
```python
class DecodeError(Exception): ...          # message says WHY (corrupt, truncated, DRM)
def decode(path: str) -> Pcm               # soundfile first; mp3/m4a via ffmpeg -> wav pipe
def pcm_hash(path: str) -> str             # sha256 of CANONICAL pcm (see below); memoized via file-bytes-hash
class CacheFS:
    def __init__(self, root: Path | None = None)   # default ~/.orpheus_cache, env-overridable (CFG.CACHE_DIR_ENV)
    def artifact_dir(self, pcm_hash: str) -> Path
    def read_json(self, pcm_hash: str, kind: str, meta: dict) -> dict | None   # None on miss OR meta mismatch
    def write_json(self, pcm_hash: str, kind: str, obj: dict, meta: dict) -> Path
    def stem_dir(self, pcm_hash: str, model_id: str, region: Window) -> Path
        # §9 layout: stems/<model-id>/<start_ms>-<end_ms>/ ; ladder runs cache under kind "ladder"
    def pin(self, pcm_hash: str) / unpin(...)       # in-flight protection (§9)
    def evict_lru(self) -> int                       # bytes freed; skips pinned
    def clear(self) -> None
```

Locked decisions an implementer must not re-decide:
- **Canonical PCM for hashing** = native sample rate + channel count, float32 samples converted to int16 (round-half-away, clip) → bytes → sha256. Rationale: float LSBs differ across decoder builds; int16 canonicalization makes the hash stable for lossless paths and stable-per-ffmpeg-build for lossy. `meta.json` of every artifact records `decoder` (`"soundfile"` or `"ffmpeg <version>"`); an ffmpeg upgrade may orphan lossy-file entries — that is a cache MISS by design, never corruption.
- **Memo file** `<root>/hash_memo.json`: `sha256(file_bytes) -> pcm_hash` so T0 lookups never re-decode (§3.1 T0).
- **DRM**: extension `.m4p` → immediate `DecodeError("DRM-protected …")`; ffmpeg nonzero exit → `DecodeError` with stderr tail (§9).
- **meta sidecar** per §9: `{model_id, checkpoint_sha256, registry_version, verify_config_hash, schema_version, budget_cap?}` — `read_json` compares field-for-field; any mismatch = miss (stale left for LRU). `verify_config_hash` = sha256 of the sorted `cal`-marked CFG fields **as effective at read time, i.e. after `data/calibrated_thresholds.json` overrides load (D7)** — a re-calibration busts stale gate artifacts. `registry_version` = sha256 of `data/model_registry.json` bytes.
- **LRU** on artifact-dir mtime; cap `CFG.CACHE_SIZE_CAP_GB`; pinned dirs skipped.

- [ ] **Step 1: failing tests** — the load-bearing ones verbatim:

```python
def test_same_audio_two_containers_share_hash(tmp_path):
    wav = make_sine(tmp_path / "a.wav"); flac = transcode(wav, tmp_path / "a.flac")
    assert pcm_hash(str(wav)) == pcm_hash(str(flac))

def test_id3_edit_does_not_change_hash(tmp_path, ci_mp3):   # tag bytes differ, audio identical
    tagged = retag(ci_mp3, tmp_path / "tagged.mp3", title="X")
    assert pcm_hash(str(ci_mp3)) == pcm_hash(str(tagged))

def test_meta_mismatch_is_miss(tmp_path):
    fs = CacheFS(tmp_path); h = "ab" * 32
    fs.write_json(h, "recon", {"x": 1}, meta={"schema_version": 1, "registry_version": "r1"})
    assert fs.read_json(h, "recon", meta={"schema_version": 2, "registry_version": "r1"}) is None
    assert fs.read_json(h, "recon", meta={"schema_version": 1, "registry_version": "r1"}) == {"x": 1}

def test_lru_skips_pinned(tmp_path, monkeypatch): ...        # cap forced tiny; pinned survives, other evicted
def test_drm_rejected(): ...                                  # .m4p -> DecodeError mentioning DRM
def test_truncated_file_raises(tmp_path): ...                 # half a wav -> DecodeError, never partial silence
def test_memo_prevents_second_decode(tmp_path, monkeypatch): ...  # decode() call-counted; 2nd pcm_hash() hits memo
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** run → PASS (CI-safe: fixtures from Task 0.4; `ci_mp3` generated by transcoding a committed FLAC with ffmpeg at session start, skipped if ffmpeg absent — marked accordingly). **Step 5:** commit (with 1.1).

### Task 1.3: `analysis/recon.py` + `recon_reference` tool

**Files:** Create `src/orpheus_mcp/analysis/recon.py`; Create `src/orpheus_mcp/tools/reference.py` (first tool); Modify `src/orpheus_mcp/registry.py` (+`"reference"` in `_CATEGORY_IMPORTS` and in `default`/`full` ONLY — §3 says default/full; `explain` stays as-is); Test: `tests/test_recon.py`, extend `tests/test_registry.py`.

> Existing `analysis/audio.py` / `AudioCharacter` / `analyze_audio_character` stay untouched — they remain the project-side quick fingerprint; this pipeline is the deep path the spec says *absorbs the roadmap line*, not a deletion of shipped code.

**Interfaces produces:** `def run_recon(pcm: Pcm) -> Recon`; tool `recon_reference(file: str) -> Recon` (decodes, caches under kind `"recon"`, `readOnlyHint`).

Locked implementation choices (librosa, all tuning-corrected per §T1):
- tempo/beats: `librosa.beat.beat_track`; downbeats/meter: onset-strength periodicity over beat groups of {3,4} → pick argmax correlation, confidence = margin ratio; grid confidence = mean beat-onset alignment score normalized [0,1] (below `BEAT_GRID_CONFIDENCE_MIN` ⇒ callers switch to fixed grid — the flag rides on `BeatGrid.confidence`, consumers decide).
- tuning: `librosa.estimate_tuning` → cents; **all chroma downstream uses `tuning=` param** (§T1 requirement — test asserts a +40-cent-detuned fixture still keys correctly).
- key: Krumhansl-Schmuckler on tuning-corrected chroma; return best + 2 alternates with correlation-derived confidence.
- sections: self-similarity novelty (`librosa.segment`) → boundaries; `duration < 30 s` ⇒ single section (§T1 short-file rule).
- band sketch: reuse `analysis/audio.py::_band_energies_db` shape but 8 log-spaced bands 60 Hz–16 kHz (names `"60-120"`, … locked in test).

- [ ] **Step 1: failing tests** — constructed-signal + fixture assertions: 120 BPM click fixture → `tempo_bpm ∈ [118,122]`, `grid.confidence > 0.8`; `ci_sparse` fixture → key matches `truth.json` root/mode; +40-cent detuned sine-progression fixture → `tuning_offset_cents ∈ [30,50]` and key still correct; 10 s noise burst → single section; recon of same file twice → second call served from cache (decode call-counted); registry test: `reference` category present in `default`/`full`, absent from `explain`; plus one `@pytest.mark.desktop` test running recon over 3 desktop-set fixtures asserting all fields present incl. tuning/downbeats/grid-confidence (the §12 slice-1 bar, verbatim).
- [ ] **Steps 2–4:** red → implement → green (CI-safe; librosa is in the `analysis` extra which CI's new job installs).
- [ ] **Step 5:** Commit `feat(analysis): T1 recon (tempo/key/tuning/sections/bands) + recon_reference tool`.

---

## Phase 2 — Slice 2: tagger timeline (T2) + SED pin (d)

### Task 2.1: `analysis/timeline.py` + `instrument_timeline` tool

**Files:** Create `src/orpheus_mcp/analysis/timeline.py`; Modify `tools/reference.py`; Test: `tests/test_timeline.py`.

**Interfaces produces:**
```python
class TaggerEngine(Protocol):
    classes: list[str]                                  # full AudioSet label list
    def activations(self, pcm: Pcm) -> "np.ndarray"     # (frames, classes) sigmoids, hop from .hop_s
    hop_s: float
class PannsTagger:                                       # lazy torch import; head = CFG-recorded pick from 2.2
    def __init__(self, head: str = "DecisionLevelAtt") ...
def build_timeline(pcm: Pcm, tagger: TaggerEngine) -> Timeline   # subset to union of registry tagger_classes (§9 tool contract)
def most_exposed_window(tl: Timeline, target_classes: list[str],
                        min_len_s: float = CFG.EXPOSURE_MIN_WINDOW_S) -> Window | None
    # §3.1b: window where normalized non-target activation (max_nontarget/(target+eps), §5.2 formula)
    # < CFG.EXPOSURE_THRESHOLD, target above CFG.TIMELINE_ACTIVITY_THRESHOLD; returns most-exposed, else None
def target_absent(tl: Timeline, target_classes: list[str]) -> bool
    # §3.1c: True iff target never exceeds CFG.TIMELINE_ACTIVITY_THRESHOLD anywhere
```

- [ ] **Step 1: failing tests** — all against a `StubTagger` (deterministic synthetic activations; class list = 8 names):
  exposure math verbatim (§5.2 formula: frame with target 0.9/others 0.05 ⇒ normalized 0.056 → exposed; 0.9/0.2 ⇒ 0.222 → not, at default 0.15); `target_absent` true when target peaks at 0.29; `build_timeline` output round-trips through `Timeline.model_validate_json` and frames align `len(frames) == ceil(duration/hop)`; max-pool aggregation to the locked 1 s hop (stub emits 10 ms frames, one 0.9 spike in a second ⇒ that second reads 0.9); empty `tagger_classes` target → `most_exposed_window` returns `None` and `target_absent` returns `False` (N/A semantics — caller must treat as unknown, locked by test); tool contract test: `instrument_timeline(file)` returns the mapped-subset classes + raw activations, cached under kind `"timeline"` (§9 contract — thresholds applied by consumers, never baked in).
- [ ] **Steps 2–4:** red → implement → green. `PannsTagger` real-model path covered by one `@pytest.mark.desktop` test on `ci_sparse` (asserts drum class activates during drum bars).
- [ ] **Step 5:** Commit `feat(analysis): T2 instrument timeline + exposure/absent-target rules (§3.1b/c)`.

### Task 2.2: §15.6 SED-head script + PIN RUN (d)

**Files:** Create `scripts/pick_sed_head.py`; Modify `data/model_registry.json` (tagger entry gains `"head": <winner>`), `docs/pins/2026-07-model-pins.md`; Test: `tests/test_sed_script.py` (script's F1 math on synthetic frames — CI-safe).

- [ ] **Steps:** script loads desktop fixtures (`~/.orpheus_fixtures` manifest), builds frame truth from `truth.json` role activity (a role is truth-active in frames where its stem RMS > −50 dBFS), runs the three heads, reports frame-F1 per head, ties → DecisionLevelAtt (§15.6). Unit test feeds hand-built truth/pred arrays and asserts F1 math + tie-break. **(HUMAN, desktop):** run, record winner in registry + pin report. Commit `feat(analysis): SED-head selection script; pin tagger head (pin run d)`.

---

## Phase 3 — Slice 3: chord baseline + theory lane (§13, CPU)

### Task 3.1: baseline recognizer (templates + Viterbi) + extensions + slash

**Files:** Create `src/orpheus_mcp/analysis/harmony.py`; Test: `tests/test_harmony_baseline.py`.

**Interfaces produces:**
```python
def beat_chroma(pcm: Pcm, grid: BeatGrid, tuning_cents: float) -> np.ndarray  # (beats, 12), HPSS-harmonic
def chord_lane_baseline(pcm: Pcm, recon: Recon) -> ChordsLane
    # grid.confidence < CFG.BEAT_GRID_CONFIDENCE_MIN -> fixed 1 s windows, lane.grid="fixed" (§13.4 fallback)
def refine_extensions(lane: ChordsLane, chroma: np.ndarray, grid: BeatGrid) -> ChordsLane   # §13.3
def detect_slash(lane: ChordsLane, bass_pc_per_beat: list[int | None]) -> ChordsLane        # §13.2
def bass_pc_lane(pcm: Pcm, grid: BeatGrid) -> list[int | None]   # low-passed mix (<250 Hz) + pYIN, per beat
def theory_layer(lane: ChordsLane, grid: BeatGrid) -> ChordsLane                            # §13.4
```

Locked algorithm (≈200 LOC budget per §13.1):
- Vocabulary: 12 roots × {maj, min, 7, maj7, min7, sus2, sus4, dim, aug} + `N` = 109 states. Binary 12-d templates, L2-normalized; emission = cosine(chroma_beat, template); `N` template = flat 1/12 vector scaled 0.85 (beats quiet frames only).
- Viterbi: self-transition `CFG.CHORD_SELF_TRANSITION`, remainder uniform across other states; log-space.
- §13.3 extensions: per segment ≥ `CHORD_EXTENSION_MIN_BEATS` beats — residual chroma after subtracting the triad template; pitch classes with energy > `CHORD_EXTENSION_ENERGY_RATIO` × triad energy, stable in ≥ `CHORD_STABILITY_RATIO` of the segment's beats → extension (candidate set: 7, maj7 upgrade, add9, 6, #11, 13, sus tones); label rewrite (`C`→`Cadd9`) with per-extension confidence.
- §13.2 slash: per segment — single stable non-root bass pc across ≥ `CHORD_STABILITY_RATIO` of beats AND a chord tone → `bass` set + label `X/Y`; stable non-chord-tone → `extensions += ["pedal"]`; unstable → untouched. (Walking-bass test is mandatory: C-major bar with bass C-D-E-G per beat must stay `C`.)
- §13.4 theory: key windows `KEY_WINDOW_BARS`/`KEY_HOP_BARS` over the lane (Krumhansl on segment-weighted pitch classes), modulation accepted only after `KEY_HYSTERESIS_WINDOWS` consecutive wins; music21 roman per segment against local key; global key = duration-weighted mode of the lane (headline stays recon's Krumhansl, §13.4 precedence).

- [ ] **Step 1: failing tests** — rendered-fixture + constructed-chroma cases: synthesized `C-Am-F-G` (from `ci_sparse` truth) → ≥ 90% majmin-correct beats (the §12 slice-3 bar, asserted numerically); constructed chroma with strong added D over C-E-G across 4 beats → `Cadd9`; 1-beat segment never upgrades (§13.3 min-beats rule); walking-bass case above; `C/E` fixture (bass pc 4 stable) → slash emitted; rubato flag: grid confidence 0.3 → `lane.grid == "fixed"` and segments on 1 s windows; hysteresis: constructed 32-bar two-key lane flips key only after 2 windows; roman: `F` in C-major window → `"IV"`.
- [ ] **Steps 2–4:** red → implement → green. **Step 5:** Commit `feat(analysis): §13 baseline chord lane — templates+Viterbi, extensions, slash, theory layer`.

### Task 3.2: `get_chords` tool + chords cache + PIN RUN (b, part 1)

**Files:** Modify `tools/reference.py`; `docs/pins/2026-07-model-pins.md`; Test: extend `tests/test_m2_tools.py`-style contract tests in `tests/test_reference_tools.py` (new).

- [ ] **Steps:** tool `get_chords(file, region=None) -> ChordsLane` — cache kind `"chords"` (meta incl. produced_by="baseline"), region filter clamps to `[0, duration]` (§3.1d — test with region end 10× duration). **(HUMAN, desktop):** run on 2–3 owned songs with known 7ths/slash chords; record exact-label impressions in the pin report (spec §12.2b — the model-lane comparison repeats in Phase 9). Commit `feat(tools): get_chords over the baseline lane`.

---

## Phase 4 — Slice 4: engines + `separate_region` + gates 1–3 (D2)

### Task 4.1: engine adapters

**Files:** Create `src/orpheus_mcp/analysis/separation.py`, `tests/stub_engines.py` (`StubEngine`, `StubTagger`, `StubAsr`, `StubChordModel` — the shared scripted fakes every later phase's CI tests import); Test: `tests/test_separation.py`.

**Interfaces produces:**
```python
class SeparatorEngine(Protocol):
    def separate(self, wav_path: str, entry: ModelEntry, out_dir: Path) -> dict[str, str]  # target -> flac path
class AudioSeparatorEngine: ...   # wraps audio_separator.Separator; model file from checkpoint_dir(); network OFF
class MSSTEngine: ...             # vendored SCNet
class StubEngine:                 # tests/CI: returns pre-baked stems from a dict; lives in tests/stub_engines.py
def separate_region(pcm: Pcm, entry: ModelEntry, region: Window | None,
                    engine: SeparatorEngine | None = None) -> dict[str, StemResult]
```
Locked behavior: crop `region ± CFG.SEPARATION_CONTEXT_PAD_S` (clamped §3.1d) → temp wav → engine under `GPU_SEMAPHORE` with §15.5 VRAM guard (`free_vram_gb()`; entry doesn't fit `VRAM_FIT_FRACTION × free` → next `resolve()` candidate → CPU chain; chosen fallback + reason recorded for tool meta) → trim outputs back to the exact requested region (§3.1d) → resample/length-align to input (§5 precondition) → mono-in ⇒ dual-mono to stereo-only engines, stereo descriptors later stay N/A (§7 mono policy).

- [ ] **Step 1: failing tests** (StubEngine): pad+trim math (region [10,20] on a 60 s file → engine receives [5,25], results trimmed to [10,20] exactly, sample-accurate); clamp at EOF (region [55,70] → [50,60] in, [55,60] out, `region_trimmed` says so); alignment (stub returns stems 137 samples long-padded → aligned to mix length); VRAM fallback (monkeypatched `free_vram_gb`→1.0, 6 GB entry → falls to `htdemucs-ft`-cpu candidate, degradation reason string present); semaphore (two threads, instrumented engine asserts no overlap).
- [ ] **Steps 2–4:** red → implement → green; `@pytest.mark.desktop @pytest.mark.gpu` test runs BS-RoFormer AND SCNet on `ci_sparse` (both engines importable, produce 4 stems — the §12 slice-4 bar's engine half).
- [ ] **Step 5:** Commit `feat(analysis): separator engines (audio-separator + vendored SCNet) with region crop + VRAM guard`.

### Task 4.2: gates 1–3 + `separate_region` tool

**Files:** Create `src/orpheus_mcp/analysis/verify.py` (gates 1–3; gate 4 lands Phase 5); Modify `tools/reference.py`; Test: `tests/test_verify_gates.py`, `tests/test_reference_tools.py`.

Locked gate math (§5, exactly):
```python
def gate_mix_consistency(mix: np.ndarray, stems: dict[str, np.ndarray], multi_stem: bool) -> GateResult:
    # N/A unless multi_stem (§5.1). value = 20*log10(rms(mix - sum(stems))/rms(mix));
    # pass iff value <= CFG.GATE_MIX_RESIDUAL_PASS_DB
def gate_bleed(stem_activations: np.ndarray, target_idx: list[int], mapped_idx: list[int]) -> GateResult:
    # tagger ON THE STEM (§5.2). active frames = target act > CFG.TIMELINE_ACTIVITY_THRESHOLD;
    # zero active frames -> N/A; empty target_idx -> N/A;
    # value = mean over active frames of max(non-target)/(target+1e-6), clipped [0,1]; pass < CFG.GATE_BLEED_PASS
def gate_silence_honesty(stem: np.ndarray, sr: int, timeline: Timeline, target_idx: list[int]) -> GateResult:
    # inactive windows minus CFG.GATE_SILENCE_TAIL_ALLOWANCE_S after each active region (§5.3 carve-out);
    # no inactive windows -> N/A (wall-to-wall);
    # value = 20*log10(rms(stem, inactive)/rms(stem, active)); pass <= CFG.GATE_SILENCE_PASS_DB
def run_gates(...) -> GateSet   # overall: fail if any applicable gate fails; unverifiable iff ALL N/A (§5 decision rule)
```

- [ ] **Step 1: failing tests** — constructed-signal, each gate: perfect stems sum → residual < −60 → pass; one stem zeroed → fail; single-target → mix_consistency N/A; bleed synthetic activations both sides of 0.20; zero-active-frames → N/A; silence: stem with a 2 s reverb tail after last note passes (tail carve-out), same energy 5 s later fails; wall-to-wall pad → N/A; `overall` decision-rule truth table incl. `unverifiable` (all N/A) and "N/A excluded" (§5).
- [ ] **Step 2–4:** red → implement → green. Tool `separate_region(file, target, region=None, model_id=None) -> StemResult` — resolves via registry, runs gates 1–3 (tagger on stem), caches stem FLAC + `verify.json` per §9 layout, returns gates in result (meta carries VRAM/degradation notes). Contract test with StubEngine + StubTagger end-to-end.
- [ ] **Step 5:** Commit `feat(analysis): verification gates 1-3 (§5) + separate_region tool with per-stem gate results`.

---

## Phase 5 — Slice 5: agreement gate, ensemble, ladder, calibration

### Task 5.1: gate 4 + ensemble

**Files:** Modify `analysis/verify.py`; Test: extend `tests/test_verify_gates.py`.

```python
def gate_agreement(stem_a: np.ndarray, stem_b: np.ndarray, sr: int) -> GateResult
    # log-mel (128 bands, hop 1024) Pearson r over flattened frames; pass >= CFG.GATE_AGREEMENT_PASS (§5.4)
def ensemble_avg_wave(stems: list[np.ndarray]) -> np.ndarray   # length-aligned mean (§5.4 avg_wave)
```
- [ ] Tests: identical stems → r≈1 pass; noise vs sine → fail; ensemble of two DC-offset copies = mean; then commit `feat(analysis): cross-model agreement gate + avg_wave ensemble`.

### Task 5.2: the escalation ladder (§6) in `analysis/router.py`

**Files:** Create `src/orpheus_mcp/analysis/router.py`; Test: `tests/test_ladder.py`.

**Interfaces produces:**
```python
TIER_ORDER = ["T0","T1","T2","T3","T4","T5"]
QUESTION_TIERS = {"recon":"T1","timeline":"T2","chords":"T2","sound_at":"T3",
                  "compare_track":"T3","lyric_navigate":"T3","notes":"T5"}   # §3.1 routing table
@dataclass
class LadderOutcome:
    stems: dict[str, StemResult] | None; rung: str        # "rank1"|"R1".."R6"|"absent"|"exposed"
    log: list[str]; budget_capped: bool; source: str      # what R6/exposure used
def climb(pcm, target, region, *, registry, engines, tagger, timeline,
          max_tier: str, max_seconds: float, cache: CacheFS, resume_from: str | None) -> LadderOutcome
```
Locked semantics (each is a named test):
- **Pre-flight order:** absent-target check (§3.1c, via `target_absent`; empty tagger_classes ⇒ skipped) → exposure check (§3.1b: exposed window found ⇒ return `rung="exposed"`, no separation) → rank-1.
- Rung sequence R1→R5 exactly as §6; every transition logs `"<rung>: <failing gate>=<value>"`.
- **R3 precondition** (§6): stems-to-subtract must individually pass gates 2–4; if the climb is happening because those same stems failed → R3 skipped (logged); residual `StemResult` flagged `complement=True` via SourceMeta downstream.
- **R6 gate:** only when `max_tier=="T5"`; input = best gate-scoring artifact if any gate passed on it, else most-exposed raw window; source recorded (§6).
- **Checkpoint-less targets** (no registry entry for target): enter at R3, then R5; R1/R2 skipped by construction (§6).
- **Budget:** wall-clock via injected `clock: Callable[[], float]` (tests fake it); expiry mid-ladder → return best-so-far + `budget_capped=True`; cache records the rung + cap; `resume_from` continues the climb (test: capped run at R2, resumed run starts at R3, no re-run of R1/R2 — §6 exhaustion semantics).
- Completion (pass or R5-exhausted at default tier) caches final; **re-asking a completed ladder never re-climbs** (test).
- [ ] Red → implement (engines/tagger all injected; tests use stubs with scripted gate outcomes) → green → commit `feat(analysis): deterministic escalation ladder (§6) with budget resume`.

### Task 5.3: `verify_stem` tool + calibration script + CI wiring

**Files:** Modify `tools/reference.py`; Create `scripts/calibrate_gates.py`; Modify `.github/workflows/ci.yml`; Test: `tests/test_calibration.py` (math only, CI-safe).

- [ ] **Steps:** tool `verify_stem(stem_path, mix_path, region, timeline=None) -> GateSet` (§9 contract). Calibration script per §15.1: run every registry separator over desktop fixtures, compute true SDR (`10*log10(rms(truth)^2/rms(truth-est)^2)`) + all gate metrics, pick per-gate ROC-knee threshold separating SDR>`CALIBRATION_SDR_GOOD` from SDR<`CALIBRATION_SDR_BAD`, emit `data/calibrated_thresholds.json` + fixture_set_hash; `config.py` loads overrides from that file when present. Unit tests: SDR math on known signals; ROC-knee picker on synthetic score distributions; CFG override loading. CI: calibration smoke uses `--ci-smoke`, which swaps in `tests/stub_engines.py` fakes over the 2 committed fixtures — asserts the pipeline (SDR math → gate metrics → ROC → JSON emit) end-to-end with ZERO checkpoint downloads (CI invariant, Task 0.1); real-model calibration is desktop-only. **(HUMAN, desktop):** full calibration run + 6-song ear-labeled sanity panel per §15.1; commit updated thresholds. Commit `feat(analysis): verify_stem tool + §15.1 gate calibration pipeline`.

---

## Phase 6 — Slice 6: descriptors (§7)

### Task 6.1: `analysis/descriptors.py` + `characterize` tool

**Files:** Create `src/orpheus_mcp/analysis/descriptors.py`; Modify `tools/reference.py`, `analysis/model_registry.py` (dereverb resolve); Test: `tests/test_descriptors.py`.

**Interfaces produces:**
```python
@dataclass
class AudioSource:            # resolver for the §9 stem-handle convention
    pcm: Pcm; meta: SourceMeta
def resolve_source(handle: StemHandle | str, cache: CacheFS) -> AudioSource
    # str => literal path (stem_type="mix-region", produced_by="file"); handle => cached stem + its meta
def compute_character(src: AudioSource, *, dereverb: SeparatorEngine | None = None) -> SoundCharacter
```
Descriptor keys (locked — these exact dict keys, tested):
- `spectral`: `centroid_hz, rolloff_hz, tilt_db_per_oct, band_<lo>_<hi>_db` (8 recon sketch bands, underscore format — `band_60_120_db`, …; recon's own `band_energy_db` dict keeps its `"60-120"` keys, the two are different artifacts), `hp_ratio` — **plus the six §15.3 rule bands, computed here precisely because the log-spaced sketch bands do NOT align with the rule table's musical bands:** `band_100_250_db, band_200_400_db, band_2000_5000_db, band_10000_16000_db` (rules 4/2/1/3), and `band_5000_9000_dyn_db` (std-dev of per-frame 5–9 kHz energy — rule 13's "dynamic Δ"). These names are the shared contract with `data/intent_rules.json`; Task 10.1 cross-checks them at load time.
- `dynamics`: `lufs_i, lufs_s_max, crest_db, attack_ms_p50, attack_ms_p90, env_kurtosis`.
- `harmonics`: `even_db, odd_db, inharmonicity, flavor` (`"clean"|"warm-even"|"gritty-odd"` — even−odd > 3 dB ⇒ warm-even, < −3 ⇒ gritty-odd, else clean; sustained segments only, via onset-free windows ≥ 400 ms).
- `stereo`: `width_low, width_mid, width_high` (per-band M/S RMS ratio), `corr` — ALL `None` when `src.meta.channels == 1` (§7 mono policy, never upmixed).
- `time_fx`: `delay_ms, delay_feedback, delay_conf, delay_approx=True` always (§7 tempo-sync confound), `mod_rate_hz, mod_depth, mod_class` (`"chorus"|"phaser"|"tremolo"|"vibrato"|None`), `reverb_wet_dry, reverb_rt60_class, reverb_early_late, reverb_approx` (False only on the vocal dereverb path).
- `pitch` (melodic stems only, else `None`): `f0_median_hz, vibrato_rate_hz, vibrato_depth_cents, portamento`.
- Reverb path split (§7): `stem_type=="vocals"` AND dereverb engine available ⇒ wet−dry measurement; else decay-slope heuristic on note releases, `reverb_approx=True`.
- `bandwidth_hz` measured here (spectral cliff: highest band whose energy is within 40 dB of peak band) and stamped into `src.meta` → rides on `SoundCharacter.source` for §8.

- [ ] **Step 1: failing tests** (all constructed signals — the §10/§12 slice-6 bar verbatim):

```python
def test_synthetic_28ms_echo_reads_back():
    sig = click_train() ; echoed = sig + 0.5 * delay(sig, ms=28)
    ch = compute_character(mono_source(echoed))
    assert ch.time_fx["delay_ms"] == pytest.approx(28, abs=3)
def test_tilt_6db_per_oct(): ...                    # shaped noise -6 dB/oct -> tilt in [-7,-5]
def test_tremolo_5hz_detected(): ...                # 5 Hz AM sine -> mod_rate_hz≈5, mod_class in ("tremolo",)
def test_mono_source_stereo_descriptors_are_none(): ...
def test_crest_and_lufs_on_reference_tone(): ...    # -20 dBFS sine: lufs≈-20±1.5, crest≈3.01±0.3
def test_even_vs_odd_flavor(): ...                  # sine+2nd harm -> warm-even; sine+3rd -> gritty-odd
def test_vocal_reverb_uses_dereverb_when_available(): ...   # stub dereverb called iff stem_type=="vocals"
def test_bandwidth_cliff_detected(): ...            # 22.05k audio lowpassed at 16k -> bandwidth_hz in [15k,17k]
```

- [ ] **Steps 2–4:** red → implement → green. Tool `characterize(source, region=None) -> SoundCharacter`, cached under `descriptors/<stem-or-region>.json` (§9).
- [ ] **Step 5:** Commit `feat(analysis): §7 SoundCharacter descriptors + characterize tool`.

---

## Phase 7 — Slice 7: `render_track_region` (§15.2 — bridge)

### Task 7.1: Lua handler + fake + tool

**Files:** Modify `src/orpheus_mcp/bridge/lua/orpheus_bridge.lua`, `tests/fake_reaper.py`, `src/orpheus_mcp/tools/render.py` (the python tool lives in the existing `render` category — it is bridge-side per §9, not a `reference` tool; bridge verb name `render_track_region`); Create `tests/lua/test_render_handlers.lua`; Modify `scripts/run_lua_tests.py` (SUITES); Test: `tests/test_render_region.py`.

Locked Lua sequence — §15.2 steps 1–8 exactly; the handler skeleton (implementer fills only mechanical locals):

```lua
handlers.render_track_region = function(p)
  if RENDER_MUTEX_HELD then return err("render busy") end          -- §15.2 serialization
  RENDER_MUTEX_HELD = true
  local ok, result = pcall(function()
    if reaper.GetPlayState() ~= 0 then error("transport is playing/recording", 0) end  -- step 1 (default: error)
    local snap = snapshot_render_state()   -- RENDER_* via GetSetProjectInfo(+_String), ALL track I_SOLO, master FX bypass
    local tr, rerr = resolve_track(p.track)  -- (fixed in review: tuple check + fail-before-mutation)
    if not tr then error(rerr, 0) end
    for i = 0, reaper.CountTracks(0) - 1 do                         -- step 3: clear ALL solos first
      reaper.SetMediaTrackInfo_Value(reaper.GetTrack(0, i), "I_SOLO", 0)
    end
    reaper.SetMediaTrackInfo_Value(tr, "I_SOLO", 2)                 -- SIP (§15.2: keeps sends, deliberate)
    set_master_fx_bypass(true)                                      -- step 4
    reaper.GetSet_LoopTimeRange(true, false, p.start_s, p.end_s, false)  -- step 5 (seconds; caller converted via project tempo map)
    set_render_target(p.out_dir, p.basename)                        -- step 6: RENDER_FILE=dir, RENDER_PATTERN=basename,
                                                                    --   WAV 32f, RENDER_SRATE=project, source=master, bounds=time sel
    reaper.Main_OnCommand(42230, 0)                                 -- step 6: render using most recent settings (NEVER 40015)
    local path = p.out_dir .. "/" .. p.basename .. ".wav"
    assert_rendered_nonsilent(path)                                 -- step 7: exists, non-empty, not digital silence
    restore_render_state(snap)                                      -- step 8
    return { path = path, start_s = p.start_s, end_s = p.end_s, sr = get_project_srate() }
  end)
  RENDER_MUTEX_HELD = false
  if not ok then pcall(restore_render_state_best_effort); return err(result) end   -- restore in error path too (§15.2)
  return result
end
```

- [ ] **Step 1: failing tests** — fake-bridge: renders into cache dir, returns `{path,start_s,end_s,sr}`; refuses while fake transport playing; **solo restore** (pre-solo track 2, render track 1, assert track 2 solo restored and track 1 back to 0); master-FX bypass asserted during render + restored; second concurrent call errors "render busy"; silent-render (fake track with no items) → `BridgeError` containing "silent". Lua-side (`test_render_handlers.lua` against the stubbed `reaper`): snapshot/restore of RENDER_* keys, solo-clear loop, 42230 dispatched, error path restores.
- [ ] **Steps 2–4:** red → fake handler + Lua + python tool `render_track_region(track, start_s, end_s)` (beats→seconds conversion documented: callers use `TimeMap2_beatsToTime` via a bridge helper — NEVER the reference song's grid, §15.2 step 5) → green (`pytest` + `run_lua_tests.py`).
- [ ] **Step 5:** Commit `feat(bridge): render_track_region with SIP solo, master-FX bypass, mutex, full state restore (§15.2)`.
- [ ] **Step 6 (HUMAN, desktop, later with 11):** live-REAPER smoke: render a real track region; play the WAV; confirm no render-settings corruption after (open Render dialog, eyeball). Log in dev-log.

---

## Phase 8 — Slice 8: lyrics pipeline + `locate_lyric` (§14) + PIN RUN (c)

### Task 8.1: VAD gate + ASR + alignment + assembly

**Files:** Create `src/orpheus_mcp/analysis/lyrics.py`; Test: `tests/test_lyrics.py`.

**Interfaces produces:**
```python
def vad_windows(vocal: np.ndarray, sr: int) -> list[Window]     # frame RMS > CFG.VAD_VOCAL_ACTIVE_DBFS (§14.1 gate)
class AsrEngine(Protocol):
    def transcribe(self, wav_path: str, windows: list[Window]) -> tuple[str | None, list[LyricWord]]
class FasterWhisperAsr: ...       # large-v3, int8_float16 GPU / int8 CPU, word_timestamps=True
def align_words(words, vocal_wav, language) -> tuple[list[LyricWord], bool]   # whisperX; (words, aligned_flag)
    # no aligner for language -> original timestamps, aligned_flag=False (§14.1.4); stable-ts fallback on ImportError
def assemble(words: list[LyricWord], recon: Recon) -> tuple[list[LyricLine], list[LyricSection]]
    # lines: gap > 1.0 s or sentence punctuation; sections: near-identical repeated line blocks -> chorus candidates,
    # cross-checked vs recon.sections; NO repeats -> structural "section A/B/C" only (§14.1.4 through-composed rule)
def build_lyrics(vocal_pcm: Pcm, recon: Recon, asr: AsrEngine) -> LyricsResult
    # zero VAD windows -> status="no_vocals", empty lists — ASR NEVER runs (§14.1 hallucination gate, BEFORE ASR)
```
- [ ] **Step 1: failing tests** (StubAsr scripted): hallucination gate — silent stem: `status=="no_vocals"` AND StubAsr.transcribe never called (call-counted — the load-bearing assertion); VAD threshold math both sides of −45 dBFS; ASR receives ONLY active windows; line splitting on gaps; repeated-block clustering names `chorus 1/2` with `index` temporal; through-composed input → kinds all `"section"`, names `section A…`; language `None`-aligner path flags coarse timestamps.
- [ ] **Steps 2–4:** red → implement → green (real ASR behind `@pytest.mark.desktop`).
- [ ] **Step 5:** Commit `feat(analysis): §14 lyrics pipeline — VAD-gated ASR + alignment + line/section assembly`.

### Task 8.2: `locate_lyric` resolver + tools + PIN RUN (c)

**Files:** Modify `analysis/lyrics.py`, `tools/reference.py`; Test: extend `tests/test_lyrics.py`, `tests/test_reference_tools.py`.

Locked scorer (§14.2): normalize both sides (casefold, strip punctuation, collapse whitespace); token-level sliding window over the word stream, window len = quote len ± 2; per-token match = exact OR double-metaphone-equal (pure-python implementation in-module, ~40 LOC — English phonetics; non-English → edit-distance-only, noted in meta) OR masked-profanity rule (`f***` matches any token sharing the visible prefix); window score = 1 − levenshtein(token-seq)/max(len); overlapping candidate windows merged keeping best; `occurrence` = temporal index over matches with score ≥ `LYRIC_MATCH_MIN_SCORE`; best < 0.6 → top-3 + `low_confidence` flag (§14.2).

- [ ] **Step 1: failing tests:** exact quote on `ci_vocal` transcript (StubAsr feeds truth lyrics) → score 1.0, right timestamps; deliberate misquote ("the son also rises" vs "the sun also rises") → still top match via metaphone; repeated chorus line → all occurrences, `occurrence` in temporal order regardless of score order; quote spanning a line boundary matches (window slides over words, not lines); `f***` masking case; score 0.4 case → 3 candidates + flag; `locate_lyric` on `no_vocals` file → defined no-vocal result, not empty list (§14.1).
- [ ] **Steps 2–4:** red → implement → green. Tools: `transcribe_lyrics(file)` (builds vocal stem via ladder if not cached — T3 cost, cached once per §3.1; `[separation]` missing ⇒ `capped_by_missing_extra`), `locate_lyric(file, quote, occurrence=None)`. Cache kinds `lyrics/lines/sections` (§9).
- [ ] **Step 5 (HUMAN, desktop):** pin (c): one owned song → vocal stem → transcribe; hand-check 3 quotes incl. one misquote; record WER impressions in pin report. Commit `feat(tools): transcribe_lyrics + locate_lyric (§14.2 fuzzy resolver)`.

---

## Phase 9 — Slice 9: chord model lane + bass refinement (§13 GPU)

### Task 9.1: vendor BTC + rank-2 + model lane integration

**Files:** Create `src/orpheus_mcp/_vendor/btc/` (pinned commit + LICENSE, checkpoint fetched by `fetch-models` per registry); Modify `analysis/harmony.py`, `tools/reference.py`; Test: `tests/test_harmony_models.py`.

**Interfaces produces:** `class ChordModel(Protocol): def predict(self, pcm: Pcm) -> list[ChordSegment]`; `BtcChordModel` (170-class vocab mapped into `ChordSegment` — NO bass, §13.1); `MusicXLabChordModel` (extensions + inversions native, rank-2 cross-check); `chord_lane_model(pcm, recon, models) -> ChordsLane` — model segments re-write baseline at T3 with higher confidence; **bass refinement now prefers the bass STEM when a separation exists** (§13.2), falling back to the low-passed-mix pYIN lane; disagreement between rank-1/rank-2 on a segment → keep rank-1 label, confidence *= 0.7, note in segment (`produced_by="btc+disputed"`).

- [ ] **Steps:** failing tests with stub models (scripted segment streams): re-write semantics (baseline `C`, model `Cmaj7` same span → lane shows `Cmaj7`, `produced_by="btc"`); label-set mapping table (all 14 BTC qualities → our quality strings, exhaustive test); bass-stem preference (both lanes available → stem wins); CPU-only box: `chord_lane_model` unavailable → lane stays baseline, no phantom tier (§13 placement note; asserted via registry resolve with `vram_free_gb=None`). Desktop test: BTC on `ci_sparse` ≥ 80% exact-label (the §12 slice-9 bar; fixture progressions include a maj7 + a slash chord per Task 0.4 truth). **(HUMAN):** pin (b) part 2 — model lane on the same owned songs; record exact-label impressions. Commit `feat(analysis): §13 model chord lane (vendored BTC + musicxlab) with bass-stem slash refinement`.

---

## Phase 10 — Slice 10: `compare_character` + FX intents (§8, §15.3)

### Task 10.1: rules data + engine + tool

**Files:** Create `data/intent_rules.json`, `src/orpheus_mcp/analysis/intents.py`; Modify `tools/reference.py`; Test: `tests/test_intents.py`.

`data/intent_rules.json` — all 14 §15.3 rows verbatim, schema:
```json
{"rule_id": 1, "trigger": {"descriptor": "band_2000_5000_db", "op": "delta_gt", "value": 3.0},
 "intent": "presence_boost",
 "apply": {"plugin": "ReaEQ", "shape": "bell", "freq_hz": 3500, "gain_from": "delta"},
 "explanation": "The reference has {delta:+.1f} dB more presence (2-5 kHz); a bell boost at 3.5 kHz closes it.",
 "vocal_only": false}
```
(rules 2–14 mapped identically from the §15.3 table; rule 11 carries `"time_snap": "project_tempo"` + reports raw ms; rule 13 `vocal_only: true`; rule 14 `"apply": null` + `"requires_opt_in": "master_bus"`; rule 10 carries `"mono_safety_corr_min"` from CFG.)

Trigger `op` vocabulary (locked, exhaustive — `intents.py` rejects any other op at load):
- `delta_gt` / `delta_lt` (signed: reference minus user — rules 1–5's direction words map to sign),
- `abs_delta_gt` (rules 6, 10 — magnitude either way, intent params carry the direction),
- `present_in_a_only` (rule 11: `delay_ms` non-null + `delay_conf ≥ 0.5` in reference, null/low in user),
- `joint` (rules 12, 14: list of sub-triggers, ALL must fire; rule 12 = wet/dry Δ AND decay-class differs).

Locked engine behavior: `diff_characters(a, b)` — resample both to `ANALYSIS_SR`, `band_limit_hz = min(a.source.bandwidth_hz, b.source.bandwidth_hz)`, spectral diffs computed only within it (§8 normalization, MANDATORY); `fire_rules(diff)` evaluates triggers on the diff; `vocal_only` rules skipped with a note when either side's `stem_type` is not `"vocals"` (§15.3 note); `complement=True` sources ⇒ tail-sensitive rules (11, 12) fire as recommend-only with a discount note (§6 R3 flag). **§8 step 2.1 (inventory-first):** the tool takes optional `inventory: list[str] | None` — Claude passes `list_installed_fx`'s result; `match_inventory` fills each `Intent.installed_matches` via a per-intent keyword table in the rules file (`"inventory_keywords": ["eq"], …`) substring-matched case-insensitively, so recommendations name the user's own plugins before stock/catalog. No inventory passed ⇒ empty lists, stock path unchanged.

- [ ] **Step 1: failing tests** — one constructed descriptor-pair per rule (14 tests, exact trigger boundary each: e.g. rule 1 fires at Δ=+3.1, not at +2.9); **the mp3-vs-wav must-NOT-fire test** (§12 slice-10 bar): full-band render vs 16 kHz-limited reference, identical below 16 kHz → rule 3 (air) does NOT fire, `band_limit_hz==16000`, note present; rule 13 skipped+noted on `mix-region`; rule 14 has `apply_path=None`; unknown descriptor in rules file → load-time `RulesError` (fail fast, not silent — this is also the cross-check that every rule's descriptor key exists in Task 6.1's locked key set); inventory matching: `inventory=["TDR Nova (Tokyo Dawn Labs)"]` + a fired EQ intent → `installed_matches` contains it; no inventory → empty.
- [ ] **Steps 2–4:** red → implement → green. Tool `compare_character(a, b) -> CharacterDiff` (§9 contract — intents live HERE, Claude reads them).
- [ ] **Step 5:** Commit `feat(analysis): §15.3 intent rules (data) + band-limited compare_character`.

---

## Phase 11 — Slice 11: FX verbs + param maps + apply loop

### Task 11.1: bridge FX verbs + `data/fx_param_maps.json`

**Files:** Modify `bridge/lua/orpheus_bridge.lua` (handlers `add_fx_by_name`, `set_fx_param`, `get_fx_params`), `tests/fake_reaper.py` (per-track FX param state), `src/orpheus_mcp/tools/mix.py` (un-stub; category already registered), `src/orpheus_mcp/registry.py` (graduate `mix` into `default` if not already); Create `data/fx_param_maps.json`, `tests/lua/test_fx_handlers.lua`; Test: `tests/test_fx_tools.py`.

Locked contracts:
```
add_fx_by_name(track, name)      -> {track, fx_index, already_present}   # TrackFX_AddByName probe-first (idempotent,
                                                                          #  same pattern as add_instrument's fix)
set_fx_param(track, fx_index, param_index, value)  -> {ok}               # value NORMALIZED 0..1 (REAPER native);
get_fx_params(track, fx_index)   -> {fx_name, params: [{index,name,value}]}  # unit/range mapping is python-side
```
`data/fx_param_maps.json`: per plugin (ReaEQ, ReaComp, ReaDelay, ReaVerb, ReaXcomp + 3 JSFX: transient controller, saturation, stereo width) → param name → `{index, unit, min, max, to_normalized: "linear|log"}`. Python-side `mix.py` translates intent params (dB, ms, Hz) → normalized via the map; unmapped plugin ⇒ tool returns the human-instruction path (§8 step 3), never guesses indices.

- [ ] **Steps:** failing fake-bridge + lua tests (idempotent add; set→get round-trip; unmapped plugin name → structured "human instructions" response; ReaEQ dB→normalized conversion against the map) → implement → green → commit `feat(mix): FX verbs (add/set/get) + ReaPlugs/JSFX param maps (§8 step 3)`.
- [ ] **(HUMAN, desktop) live smoke with Task 7:** apply a ReaEQ bell via the map on a real track, `get_fx_params` back, render before/after, confirm the §7 band moved the mapped amount (±1 dB). **Param indices verified live get `"live_verified": true` in the map** — the `<PIN_BEFORE_SHIP>` lesson.

### Task 11.2: the closed loop — fake-bridge convergence test

**Files:** Test: `tests/test_apply_loop.py` (integration; no new src).

- [ ] **Steps:** extend `FakeReaperBridge` render handler to apply its FX state as a real biquad EQ on synthesized track audio (≈40 LOC — the fake renders WAVs whose spectrum reflects `set_fx_param` calls). Test drives the full loop as Claude would: characterize(ref) vs characterize(render) → compare → apply rule-1 intent via mix tools → re-render → re-compare → assert the 2–5 kHz delta shrank monotonically and falls under the rule-1 trigger within `FX_LOOP_MAX_ITERATIONS` (§12 slice-11 bar). Commit `test: closed FX-loop convergence against the fake bridge`.

---

## Phase 12 — Slice 12: orchestrator + catalog + cache tool + docs

### Task 12.1: `analyze_reference` orchestrator

**Files:** Modify `analysis/router.py` (orchestration entry), `tools/reference.py`; Test: `tests/test_orchestrator.py` (golden envelopes).

Locked behavior (§9 contract + §3.1 table): `analyze_reference(file, question_focus=None, region=None, max_tier=None, max_seconds=None) -> AnswerEnvelope`; `question_focus` must be one of the locked `question_class` values (None ⇒ `"recon"`); runs stages up to `min(QUESTION_TIERS[class], max_tier)` by `TIER_ORDER` index, populating ONLY the artifact groups that class needs (§9); missing extra at the needed tier ⇒ answer at highest available + `capped_by_missing_extra` (§9 packaging note); every stage's cache hits/misses and ladder log ride the envelope. Two closures a worker must not re-invent:
- **`lyric_navigate` region math (§14.2):** first call builds the lyric index (T3-scale, cached), then region = `[match.end_s, match.end_s + 8 bars]` where a bar = the reference's measured downbeat spacing from recon — or `8 × CFG.FIXED_GRID_BARS_TO_SECONDS ÷ … = +16 s` when `grid.kind == "fixed"` (§13.4's stated rule); the envelope answers with the timeline diff of that window vs the preceding one (which activations rise).
- **`compare_track` scope:** the envelope delivers the REFERENCE side only (stem + gates + descriptors). The project side (render_track_region → characterize → compare_character) stays separate tool calls composed by Claude — `analysis/` has no bridge access by design (§3 module boundary), and the envelope's `notes` say exactly which calls come next.

- [ ] **Steps:** failing golden tests — one per question class against stub engines + CI fixtures: exact envelope JSON (artifact keys present/absent per class — `"recon"` envelope has NO `timeline` key; `"sound_at"` has stems+gates+descriptors; absent-target question → T2-cost envelope with `notes` containing "target not detected" and NO stems (§3.1c golden); missing-`[separation]` monkeypatch → `capped_by_missing_extra="separation"` envelope (§12 slice-12 bar); `budget_capped` envelope. Implement → green → commit `feat(tools): analyze_reference orchestrator with §9 answer envelope`.

### Task 12.2: `suggest_free_vst` + `clear_analysis_cache` + docs advance

**Files:** Create `data/vst_catalog.json` (§8.1 both tables verbatim — every entry with name, vendor, category, license, `account_required`, url, `intents` list); Modify `tools/reference.py`, `README.md`, `docs/roadmap.md`, `docs/dev-log.md`; Test: `tests/test_vst_catalog.py`.

- [ ] **Steps:** failing tests: catalog parses; every §8.1 row present (count + spot names); account-walled entries flagged (`Vital`, `Decent Sampler`, `Splice`, `Analog Obsession`, `Polyverse Wider` — exact set); `suggest_free_vst("cavern reverb")` returns Valhalla Supermassive in top-3 (intent keyword match over `intents` fields); recommend-only invariant: tool result contains urls, never triggers a download (assert no network call — the tool is pure data lookup). `clear_analysis_cache()` wraps `CacheFS.clear` (destructiveHint). Docs: README status paragraph + roadmap M2/M6 lines advance; dev-log entry per house format. Commit `feat(tools): suggest_free_vst catalog + clear_analysis_cache; docs advance`.

### Task 12.3: full-gate finish line

- [ ] `uv run pytest -q` green (CI subset), `uv run pytest -q -m "desktop"` green on the desktop, `run_lua_tests.py` green, `ruff` + `mypy` clean, CI green on both jobs.
- [ ] Live end-to-end: one owned mp3 → `analyze_reference(question_focus="sound_at", region=...)` → stems + gates + descriptors → `compare_character` vs a rendered track → apply one intent → hear it. Log in dev-log (the M2 "understand" demo).

---

## Cross-cutting invariants (every task inherits)

1. **No network at call time** — any tool test may monkeypatch `socket.socket` to raise; tools must pass. (`fetch-models`/`install_sound_pack` exempt.)
2. **Every artifact JSON carries `schema_version`**; bumping one invalidates via meta compare (Task 1.2 machinery).
3. **Every degradation is named** in tool meta: `capped_by_missing_extra`, VRAM fallback reason, `grid: "fixed"`, `reverb_approx`, `complement`, `low_confidence`, `budget_capped`. A silent downgrade found in review is a task failure.
4. **Stubs-in-full-only** honesty rule for any tool landing before its dependencies.
5. **Heavy imports stay inside functions** (`torch`, `librosa`, engines) — Task 0.1's lean-import test enforces at the package root; tool modules import lazily so `register_tools` skip-on-ImportError keeps working.
6. **Desktop/human tasks are real tasks** — each pin run ends with the pin report updated and committed; an unpinned hash or unverified param index reaching a release tag is the `<PIN_BEFORE_SHIP>` failure mode this repo already learned once.


