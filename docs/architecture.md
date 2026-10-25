# Orpheus Architecture

What Orpheus is and isn't for is settled in
[`specs/2026-10-23-orpheus-definition-design.md`](specs/2026-10-23-orpheus-definition-design.md).
This page covers how it's built. The short version: REAPER is home base. Orpheus reads the live
project through the bridge, and everything it does lands back in the project.

## The shape, in one picture

```
┌──────────────────────────────┐         ┌─────────────────────────────────────────────┐
│   MCP client (Claude, etc.)  │         │                  REAPER                      │
│                              │  stdio  │  ┌────────────────────────────────────────┐ │
│  ┌────────────────────────┐  │ ◄─────► │  │  orpheus_bridge.lua                    │ │
│  │  Orpheus FastMCP server│  │         │  │  persistent reaper.defer() poll loop   │ │
│  │  (external Python proc)│  │         │  │  read request_N.json → static dispatch │ │
│  │                        │  │         │  │  → reaper.* → write response_N.json    │ │
│  │  listen / explain /    │  │         │  │  (+ heartbeat, hot reload of handlers) │ │
│  │  work / theory         │  │         │  └────────────────────────────────────────┘ │
│  └───────────┬────────────┘  │         └───────────────────────▲─────────────────────┘
│              │ BridgeClient   │                                 │
└──────────────┼────────────────┘                                │
               │   file-based JSON IPC (atomic temp-then-rename)  │
               └───────────────────►  <bridge dir>/  ◄────────────┘
                                      request_N.json, response_N.json,
                                      heartbeat.lock
```

The Python process never touches REAPER's API directly. Everything that changes or reads the
project goes through the bridge. Analysis of audio files runs in Python, next to the server, and
reaches REAPER only through bridge verbs such as `render_track_region` and `import_audio_files`.

## Why this bridge

Two transports are proven in the wild by real servers (shiehn, xDarkzx): a file-based JSON channel
and a loopback socket. Orpheus uses the file channel. It needs no dependencies, works on every OS,
and you can debug it by reading the JSON files, which matters because the bridge is the riskiest
piece of infrastructure. A socket transport is a possible future option; only the file bridge is
built and tested.

### Why not OSC
REAPER's OSC interface can only fire predefined actions and cannot pass arguments to custom
actions: `/action/_COMMAND_ID` takes no parameters. So it can't create a track with a name or write
a MIDI note at a pitch. `itsuzef/reaper-mcp` ran into exactly this. OSC is out for construction.

### Why not python-reapy
`python-reapy` is effectively unmaintained (documented REAPER 7 / Python 3.13 connection
failures). It needs REAPER to load a matching Python shared library via `reaper-python.ini`, and
its external "distant API" throttles to about 30 to 60 calls a second. Lua ships inside REAPER, so
an in-REAPER Lua loop has no Python DLL or version matching at all. The most-starred prior art used
reapy over a socket; Orpheus diverges on purpose.

## Bridge hardening

| Requirement | Why |
|---|---|
| **Atomic writes** (write `*.tmp`, then `os.replace`) | The poller must never read a half-written JSON file. This is the critical correctness rule. |
| **Heartbeat lock-file** | The Python side knows the bridge is alive and fails fast with a clear message instead of hanging. |
| **`EnumerateFiles` cache invalidation** (`fileindex=-1`) | REAPER caches directory listings; without invalidation the poller never sees new requests. |
| **Static dispatch** | Requests name a verb in a fixed `HANDLERS` table. Nothing arriving over the channel is ever evaluated. |
| **Per-call caps** | Each command must finish in about 2 s. ReaScript runs on REAPER's main thread, and a long call stutters playback. |
| **Stable identity** (GUIDs/indices, never live pointers) | A `MediaTrack*` pointer can't be serialized across calls and goes stale. Tracks are addressed by GUID, index or name. |
| **Batch ops** | One MCP call is one musical intent. "Create a track and write 64 notes" is a single bridge call; the file channel tops out around 10 ops a second. |
| **Undo blocks** | Verbs added since 2026-07-26 wrap their edits in `Undo_BeginBlock`/`Undo_EndBlock`, so one Ctrl+Z reverts one tool call. Older verbs (`set_tempo`, `insert_midi_notes`, `transpose_notes`, `add_fx_by_name`, `set_fx_param`) don't yet. |
| **Hot reload** | The loop re-reads its own installed script every 2 s and swaps the `HANDLERS` table in place, so `orpheus-mcp install-bridge` delivers new verbs without re-running the action. It loads only that file. The loop, tick and heartbeat core are not swapped; changing those needs one manual re-run. |

The bridge has 37 verbs as of 2026-10-23, plus `__batch__`.

## How a request flows: listen, explain, work

```
  LISTEN (readOnlyHint)             EXPLAIN                      WORK (destructiveHint)
  ┌──────────────────────┐   ┌────────────────────────┐   ┌──────────────────────────────┐
  │ audio you own:        │   │ numbers + confidence    │   │ tracks, tempo, MIDI in beats  │
  │  recon, timeline,     │──►│ → Claude's plain-       │──►│ composers (arrange_song …)    │
  │  chords, stems,       │   │   language answer       │   │ effects (add / set / read)    │
  │  descriptors, lyrics  │   │ compare_character →     │   │ audio import, render region,  │
  │ your project:         │   │   concrete FX moves     │   │ clone / trim, sampler zones   │
  │  MIDI → key, harmony, │   │ (the user approves      │   │                               │
  │  groove               │   │  before anything below) │   │ → re-listen to check          │
  └──────────────────────┘   └────────────────────────┘   └──────────────────────────────┘
```

- **Listen.** Reads never change the project. For audio, analyses are cached under
  `~/.orpheus_cache/<pcm-sha256>/`, keyed by the decoded audio, so the same song in a different
  container hits the cache. Every result is a typed Pydantic model (structured content), with
  confidence and an honest list of what couldn't be measured. `analyze_reference` routes a
  question to the cheapest analysis that can answer it.
- **Explain.** Claude does the explaining; Orpheus supplies measured facts. `compare_character`
  turns the difference between two sounds into effect moves using 14 rules in
  `data/intent_rules.json`.
- **Work.** Every tool that edits the project is annotated `destructiveHint`, so a client can ask
  the user before each call. Claude is told to say what a tool will change and wait for a yes
  (`data/instructions/00_core.md`). After an edit, Orpheus can render the region and listen again,
  which closes the loop.

The approval gate is the MCP tool annotations plus those instructions. Clients enforce it; it is
advisory at the protocol level.

**Later, on the roadmap:** a `LeadSheet` score object between the user's sheet and REAPER (with ABC
export for YuE2), and `apply_changes` running a whole Claude-written `EditPlan` in one undo block.
`recommend_changes` (diffing the project against a style fingerprint) comes after that.

## Heavy engines and optional backends

- **In-process, behind extras:** librosa, PANNs, faster-whisper + whisperX, vendored MSST SCNet
  and BTC chord models. On this desktop the environment is
  `uv sync --locked --extra separation --extra dev` (torch with CUDA).
- **Isolated environments, called as subprocesses:** audio-separator (its dependencies cap
  `fastmcp`) and basic-pitch (needs Python 3.11 or older). Set up by `scripts/setup_*_env.py`.
- **Non-commercial backends:** off unless named in `ORPHEUS_NC_BACKENDS` (for example `yue2`).
  `model_registry.resolve(allow_nc=...)` filters registry entries. YuE2 runs as a separate
  program: `backends/yue2.py` starts yue2.cpp's `yue-synth` CLI once per sketch, with weights the
  user downloads, and holds the GPU semaphore while it runs. `tools/sketch.py` registers
  `render_sketch` only when `ORPHEUS_NC_BACKENDS` names `yue2`.
- **GPU:** one semaphore serializes Orpheus's own GPU stages on the 10 GB card.

## How the modules fit

- `bridge/`: the single point of contact with REAPER (client plus the bundled Lua loop).
- `tools/`: thin FastMCP wrappers grouped by domain, registered via `registry.py`. Profiles:
  `explain` (a smaller set for understanding a project, without the composers or reference
  engine), `default` (54 tools), `full` (adds 11 stubs).
- `analysis/`: pure functions that don't need REAPER (recon, timeline, harmony, separation and its
  verification gates, descriptors, lyrics, intents, section-diff, cache, model registry).
  Unit-testable with no REAPER.
- `theory/`: scales, chords, cadences, melody notation, drum patterns. A read-only knowledge layer.
- `notation/`: the owner's chord charts and riff tab, lead sheets, and YuE2's ABC dialect (writer,
  checker, reader).
- `backends/`: optional external generators (YuE2 through yue2.cpp), each behind its own opt-in.
- `models.py`: the Pydantic contracts (`Recon`, `ChordsLane`, `LyricsResult`, `SoundCharacter`,
  `CompositionSpec`, `EditPlan`, …) that become structured content.
- `data/`: calibrated FX parameter maps, the model registry, intent rules, the free-VST catalog,
  fingerprints, and the instructions file. Shipped inside the wheel as `orpheus_mcp/data`.

Composers are thin orchestrators over the same public tools Claude can call directly. They get no
private shortcuts.

## Testing

- `tests/fake_reaper.py` is a behavioural fake of REAPER (tracks, items, takes, tempo, markers, FX,
  the same 960-PPQ math) and the executable spec for the Lua handlers.
- `scripts/run_lua_tests.py` runs the real `orpheus_bridge.lua` handlers against a stubbed
  `reaper` table, in-process via lupa.
- `-m desktop` and `-m gpu` tests need pinned checkpoints or CUDA and are deselected in CI.
- Live checks against real REAPER are recorded in `docs/pins/` and the dev-log.
