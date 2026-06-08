# Orpheus Architecture

## The shape, in one picture

```
┌──────────────────────────────┐         ┌─────────────────────────────────────────────┐
│   MCP client (Claude, etc.)  │         │                  REAPER                      │
│                              │  stdio  │  ┌────────────────────────────────────────┐ │
│  ┌────────────────────────┐  │ ◄─────► │  │  orpheus_bridge.lua                    │ │
│  │  Orpheus FastMCP server│  │         │  │  persistent reaper.defer() poll loop   │ │
│  │  (external Python proc)│  │         │  │  ~100ms: read request_N.json →         │ │
│  │                        │  │         │  │  STATIC dispatch → reaper.* → write     │ │
│  │  analyze / recommend / │  │         │  │  response_N.json → delete request       │ │
│  │  apply / theory / ...  │  │         │  └────────────────────────────────────────┘ │
│  └───────────┬────────────┘  │         └───────────────────────▲─────────────────────┘
│              │ BridgeClient   │                                 │
└──────────────┼────────────────┘                                │
               │   file-based JSON IPC (atomic temp-then-rename)  │
               └───────────────────►  <bridge dir>/  ◄────────────┘
                                      request_1.json, response_1.json,
                                      heartbeat.lock
```

The Python process **never** touches REAPER's API directly. Everything goes through one bridge.

## Why this bridge

Two transports are validated in the wild by real servers (shiehn, xDarkzx): a **file-based JSON channel** and a loopback socket. Orpheus uses the **file channel** as the primary path because it is dependency-free, cross-platform, and *debuggable by `cat`-ing the JSON* — which matters enormously for a project where the bridge is the riskiest infrastructure. (A loopback-socket transport is a possible future option, not a v1 promise — only the file bridge is implemented and tested.)

### Why not OSC
REAPER's OSC interface can only fire predefined actions and **cannot pass arguments to custom actions** — so `/action/_COMMAND_ID` takes no parameters. It literally cannot create a track with a name or write a MIDI note at a pitch. `itsuzef/reaper-mcp` proves this the hard way. OSC is out for construction.

### Why not python-reapy
`python-reapy` is effectively unmaintained (documented REAPER 7 / Python 3.13 connection failures), requires REAPER to load a matching Python shared library via `reaper-python.ini`, and its external "distant API" throttles to ~30–60 calls/sec. The in-REAPER Lua loop sidesteps all of it — Lua ships *inside* REAPER, so there's no Python-DLL/version matching at all. This is a deliberate divergence from the most-starred prior art (which used reapy-over-socket); the tradeoff is intentional and defensible.

## Bridge hardening (day-one requirements)

These aren't polish — they're the difference between "works in the demo" and "hangs mysteriously":

| Requirement | Why |
|---|---|
| **Atomic writes** (write `*.tmp`, then `os.replace`) | The poller must never read a half-written JSON file. This is *the* critical correctness rule, more than the heartbeat. |
| **Heartbeat lock-file** | So the Python side knows REAPER/the bridge is alive and fails fast with a clear message instead of hanging. |
| **`EnumerateFiles` cache invalidation** (`fileindex=-1`) | REAPER caches directory listings; the Lua poller must invalidate or it never sees new requests. |
| **Static dispatch** (no `loadstring`/`dofile`) | Safety — never `eval` arbitrary strings arriving over the channel. |
| **Per-call note/track caps** | Each command must finish under ~2s; REAPER's ReaScript runs on the single-threaded audio path and a long call stutters playback. |
| **Stable identity (GUIDs/indices), never live pointers** | A `MediaTrack*` userdata pointer cannot be serialized across calls and goes stale. Address tracks/takes by GUID or index. |
| **Composite/batch ops** | One MCP round-trip = one musical intent. "create track + write 64 notes" must be a *single* bridge call — the file channel ceiling is ~10 ops/sec, so per-note round-trips are death. |

## The spine: analyze → recommend → apply

This loop is the differentiator. The universal contract between its stages is the **`CompositionSpec`** (see [`models.py`](../src/orpheus_mcp/models.py)).

```
                          ┌─────────────── theory toolset (read-only) ───────────────┐
                          │  music21 + genre tables keep proposed notes in-key       │
                          └──────────────────────────┬──────────────────────────────┘
                                                     │
  ANALYZE (readOnlyHint)          RECOMMEND (readOnlyHint)        APPLY (destructiveHint)
  ┌───────────────────┐           ┌────────────────────┐         ┌─────────────────────┐
  │ export MIDI →      │          │ diff Spec(current)  │         │ execute approved     │
  │   music21 harmony  │   Spec   │   vs style          │  Edit   │   EditPlan in ONE    │
  │ raw PPQ → groove   │ ───────► │   fingerprint       │ ──Plan► │   undo block:        │
  │ render → librosa   │ current  │ → EditPlan with a   │         │   transpose, rewrite │
  │   audio character  │          │   reason per change │   ▲     │   MIDI, swap FX,     │
  │ → build_project_   │          │ (no mutation)       │   │     │   Matchering master  │
  │   spec + report    │          └─────────┬──────────┘   │     └──────────┬──────────┘
  └───────────────────┘                     │              │                │
                                            ▼         you approve           ▼
                                   "here's why" prose ──────┘        render_and_audit
                                   surfaced to the user             (verify vs reference)
```

- **ANALYZE** builds a structured `CompositionSpec(current)` and an "LLM-Readable Music Report" (facts, not raw numbers — AI TrackMate's pattern). Symbolic understanding is cheap (export MIDI, run music21, filter drums first). Audio understanding renders stems/master and runs librosa + four objective DSP analyses.
- **RECOMMEND** loads a cached **style fingerprint** (the same pipeline run over 3–5 per-era reference tracks), diffs, and emits an `EditPlan` — a list of `ProposedEdit{target, action, reason, params}`. **Read-only.** It decides; it never mutates.
- **APPLY** takes the *exact* approved `EditPlan` and executes it through the bridge inside one `Undo_BeginBlock`/`Undo_EndBlock` so the whole transform is a single `Ctrl+Z`.

The approval gate is enforced by **MCP tool annotations** (`readOnlyHint` vs `destructiveHint`), not bolted on — compliant clients can auto-allow reads and force confirmation on the destructive apply.

## How the modules fit

- `bridge/` — the single point of contact with REAPER (client + bundled Lua loop).
- `tools/` — thin FastMCP wrappers, grouped by domain, registered via `registry.py` and toolset-gateable.
- `analysis/` — **protocol-independent pure functions** (music21 symbolic, librosa audio, fingerprint diffing, Matchering). Unit-testable with zero REAPER.
- `theory/` — reimplemented scales/chords/cadences + ported genre profiles. A read-only knowledge oracle.
- `models.py` — the Pydantic contract (`CompositionSpec`, `EditPlan`, …) that becomes `structuredContent` for free.

Composers and the NL DSL are **thin orchestrators over the same public primitives** the agent can call directly — they never get private superpowers.
