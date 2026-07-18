# 🎵 Orpheus

**An AI agent that doesn't just build music in REAPER — it listens to what you've made, tells you *why* it sounds the way it does, and reshapes it toward the sound you want.**

> *"Why does this beat sound like Dominic Fike?"* · *"Make this sound Classical."*
> Orpheus analyzes your project, recommends concrete changes **with reasons**, and — on your approval — applies them as real, editable tracks. One `Ctrl+Z` undoes the whole thing.

> 🚧 **Status: early (pre-alpha), and moving.** What's **built + tested** today: the full architecture, a [documented analysis of the entire Reaper/DAW-MCP frontier](docs/frontier-analysis.md), the music-theory core, the **REAPER bridge** (M0 — connect to a live REAPER and round-trip commands), the **construction core** (M1): track listing/creation, transport (tempo/meter/play-stop-record), and the load-bearing **PPQ-correct MIDI writer** (`insert_midi_notes` / `get_track_midi` / `transpose_notes`) — the model speaks beats, all tick math lives in the bridge, and a note written at beat B reads back at beat B (guarded by the round-trip gate) — and now a first slice of **M4 GENERATE**: `compose_section` turns a genre + bar count into an editable, audible section in one call (tempo, drums/chords/bass tracks, a genre-appropriate groove), backed by four atomic composers (`create_chord_progression`, `create_bassline`, `create_drum_pattern`, `humanize_pass`) and a deterministic instrument-selection ladder that **prefers the instruments you already have installed** (`list_installed_fx` was pulled forward from the mix pass to make that possible) before falling back to stock ReaSynth / a synthesized drum kit. All proven by Python + Lua + cross-language tests (167 passed, 2 skipped; 44 Lua assertions; ruff + mypy clean) against a behavioural REAPER fake — **live-REAPER verification of this slice is still pending** (see [`docs/dev-log.md`](docs/dev-log.md)), no real REAPER has confirmed the compose output plays. What's **still stubs**: FX/mix verbs beyond `list_installed_fx`, and the analyze/recommend/apply tools (M2–M3 on the [roadmap](docs/roadmap.md)). So it connects, builds, and modifies a project correctly and can now generate a section from scratch, but doesn't yet analyze or transform-toward-a-reference. The first feature release (`v0.1`) ships the **understand-and-explain** half; the **transform** half follows in `v0.3`. Star/watch to follow along. ⭐

Orpheus is an [MCP](https://modelcontextprotocol.io) server. It plugs into any MCP-compatible client (Claude Desktop, Cursor, Claude Code) and gives the model a set of tools to read, reason about, and edit a live [REAPER](https://www.reaper.fm/) session.

---

## Why Orpheus exists

There is a thriving ecosystem of MCP servers that let an AI **build** music in a DAW by natural language. After reading the source of the leading projects (see [`docs/frontier-analysis.md`](docs/frontier-analysis.md)), one gap is unmistakable:

> **The "build me a track" half is solved and even viral. The "look at my track, explain it, and improve it toward a reference" half is completely unclaimed.**

No shipping server can look at your *existing* project, reason over **both its composition and its sound**, propose reasoned changes toward a target style, and apply them back as **editable** tracks. That loop —

```
ANALYZE  →  RECOMMEND (with reasons)  →  [you approve]  →  APPLY  →  VERIFY
```

— is Orpheus's entire reason to exist.

### How it's different from what's out there

| Capability | Control servers (shiehn, xDarkzx, Ableton-MCP) | Analysis servers (dschuler36) | Mix-match tools (Matchering, Ozone) | Neural "genre changers" | **Orpheus** |
|---|:---:|:---:|:---:|:---:|:---:|
| Build a track from NL | ✅ | — | — | — | ✅ |
| Read the *current* project's harmony | — | partial (MIDI-blind) | — | — | ✅ |
| Explain *why* it sounds like X | — | objective warnings only | — | — | ✅ |
| Recommend changes **with reasons** | — | — | — | — | ✅ |
| Human approval gate | — | n/a | — | — | ✅ |
| Apply changes as **editable** tracks | fire-and-forget | — | audio only | ❌ (baked WAV) | ✅ |
| Match harmony / instrumentation, not just EQ | — | — | — | ✅ (unexplained) | ✅ |

Orpheus is the first to make a single `CompositionSpec` the contract for an **analyze → recommend → apply** loop over an existing project. The compositional dimensions (key, harmony, instrumentation) are where it's *genuinely first*; the mix/master dimension it assembles from solved parts (it vendors [Matchering](https://github.com/sergree/matchering)).

---

## The north-star demo

```
You:      (open an 8-bar beat in REAPER) "Make this sound Classical."

Orpheus:  Analyzing… detected A minor (confidence 0.71), 120 BPM, a i–iv vamp,
          drum-forward mix, low band +8 dB hotter than the classical reference.

          Here's my plan (each change, and why):
            1. 120 → ~70 BPM        — classical phrasing breathes slower
            2. transpose to A minor — keep your tonal center, lean modal
            3. i–iv  →  ii–V–i with a secondary dominant — functional harmony is the
                                                            classical signature
            4. add a Strings track   (from your installed FX) — idiomatic voice
            5. master-match tonal balance to the reference — cut ~3 dB at 120 Hz

          Apply all 5?  [you approve]

Orpheus:  Done — wrapped in one undo block. Re-rendered and re-measured:
          now within 1.2 dB of the reference across all three bands.
```

Every line of "here's why" is a real diff between **your project** and a cached **style fingerprint** — not a vibe.

---

## Architecture (one paragraph)

Orpheus is an external Python [FastMCP](https://gofastmcp.com) server. It **never** calls REAPER's API directly. Instead it speaks to a single hardened **file-based JSON bridge** watched by a persistent Lua [ReaScript](https://www.reaper.fm/sdk/reascript/reascript.php) loop *inside* REAPER — the same dependency-free transport the most comprehensive existing server ships, hardened with a heartbeat lock-file, atomic writes, static dispatch, and per-call note caps. The model talks in **beats**, never ticks; all PPQ/tempo math lives inside the bridge. Read-only `analyze_*` tools build a `CompositionSpec` of your current project; `recommend_changes` diffs it against a style fingerprint into a reason-annotated `EditPlan`; a separate, `destructiveHint`-flagged `apply_changes` executes the approved plan in one undo block. Full detail: [`docs/architecture.md`](docs/architecture.md).

> **Why not `python-reapy` or OSC?** REAPER's OSC can't pass arguments to custom actions, so it [literally cannot create tracks or write MIDI notes](docs/architecture.md#why-not-osc). `python-reapy` is effectively unmaintained (documented REAPER 7 / Python 3.13 failures) and its distant API throttles to ~30–60 calls/sec. The in-REAPER Lua bridge sidesteps both. This is a deliberate divergence from the most-starred prior art — see [the architecture doc](docs/architecture.md) for the full justification.

---

## Install

> Not yet published. These are the **planned** one-line install paths once `v0.1` is on PyPI.

```bash
# 1. Run the server (no clone needed, once published)
uvx orpheus-mcp

# 2. Load the in-REAPER bridge script (one-time)
#    Copies orpheus_bridge.lua into REAPER's Scripts folder and tells you how to run it.
orpheus-mcp install-bridge
```

Then add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "orpheus": { "command": "uvx", "args": ["orpheus-mcp"] }
  }
}
```

Restart Claude Desktop, open a project in REAPER, run the Orpheus bridge action, and ask Claude to *"check the Orpheus connection."* Full per-OS steps: [`docs/installation.md`](docs/installation.md).

---

## Roadmap

Orpheus ships in honest milestones (full detail + the "why this order" in [`docs/roadmap.md`](docs/roadmap.md)):

| Milestone | What works | Scale |
|---|---|---|
| **M0** ✅ | Hardened bridge + FastMCP scaffold + `get_connection_status` | foundation |
| **M1** ✅ | Build/modify correctly: tracks, transport, PPQ-correct MIDI (FX verbs stubbed) | construction core |
| **M2** | **Understand** a project: harmony, groove, audio character, theory scaffolding | → `v0.1`: *it explains your track* |
| **M3** | **Transform**: `recommend_changes` + gated `apply_changes` + style fingerprints | → `v0.3`: *the differentiator* |
| **M4** | NL ergonomics + generate-from-scratch composers | polish |
| **M5** | Docs, PyPI, MCP Registry, the launch demo | ship |
| **M6** | Reach: MIDI recording, audio→reference ingest, groove transfer | post-launch |

**v0.1 (weeks) explains; v0.3 (months) transforms.** The full loop is a real build, scoped honestly.

---

## Standing on shoulders — credits

Orpheus deliberately aggregates the best of a generous open-source ecosystem rather than reinventing it. Full per-project analysis with citations in [`docs/frontier-analysis.md`](docs/frontier-analysis.md).

- **[shiehn/total-reaper-mcp](https://github.com/shiehn/total-reaper-mcp)** — the file-JSON IPC bridge pattern + NL DSL resolvers + tool profiles.
- **[xDarkzx/Reaper-MCP](https://github.com/xDarkzx/Reaper-MCP)** — bridge hardening (heartbeat, static dispatch, per-call caps).
- **[ahujasid/ableton-mcp](https://github.com/ahujasid/ableton-mcp)** — beats-not-ticks note model, atomic-tool altitude, the demo-led launch playbook.
- **[dschuler36/reaper-mcp-server](https://github.com/dschuler36/reaper-mcp-server)** — the typed project tree + four objective audio analyses (the seed of the recommend engine).
- **[jarmstrong158/waveform-MCP](https://github.com/jarmstrong158/waveform-MCP)** — the `CompositionSpec` IR, primitives-vs-composers split, mix-calibration table (patterns reimplemented under MIT, not copied — its code is GPL-3.0).
- **[bonfire-audio/reaper-mcp](https://github.com/bonfire-audio/reaper-mcp)** — chord/drum theory primitives + correct headless render recipes.
- **[music21](https://music21.org)**, **[asume21/music-theory-mcp](https://github.com/asume21/music-theory-mcp)** — the theory + genre knowledge layer.
- **[Matchering](https://github.com/sergree/matchering)** — the entire mix/master "sound like" layer.
- **[AI TrackMate](https://arxiv.org/pdf/2412.06617)** — the analyze → "LLM-readable music report" → reasoned-feedback pattern.

---

## License

[MIT](LICENSE) © 2026 Mal0ss. Orpheus reimplements *patterns* from the projects above; it does not copy GPL-licensed source.
