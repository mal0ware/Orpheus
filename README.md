# 🎵 Orpheus

Claude's ears and hands inside REAPER.

Orpheus is an [MCP](https://modelcontextprotocol.io) server that connects Claude (Claude Code,
Claude Desktop, or any MCP client) to a live [REAPER](https://www.reaper.fm/) session.

It listens to your project and to audio you own: tempo, key, chords, sections, stems, lyrics
and sound character, each with an honest confidence. It explains what it hears in plain
language. It works in your session: it builds tracks, writes MIDI, arranges songs, and adds and
sets effects. It asks before it changes anything, and every change can be undone.

It's built for one songwriter-producer. Lyrics live in a Google Sheet, Claude reads them, Orpheus
turns them into a song in REAPER, and you record, mix and refine with Claude's help. It makes
editable sketches and helps; you make the record.

> **Status (2026-10-23):** early, pre-alpha, not on PyPI yet. It has 54 tools in the default profile,
> 787 Python tests and 368 Lua checks (ruff and mypy clean). Much of it was used for real in July
> on REAPER 7.78 with a song the owner owns; the rest is tested against a behavioural fake of
> REAPER. [What works today](#what-works-today) says which is which.

## What you can ask

- *"What key and tempo is this song, and where does the chorus start?"*
- *"Split the vocals out of this mp4 I made and put the stems on tracks."*
- *"Where in the song does the hook come back?"*
- *"Here are my lyrics and chords from the sheet. Lay out the song: verse, pre-chorus, chorus, bridge."*
- *"Add a bass line and a half-time drum part under the verse."*
- *"Make the outro vocal sound like it's coming through a telephone."*

## What works today

### Listen

All local, on audio you own:
- tempo, beat grid, downbeats, key, loudness and sections (`recon_reference`);
- an instrument-activity timeline (`instrument_timeline`);
- chords (`get_chords`): an always-on baseline plus an optional model lane;
- stem separation with quality gates and a retry ladder (`separate_region`, `verify_stem`);
- sound character: spectrum, dynamics, stereo, reverb, pitch (`characterize`);
- lyrics with word timings, and "find where this line is sung" (`transcribe_lyrics`, `locate_lyric`);
- comparing two sections of a song that should match (`compare_sections`);
- one entry point that routes a question to the right analysis and reports what it could and
  couldn't measure (`analyze_reference`).

It also reads your own project's MIDI for key, harmony and groove (`build_project_spec`,
`analyze_harmony`, `analyze_groove`).

### Explain

Analyses return measured numbers with confidence values, which lets Claude explain why
something sounds the way it does (for example "the low band is 6 dB hotter than the
reference"). `compare_character` turns the difference between two sounds into concrete effect moves, using
14 rules.

### Work in the session

- Tracks, tempo, meter, transport, and MIDI written in beats, never ticks (a note written at beat B
  reads back at beat B).
- Composing: chord progressions, bass lines, drum patterns, melodies, humanizing, whole sections,
  and `arrange_song` for a full sectioned song with markers and lyric lines.
- Effects: add by name, set and read parameters, using calibrated maps for REAPER's stock plugins
  and Surge XT.
- Audio: import your stems as time-locked tracks, render a track region to WAV, and clone a take
  onto new tracks (for doubles).
- Instruments: Orpheus prefers instruments you have installed, and `suggest_free_vst` recommends
  free ones.

Fifteen more bridge commands were built and used live in July but don't have MCP tools yet:
renaming, colouring, moving and deleting tracks; regions; automation envelopes; trimming, windowing and
nudging items; factory presets; refreshing media and peaks; and instruments built from your own
samples.

### Verified on real REAPER

In July 2026, on REAPER 7.78: the bridge, the effect maps, region rendering,
stem import, item cloning, those fifteen bridge commands, hot reload, and analysis of an owned
200-second song (separation, lyrics, sections). The full-song composer and the chord and lyric
quality still need a live check by ear. The [dev-log](docs/dev-log.md) has the details.

## How a song gets made

1. You write lyrics, sections and chord charts in your Google Sheet.
2. Claude reads the sheet through the Google Drive connector. Orpheus never signs into Google;
   it only gets text.
3. Orpheus lays the song out in REAPER: section markers, chords, bass, drums, a melody, and your
   lyric lines on the markers.
4. You record. Orpheus analyses the take, and Claude suggests changes. You approve each one.

Reading the chord-chart and riff-tab notation from the sheet directly, a lead-sheet format, and
sung sketches are being built now (see [the roadmap](docs/roadmap.md)).

## What it won't do

- Fetch, rip or capture music you don't own. Analysis runs only on files you give it.
- Need an account, a subscription or a paid API. Everything runs on your machine.
- Master a track for release, or replace your ears.
- Change your project without asking first.

## Not built yet

- Lead sheets and notation: reading your chord charts and bass tab, and ABC export/import.
- Sung sketches through an optional YuE2 backend (below).
- Beat-map export for cutting music videos in Premiere (OpenTimelineIO).
- Edit plans applied in one undo step (`apply_changes`) and style-based suggestions
  (`recommend_changes`). Both are stubs, registered only in the `full` profile.
- Master rendering and audit (`render_project`, `render_and_audit`) and mix balancing.

## Optional: non-commercial backends

Some useful models allow only non-commercial use: YuE2 (sung song sketches), LarsNet (drum
splitting). Orpheus is MIT, so these are off by default. You switch each one on by name:

```json
"env": { "ORPHEUS_NC_BACKENDS": "yue2" }
```

Their weights are never bundled. You download them yourself, under their own licence (CC BY-NC
4.0 for YuE2), and Orpheus talks to them as a separate program.

## Install

> Not published yet. These are the planned install paths once `v0.1` is on PyPI.

```bash
# 1. Run the server
uvx orpheus-mcp

# 2. Copy the bridge script into REAPER's Scripts folder (one-time)
orpheus-mcp install-bridge
```

Then add it to your MCP client, for example Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "orpheus": { "command": "uvx", "args": ["orpheus-mcp"] }
  }
}
```

Open a project in REAPER, run the Orpheus bridge action, and ask Claude to *"check the Orpheus
connection."* Per-OS steps: [`docs/installation.md`](docs/installation.md). Working from source:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## How it works

Orpheus is a Python [FastMCP](https://gofastmcp.com) server. It never calls REAPER's API
directly. It writes JSON request files that a Lua script running *inside* REAPER picks up,
executes and answers. The bridge is hardened: atomic writes, a heartbeat, a fixed table of
commands (nothing evaluated from the wire), per-call caps, and hot reload of new commands without
restarting REAPER. Claude talks in beats; all tick and tempo math happens in the bridge. Tools that
change the project are marked `destructiveHint`, so your client can ask you before each one. Engines with
clashing dependencies (audio-separator, basic-pitch) run in their own environments as separate
processes. More: [`docs/architecture.md`](docs/architecture.md).

> **Why not `python-reapy` or OSC?** REAPER's OSC can't pass arguments to custom actions, so it
> [can't create tracks or write MIDI notes](docs/architecture.md#why-not-osc). `python-reapy` is
> effectively unmaintained and throttles to about 30 to 60 calls a second. A Lua loop inside REAPER
> avoids both.

## Credits

Orpheus reimplements patterns from these projects under MIT; it does not copy GPL-licensed code.
Per-project notes: [`docs/frontier-analysis.md`](docs/frontier-analysis.md).

- [shiehn/total-reaper-mcp](https://github.com/shiehn/total-reaper-mcp): the file-JSON bridge pattern and tool profiles.
- [xDarkzx/Reaper-MCP](https://github.com/xDarkzx/Reaper-MCP): bridge hardening (heartbeat, static dispatch, per-call caps).
- [ahujasid/ableton-mcp](https://github.com/ahujasid/ableton-mcp): the beats-not-ticks note model and tool granularity.
- [dschuler36/reaper-mcp-server](https://github.com/dschuler36/reaper-mcp-server): the typed project tree and objective audio checks.
- [jarmstrong158/waveform-MCP](https://github.com/jarmstrong158/waveform-MCP): the `CompositionSpec` idea and the primitives-vs-composers split (patterns only; its code is GPL-3.0).
- [bonfire-audio/reaper-mcp](https://github.com/bonfire-audio/reaper-mcp): chord and drum primitives, headless render recipes.
- [music21](https://music21.org) and [asume21/music-theory-mcp](https://github.com/asume21/music-theory-mcp): the theory layer.
- [AI TrackMate](https://arxiv.org/pdf/2412.06617): the pattern of analysing, writing a readable report, then giving feedback.
- [Matchering](https://github.com/sergree/matchering) (GPL-3.0): the likely engine for future master matching. It will be an optional install, not a dependency.

Model credits and licences for the analysis engines are in
[`data/model_registry.json`](data/model_registry.json) and [`docs/pins/`](docs/pins/).

## License

[MIT](LICENSE) © 2026 Mal0ss.
