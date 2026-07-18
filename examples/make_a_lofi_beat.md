# Walkthrough: "Make me a lofi beat" — `compose_section` demo

> **Pending live run.** This is the *intended* flow once REAPER + the Orpheus bridge are
> connected — it has not yet been run against a live REAPER session. `compose_section` is
> implemented and verified against the `FakeReaperBridge` contract tests and the lupa-driven Lua
> handler suite (167 passed / 2 skipped in the full test suite; 44 Lua assertions; see
> [`docs/dev-log.md`](../docs/dev-log.md), 2026-07-18 entry, for exactly what's pending: the
> `EnumInstalledFX` return shape, the `ReaSamplOmatic5000` `FILE0`/note-range param indices, the
> `MIDI_DeleteNote` descending-delete idiom, and — most importantly — whether the section is
> actually *audible*). Nobody has pressed play on this yet. Treat everything below as the
intended script, not a recording of a real session.

### Setup (once REAPER is connected)
1. Open REAPER, run the Orpheus bridge action (`orpheus_bridge.lua`).
2. In Claude: *"Check the Orpheus connection."* → expect `{ok: true, reaper_version: '7.x/...'}`.

### The conversation (intended)

> **You:** Make me a lofi beat.

Behind the scenes the agent calls the one-shot orchestrator:

```
compose_section(genre="lofi", bars=4)
```

`compose_section` resolves the `"lofi"` genre profile (tempo range, typical mode, a stock
progression), then:

1. `set_tempo` — picks the midpoint of the genre's BPM range (lofi's range sits inside 60–90).
2. `list_installed_fx` — reads the user's installed-plugin inventory once, up front.
3. Creates three tracks: `drums`, `chords`, `bass`.
4. `drums` — writes a one-bar kick/snare/hat backbeat, then loads an instrument for the role:
   the user's own drum plugin if one is installed (MT-PowerDrumKit / Battery / EZdrummer /
   Superior Drummer / Kontakt, matched by substring against the inventory), otherwise the stock
   3-voice kit (three `ReaSamplOmatic5000` instances, each pinned to its GM note, using the
   stdlib-synthesized kick/snare/hat one-shots).
5. `chords` — voice-leads the genre's progression across the requested bars, writes it as block
   chords, then loads an instrument for the `keys` role (the user's Vital/Surge XT/Pianoteq/
   Kontakt/Decent Sampler if installed, else stock ReaSynth).
6. `bass` — writes root-note bass following the same progression one octave down, loads the
   `bass`-role instrument (same ladder, stock ReaSynth if nothing installed matches).

### Expected result shape (what the tool returns)

```json
{
  "genre": "lofi",
  "tempo": 75.0,
  "key": "A",
  "tracks": ["drums", "chords", "bass"],
  "instruments": {
    "drums": {"kind": "drumkit", "name": null, "source": "stock"},
    "chords": {"kind": "named", "name": "ReaSynth", "source": "stock"},
    "bass":   {"kind": "named", "name": "ReaSynth", "source": "stock"}
  }
}
```

(The exact tempo/key/instrument sources depend on the live genre profile and whatever plugins
are actually installed on the machine — `"source": "installed"` instead of `"stock"` wherever a
curated plugin from the allowlist is found in `list_installed_fx`'s output.)

> **You:** (in REAPER) press play.

**This is the step that has not been verified.** Once live-confirmed, this doc should be updated
to say what was actually heard, any tempo/instrument corrections needed, and whether one
`Ctrl+Z` cleanly reverts the whole section (per Task 15, step 2 — wrap the orchestrator's writes
in an undo block if the live test shows more than one undo step).

### Why this depends on a live REAPER session

Every claim above the "press play" line is proven by the fake bridge and the Lua stub suite —
tracks get created, notes land at the right beats, instrument-load calls are made with the
right arguments. None of it proves the result actually makes sound: that requires
`EnumInstalledFX` to behave as assumed, the RS5k config-parm names to be correct on a live
instance, and a human ear at the end of the chain. Until that live run happens, `compose_section`
should be read as "programmatically correct, not yet heard."
