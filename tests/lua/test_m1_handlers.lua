-- Lua-side tests for the M1 handlers in orpheus_bridge.lua.
-- Run: lua tests/lua/test_m1_handlers.lua
--
-- Self-contained (no shell / no real REAPER): stubs the `reaper` global with an
-- in-memory project that models tracks, MIDI items/takes, tempo, and the same 960
-- PPQ-per-quarter-note math REAPER uses, then drives the handlers through M.dispatch.
-- This is the Lua half of the M1 contract; tests/fake_reaper.py is the Python half,
-- and both must agree on the beats<->PPQ conversion (the load-bearing invariant).

local passed, failed = 0, 0
local function ok(cond, msg)
  if cond then passed = passed + 1
  else failed = failed + 1; io.stderr:write("FAIL: " .. tostring(msg) .. "\n") end
end
local function eq(a, b, msg)
  ok(a == b, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end
local function approx(a, b, msg)
  ok(math.abs(a - b) < 1e-6, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end

local PPQ = 960  -- REAPER default ticks per quarter note

-- --------------------------------------------------------------------------- --
-- In-memory REAPER stub
-- --------------------------------------------------------------------------- --

local proj = {
  tempo = 120.0,
  ts_num = 4,
  ts_den = 4,
  play_state = 0,
  tracks = {},   -- array of { guid, name, vol, pan, mute, solo, items = { {take=...} } }
  guid_seq = 0,
}

local function next_guid()
  proj.guid_seq = proj.guid_seq + 1
  return string.format("{LUA-%04d}", proj.guid_seq)
end

reaper = {
  GetExtState = function(a, b) if a == "orpheus" and b == "bridge_dir" then return "/tmp/x" end return "" end,
  GetAppVersion = function() return "7.0/luatest" end,
  RecursiveCreateDirectory = function() end,
  ShowConsoleMsg = function() end,
  defer = function() end,
  EnumerateFiles = function() return nil end,

  CountTracks = function() return #proj.tracks end,
  GetTrack = function(_, i) return proj.tracks[i + 1] end,  -- 0-based in
  GetTrackGUID = function(tr) return tr.guid end,
  GetTrackName = function(tr) return true, tr.name end,
  GetSetMediaTrackInfo_String = function(tr, key, val, set)
    if key == "P_NAME" and set then tr.name = val end
    return true, tr.name
  end,
  GetMediaTrackInfo_Value = function(tr, key)
    if key == "D_VOL" then return tr.vol or 1.0 end
    if key == "D_PAN" then return tr.pan or 0.0 end
    if key == "B_MUTE" then return tr.mute and 1 or 0 end
    if key == "I_SOLO" then return tr.solo and 1 or 0 end
    return 0
  end,
  InsertTrackAtIndex = function(idx)  -- 0-based
    local tr = { guid = next_guid(), name = "", vol = 1.0, pan = 0.0, items = {} }
    table.insert(proj.tracks, idx + 1, tr)
  end,
  CountTrackMediaItems = function(tr) return #tr.items end,
  -- Real signature: GetTrackMediaItem(MediaTrack tr, integer itemidx) — NO project arg
  -- (unlike GetTrack). A stub that tolerated a leading 0 masked exactly that bug.
  GetTrackMediaItem = function(tr, i) return tr.items[i + 1] end,  -- 0-based
  GetActiveTake = function(item) return item.take end,
  GetMediaItemInfo_Value = function(item, key)
    if key == "IP_ITEMNUMBER" then return item.index end
    return 0
  end,

  Master_GetTempo = function() return proj.tempo end,
  SetCurrentBPM = function(_, bpm) proj.tempo = bpm end,
  -- Real return order: (timesig_num, timesig_denom, tempo).
  TimeMap_GetTimeSigAtTime = function() return proj.ts_num, proj.ts_den, proj.tempo end,
  -- (proj, ptidx, timepos, measurepos, beatpos, bpm, timesig_num, timesig_denom, linear)
  SetTempoTimeSigMarker = function(_, _, _, _, _, _, num, den) proj.ts_num = num; proj.ts_den = den end,
  GetProjectLength = function() return 0.0 end,
  GetPlayState = function() return proj.play_state end,
  Main_OnCommand = function(id)
    if id == 1007 then proj.play_state = 1
    elseif id == 1016 then proj.play_state = 0
    elseif id == 1013 then proj.play_state = 5 end
  end,

  -- Time/QN map: linear (constant tempo). 1 QN == 1 second-equivalent here; only the
  -- QN<->PPQ relationship matters for the round-trip, and that is exact.
  TimeMap2_QNToTime = function(_, qn) return qn end,
  CreateNewMIDIItemInProj = function(tr, start_t, _end_t)
    local take = { notes = {}, item_start_qn = start_t }
    local item = { take = take, index = #tr.items }
    take.item = item
    table.insert(tr.items, item)
    return item
  end,
  MIDI_GetPPQPosFromProjQN = function(_, qn) return qn * PPQ end,
  MIDI_GetProjQNFromPPQPos = function(_, ppq) return ppq / PPQ end,
  MIDI_InsertNote = function(take, sel, muted, sppq, eppq, chan, pitch, vel)
    take.notes[#take.notes + 1] =
      { sel = sel, muted = muted, sppq = sppq, eppq = eppq, chan = chan, pitch = pitch, vel = vel }
  end,
  MIDI_CountEvts = function(take) return true, #take.notes, 0, 0 end,
  MIDI_GetNote = function(take, i)
    local n = take.notes[i + 1]
    return true, n.sel, n.muted, n.sppq, n.eppq, n.chan, n.pitch, n.vel
  end,
  MIDI_SetNote = function(take, i, sel, muted, sppq, eppq, chan, pitch, vel)
    local n = take.notes[i + 1]
    n.sel, n.muted, n.sppq, n.eppq, n.chan, n.pitch, n.vel = sel, muted, sppq, eppq, chan, pitch, vel
    return true
  end,
  MIDI_DeleteNote = function(take, i)
    table.remove(take.notes, i + 1)
    return true
  end,
  MIDI_Sort = function(take)
    table.sort(take.notes, function(a, b)
      if a.sppq == b.sppq then return a.pitch < b.pitch end
      return a.sppq < b.sppq
    end)
  end,

  EnumInstalledFX = function(i)
    local fx = { "VSTi: ReaSynth (Cockos)", "VSTi: Surge XT" }
    if fx[i + 1] then return true, fx[i + 1], "ident" end
    return false
  end,

  -- FX chain: tr.fx = {} array of FX names, in add order.
  TrackFX_AddByName = function(tr, name, rec, mode)
    tr.fx = tr.fx or {}
    if mode == 0 then  -- find-only
      for i, n in ipairs(tr.fx) do if n == name then return i - 1 end end
      return -1
    end
    tr.fx[#tr.fx + 1] = name
    return #tr.fx - 1
  end,
  TrackFX_SetNamedConfigParm = function() return true end,
  TrackFX_SetParamNormalized = function() return true end,

  -- (proj, isrgn, pos, rgnend, name, wantidx, color) -> marker index.
  AddProjectMarker2 = function(_, _, _, _, _, wantidx) return wantidx == -1 and 1 or wantidx end,
}

-- --------------------------------------------------------------------------- --
-- Load the bridge WITHOUT starting the defer loop
-- --------------------------------------------------------------------------- --

_G.ORPHEUS_NO_AUTORUN = true
local here = arg[0]:match("(.*[/\\])") or "./"
local M = assert(loadfile(here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"))()

local function call(fn, params)
  local r = M.dispatch(fn, params or {})
  return r
end

-- 1. get_project_info reflects tempo + meter
local r = call("get_project_info")
eq(r.ok, true, "get_project_info ok")
eq(r.result.tempo, 120.0, "project tempo")
eq(r.result.time_signature[1], 4, "ts numerator")

-- 2. create_track returns a stable GUID and shows up in list_tracks
local t = call("create_track", { name = "Bass" })
eq(t.ok, true, "create_track ok")
ok(t.result.guid:sub(1, 1) == "{", "create_track returns a GUID")
eq(t.result.name, "Bass", "create_track name")
local guid = t.result.guid

local lt = call("list_tracks")
eq(lt.ok, true, "list_tracks ok")
eq(#lt.result, 1, "one track listed")
eq(lt.result[1].name, "Bass", "listed track name")

-- 3. set_tempo / set_time_signature / play_stop_record
eq(call("set_tempo", { bpm = 72 }).result.tempo, 72, "set_tempo applied")
local ts = call("set_time_signature", { numerator = 3, denominator = 4 })
eq(ts.result.time_signature[1], 3, "set_time_signature numerator")
eq(call("play_stop_record", { command = "play" }).result.play_state, 1, "play -> playing")
eq(call("play_stop_record", { command = "stop" }).result.play_state, 0, "stop -> stopped")
eq(call("play_stop_record", { command = "rewind" }).ok, false, "bad transport command -> ok=false")

-- reset meter to 4/4 for the MIDI math below
call("set_time_signature", { numerator = 4, denominator = 4 })

-- 4. THE INVARIANT: insert notes in beats, read them back in beats unchanged.
local notes = {
  { pitch = 60, start_beat = 0.0, duration_beats = 1.0, velocity = 100 },
  { pitch = 64, start_beat = 1.5, duration_beats = 0.5, velocity = 80 },
}
local ins = call("insert_midi_notes", { track = guid, notes = notes })
eq(ins.ok, true, "insert_midi_notes ok")
eq(ins.result.inserted, 2, "inserted 2 notes")

local back = call("get_track_midi", { track = guid })
eq(back.ok, true, "get_track_midi ok")
eq(#back.result.notes, 2, "read back 2 notes")
eq(back.result.notes[1].pitch, 60, "note1 pitch round-trips")
approx(back.result.notes[1].start_beat, 0.0, "note1 start_beat round-trips")
approx(back.result.notes[1].duration_beats, 1.0, "note1 duration round-trips")
eq(back.result.notes[2].pitch, 64, "note2 pitch round-trips")
approx(back.result.notes[2].start_beat, 1.5, "note2 fractional start_beat round-trips")
approx(back.result.notes[2].duration_beats, 0.5, "note2 fractional duration round-trips")

-- 5. at_bar is a relative anchor: written at bar 3, read with the same anchor == beat 0.
local t2 = call("create_track", { name = "Lead" }).result.guid
call("insert_midi_notes", { track = t2, at_bar = 3,
  notes = { { pitch = 67, start_beat = 0.0, duration_beats = 2.0, velocity = 90 } } })
local same = call("get_track_midi", { track = t2, at_bar = 3 })
approx(same.result.notes[1].start_beat, 0.0, "at_bar anchor: same anchor reads beat 0")
local from1 = call("get_track_midi", { track = t2, at_bar = 1 })
approx(from1.result.notes[1].start_beat, 8.0, "at_bar anchor: bar 3 is 8 beats from bar 1")

-- 6. transpose shifts pitch, preserves timing; out-of-range notes untouched.
call("transpose_notes", { track = guid, semitones = -2 })
local tr = call("get_track_midi", { track = guid })
eq(tr.result.notes[1].pitch, 58, "transpose -2: 60 -> 58")
eq(tr.result.notes[2].pitch, 62, "transpose -2: 64 -> 62")
approx(tr.result.notes[1].start_beat, 0.0, "transpose preserves note1 timing")

-- 6b. clear_track_midi deletes every note in the first take.
do
  local cleared = call("clear_track_midi", { track = guid })
  eq(cleared.ok, true, "clear_track_midi ok")
  eq(cleared.result.cleared, 2, "clear_track_midi reports 2 notes cleared")
  local after = call("get_track_midi", { track = guid })
  eq(#after.result.notes, 0, "clear_track_midi empties the take")
end

-- 7. unknown track -> ok=false
eq(call("insert_midi_notes", { track = "{NOPE}", notes = {} }).ok, false, "unknown track -> ok=false")

-- 8. __batch__ over M1 handlers stays one round-trip
local b = call("__batch__", { calls = {
  { fn = "create_track", params = { name = "Drums" } },
  { fn = "set_tempo", params = { bpm = 90 } },
} })
eq(b.ok, true, "batch ok")
eq(#b.result, 2, "batch returns 2 results")
eq(b.result[2].tempo, 90, "batch second call applied tempo")

-- 9. list_installed_fx enumerates via EnumInstalledFX until it returns false
do
  local r = M.dispatch("list_installed_fx", {})
  eq(r.ok, true, "list_installed_fx ok")
  eq(#r.result.fx, 2, "list_installed_fx returns both installed")
end

-- 10. add_instrument: kind="drumkit" adds three ReaSamplOmatic5000 instances
do
  local dguid = call("create_track", { name = "Drums2" }).result.guid
  local r = call("add_instrument", { track = dguid, kind = "drumkit",
    samples = { kick = "a", snare = "b", hat = "c" } })
  eq(r.ok, true, "add_instrument drumkit ok")
  eq(r.result.loaded, "drumkit", "drumkit loaded")
  local raw
  for _, tr in ipairs(proj.tracks) do
    if tr.guid == dguid then raw = tr end
  end
  eq(#raw.fx, 3, "three samplers added")
end

-- 11. add_instrument: kind="named" is idempotent
do
  local kguid = call("create_track", { name = "Keys" }).result.guid
  local first = call("add_instrument", { track = kguid, kind = "named", name = "ReaSynth" })
  eq(first.ok, true, "add_instrument named ok")
  eq(first.result.already_present, false, "first add_instrument not already present")
  local second = call("add_instrument", { track = kguid, kind = "named", name = "ReaSynth" })
  eq(second.result.already_present, true, "second add_instrument already present")
end

-- 12. add_marker places a named marker and returns {name, bar, index}
do
  local r = M.dispatch("add_marker", { name = "Verse", bar = 3 })
  eq(r.ok, true, "add_marker ok")
  eq(r.result.name, "Verse", "marker name")
  eq(r.result.bar, 3, "marker bar")
end

print(string.format("lua m1 handlers: %d passed, %d failed", passed, failed))
os.exit(failed == 0 and 0 or 1)
