-- Lua-side tests for the REAPER-sync read verbs (spec 2026-10-23 REAPER sync §2.1, §2.3):
-- get_project_timeline and get_project_snapshot.
-- Run: lua tests/lua/test_sync_handlers.lua (or scripts/run_lua_tests.py, via lupa).
--
-- The stub project: 120 BPM (1 QN = 0.5 s), bars 1-4 in 4/4, 3/4 from bar 5. So bar 5 starts
-- at QN 16 (8.0 s), bar 6 at QN 19 (9.5 s), bar 9 at QN 28 (14.0 s).

local passed, failed = 0, 0
local function ok(cond, msg)
  if cond then passed = passed + 1
  else failed = failed + 1; io.stderr:write("FAIL: " .. tostring(msg) .. "\n") end
end
local function eq(a, b, msg)
  ok(a == b, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end
local function approx(a, b, msg, tol)
  ok(a ~= nil and math.abs(a - b) < (tol or 1e-6),
    (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end

local BPS = 2.0  -- quarter notes per second at 120 BPM
local proj = {
  tempo = 120.0,
  meter = { { bar = 1, num = 4, den = 4 }, { bar = 5, num = 3, den = 4 } },
  -- (time, bpm, num, den, linear); num/den 0 = inherit, as REAPER reports it
  tempo_markers = {
    { t = 0.0, bpm = 120.0, num = 4, den = 4, linear = false },
    { t = 8.0, bpm = 120.0, num = 3, den = 4, linear = false },
    { t = 12.5, bpm = 90.0, num = 0, den = 0, linear = true },
  },
  markers = {
    { isrgn = false, pos = 9.5, name = "Verse 2", idx = 1, color = 0 },
    { isrgn = true, pos = 8.0, rgnend = 14.0, name = "Bridge", idx = 1,
      color = 0x1000000 | (0xFF << 16) },  -- custom colour flag + pure blue (native BGR)
  },
  name = "song.rpp", path = "C:/proj/song.rpp", length = 30.0, changes = 42,
  play_state = 1, cursor = 3.0, play_pos = 3.5,
  tracks = {
    { guid = "{T1}", name = "Keys", vol = 0.5, pan = -0.25, mute = 1, solo = 0, fx = 2,
      items = { { guid = "{I1}", pos = 0.0, len = 8.0, midi = true, hash = "h1" } } },
    { guid = "{T2}", name = "Vox", vol = 1.0, pan = 0.0, mute = 0, solo = 2, fx = 0,
      items = { { guid = "{I2}", pos = 9.5, len = 4.0, midi = false, file = "C:/x/vox.wav" },
                { guid = "{I3}", pos = 20.0, len = 1.0, empty = true } } },
  },
}

local function bar_len(bar)
  local num, den = 4, 4
  for _, m in ipairs(proj.meter) do if bar >= m.bar then num, den = m.num, m.den end end
  return num * 4 / den, num, den
end
local function bar_start(bar)  -- QN
  local qn = 0
  for b = 1, bar - 1 do qn = qn + bar_len(b) end
  return qn
end

local TMPDIR = (os.getenv("TEMP") or os.getenv("TMPDIR") or "/tmp"):gsub("\\", "/")
proj.now = 100.0  -- reaper.time_precise(), wall-clock seconds

reaper = {
  GetExtState = function(_, b) if b == "bridge_dir" then return TMPDIR end return "" end,
  time_precise = function() return proj.now end,
  GetAppVersion = function() return "7.0/luatest" end,
  RecursiveCreateDirectory = function() end,
  ShowConsoleMsg = function() end,
  defer = function() end,
  EnumerateFiles = function() return nil end,

  Master_GetTempo = function() return proj.tempo end,
  -- the meter of the bar holding t, and the tempo of the last tempo marker at or before t
  TimeMap_GetTimeSigAtTime = function(_, t)
    local qn, bar = (t or 0) * BPS, 1
    while bar_start(bar + 1) <= qn + 1e-9 do bar = bar + 1 end
    local _, num, den = bar_len(bar)
    local bpm = proj.tempo
    for _, m in ipairs(proj.tempo_markers) do if m.t <= (t or 0) + 1e-9 then bpm = m.bpm end end
    return num, den, bpm
  end,
  TimeMap2_QNToTime = function(_, qn) return qn / BPS end,
  TimeMap2_timeToQN = function(_, t) return t * BPS end,
  TimeMap2_beatsToTime = function(_, tpos, measures)
    return (bar_start((measures or 0) + 1) + (tpos or 0)) / BPS
  end,
  -- (beats in measure, measures (0-based), beats per measure, full beats, denominator)
  TimeMap2_timeToBeats = function(_, t)
    local qn, bar = t * BPS, 1
    while bar_start(bar + 1) <= qn + 1e-9 do bar = bar + 1 end
    local _, num, den = bar_len(bar)
    return (qn - bar_start(bar)) * den / 4, bar - 1, num, qn, den
  end,
  CountTempoTimeSigMarkers = function() return #proj.tempo_markers end,
  GetTempoTimeSigMarker = function(_, i)
    local m = proj.tempo_markers[i + 1]
    if not m then return false end
    return true, m.t, 0, 0, m.bpm, m.num, m.den, m.linear
  end,
  EnumProjectMarkers3 = function(_, i)
    local m = proj.markers[i + 1]
    if not m then return 0 end
    return i + 1, m.isrgn, m.pos, m.rgnend or 0.0, m.name, m.idx, m.color
  end,
  ColorFromNative = function(c) return c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF end,
  EnumProjects = function() return "PROJ", proj.path end,
  GetProjectName = function() return proj.name end,
  GetProjectStateChangeCount = function() return proj.changes end,
  GetProjectLength = function() return proj.length end,
  GetPlayState = function() return proj.play_state end,
  GetCursorPosition = function() return proj.cursor end,
  GetPlayPosition = function() return proj.play_pos end,

  CountTracks = function() return #proj.tracks end,
  GetTrack = function(_, i) return proj.tracks[i + 1] end,
  GetTrackGUID = function(tr) return tr.guid end,
  GetTrackName = function(tr) return true, tr.name end,
  GetMediaTrackInfo_Value = function(tr, key)
    if key == "D_VOL" then return tr.vol end
    if key == "D_PAN" then return tr.pan end
    if key == "B_MUTE" then return tr.mute end
    if key == "I_SOLO" then return tr.solo end
    return 0
  end,
  TrackFX_GetCount = function(tr) return tr.fx end,
  CountTrackMediaItems = function(tr) return #tr.items end,
  GetTrackMediaItem = function(tr, i) return tr.items[i + 1] end,
  GetSetMediaItemInfo_String = function(item, key) if key == "GUID" then return true, item.guid end end,
  GetMediaItemInfo_Value = function(item, key)
    if key == "D_POSITION" then return item.pos end
    if key == "D_LENGTH" then return item.len end
    return 0
  end,
  GetActiveTake = function(item) if item.empty then return nil end return item end,
  TakeIsMIDI = function(take) return take.midi end,
  MIDI_GetHash = function(take) return true, take.hash end,
  GetMediaItemTake_Source = function(take) return take end,
  GetMediaSourceFileName = function(src) return src.file end,
}

local here = (arg and arg[0] or ""):match("^(.*[/\\])") or "./"
_G.ORPHEUS_NO_AUTORUN = true
local M = assert(loadfile(here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"))()
local function call(fn, params) return M.dispatch(fn, params or {}) end

-- 1. get_project_timeline: identity, transport, tempo map, markers, regions.
local r = call("get_project_timeline")
eq(r.ok, true, "timeline ok")
local tl = r.result
eq(tl.project.name, "song.rpp", "project name")
eq(tl.project.path, "C:/proj/song.rpp", "project path")
approx(tl.project.length_s, 30.0, "project length")
eq(tl.project.state_change_count, 42, "state change count")
eq(tl.transport.playing, true, "playing")
eq(tl.transport.recording, false, "not recording")
approx(tl.transport.cursor_s, 3.0, "edit cursor")
approx(tl.transport.play_position_s, 3.5, "play position")

eq(#tl.tempo_map, 3, "three tempo/time-signature markers")
eq(tl.tempo_map[2].bar, 5, "meter change at bar 5")
approx(tl.tempo_map[2].beat, 1.0, "on beat 1")
approx(tl.tempo_map[2].qn, 16.0, "QN 16")
eq(tl.tempo_map[2].meter[1], 3, "3/4 numerator")
eq(tl.tempo_map[2].meter[2], 4, "3/4 denominator")
approx(tl.tempo_map[3].bpm, 90.0, "tempo change to 90")
eq(tl.tempo_map[3].meter[1], 3, "a 0/0 meter on a tempo marker inherits 3/4")
eq(tl.tempo_map[3].linear, true, "linear ramp flag")
eq(tl.tempo_map[3].bar, 8, "12.5 s is bar 8 on the 4/4 -> 3/4 map")

eq(#tl.markers, 1, "one marker")
eq(tl.markers[1].name, "Verse 2", "marker name")
eq(tl.markers[1].bar, 6, "9.5 s is bar 6")
approx(tl.markers[1].beat, 1.0, "beat 1")
eq(tl.markers[1].color, nil, "no custom colour -> nil")
eq(#tl.regions, 1, "one region")
eq(tl.regions[1].start_bar, 5, "region starts bar 5")
eq(tl.regions[1].end_bar, 9, "region ends at bar 9 (QN 28)")
eq(tl.regions[1].color, "#0000FF", "custom colour as hex")
eq(tl.truncated, false, "not truncated")

-- 1b. A project with no tempo markers still reports its tempo and meter at time 0.
local saved = proj.tempo_markers
proj.tempo_markers = {}
local t0 = call("get_project_timeline").result.tempo_map
eq(#t0, 1, "synthesized single tempo entry")
eq(t0[1].bar, 1, "at bar 1")
approx(t0[1].bpm, 120.0, "project tempo")
eq(t0[1].meter[1], 4, "project meter")
proj.tempo_markers = saved

-- 2. get_project_snapshot: tracks, items (with MIDI hash / audio source), markers, tempo map.
local s = call("get_project_snapshot")
eq(s.ok, true, "snapshot ok")
local snap = s.result
eq(snap.project.state_change_count, 42, "snapshot carries the change count")
eq(#snap.tracks, 2, "two tracks")
local keys, vox = snap.tracks[1], snap.tracks[2]
eq(keys.guid, "{T1}", "track guid")
eq(keys.index, 1, "1-based index")
eq(keys.mute, true, "mute")
eq(vox.solo, true, "solo (any non-zero I_SOLO)")
approx(keys.volume_db, -6.0206, "volume in dB", 1e-3)
approx(keys.pan, -0.25, "pan")
eq(keys.fx_count, 2, "fx count")
eq(keys.items[1].kind, "midi", "midi item")
eq(keys.items[1].midi_hash, "h1", "midi hash")
eq(keys.items[1].bar, 1, "item bar")
eq(vox.items[1].kind, "audio", "audio item")
eq(vox.items[1].source, "C:/x/vox.wav", "audio source")
eq(vox.items[1].bar, 6, "audio item at bar 6")
eq(vox.items[2].kind, "empty", "item without a take")
eq(#snap.markers, 1, "markers in the snapshot")
eq(#snap.regions, 1, "regions in the snapshot")
eq(#snap.tempo_map, 3, "tempo map in the snapshot")
eq(snap.truncated, false, "snapshot not truncated")

-- 2b. get_bar_time (spec 2026-10-24 sketch §3.5): where a bar starts on REAPER's map, with the
-- tempo and meter there. Bar 9 is after the 3/4 change (QN 28, 14.0 s) and the 90 BPM marker.
local bt = call("get_bar_time", { bar = 9 })
eq(bt.ok, true, "get_bar_time ok")
eq(bt.result.bar, 9, "bar echoed")
approx(bt.result.time_s, 14.0, "bar 9 starts at 14.0 s")
approx(bt.result.qn, 28.0, "bar 9 starts at QN 28")
approx(bt.result.bpm, 90.0, "tempo at bar 9")
eq(bt.result.meter[1], 3, "meter numerator at bar 9")
eq(bt.result.meter[2], 4, "meter denominator at bar 9")
approx(call("get_bar_time", { bar = 1 }).result.time_s, 0.0, "bar 1 starts at 0")
eq(call("get_bar_time", { bar = 0 }).ok, false, "bar 0 is refused")
eq(call("get_bar_time", {}).ok, false, "a missing bar is refused")

-- 3. Heartbeat v2 (spec §2.4): JSON with project identity, written every tick, refreshed about
-- once a second; liveness stays the file's modification time.
local HB = TMPDIR .. "/heartbeat.lock"
local function read_hb()
  local f = assert(io.open(HB, "rb")); local t = f:read("*a"); f:close(); return t
end
M.touch_heartbeat()
local hb = read_hb()
ok(hb:find('"v":2', 1, true), "heartbeat v2 marker: " .. hb)
ok(hb:find('"project_name":"song.rpp"', 1, true), "project name in heartbeat")
ok(hb:find('"project_path":"C:/proj/song.rpp"', 1, true), "project path in heartbeat")
ok(hb:find('"playing":true', 1, true), "playing in heartbeat")
ok(hb:find('"state_change_count":42', 1, true), "change count in heartbeat")
ok(hb:find('"reaper_version":"7.0/luatest"', 1, true), "REAPER version in heartbeat")
proj.name = "other.rpp"
proj.now = proj.now + 0.3
M.touch_heartbeat()
ok(read_hb():find('"project_name":"song.rpp"', 1, true), "within a second: cached content")
proj.now = proj.now + 1.0
M.touch_heartbeat()
ok(read_hb():find('"project_name":"other.rpp"', 1, true), "after a second: refreshed")
os.remove(HB)

print(string.format("lua sync handlers: %d passed, %d failed", passed, failed))
os.exit(failed == 0 and 0 or 1)
