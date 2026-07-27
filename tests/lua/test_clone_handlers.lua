-- Lua-side tests for clone_track_items + set_track_params in orpheus_bridge.lua —
-- the artificial-double-tracking primitives (clone a recorded take onto a new track,
-- then detune/nudge/pan the clone; set/read a track's fader volume and pan).
-- Run: lua tests/lua/test_clone_handlers.lua  (also driven in-process via lupa by
-- scripts/run_lua_tests.py).
--
-- Self-contained, same pattern as test_import_handlers.lua: stubs the `reaper` global
-- with an in-memory project modelling tracks, media items, takes, PCM sources (a
-- path -> length registry; an unregistered path yields a nil source on re-open), and
-- track D_VOL/D_PAN. This is the Lua half of the contract; the Python half is
-- tests/test_clone_and_track_params.py against the behavioural fake — both must agree
-- on the {track_guid, track_name, cloned, skipped_midi, failed} shape, the f4394e9
-- tempo-immunity pins, and the one-undo-block promise.

local passed, failed_count = 0, 0
local function ok(cond, msg)
  if cond then passed = passed + 1
  else failed_count = failed_count + 1; io.stderr:write("FAIL: " .. tostring(msg) .. "\n") end
end
local function eq(a, b, msg)
  ok(a == b, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end
local function approx(a, b, msg, tol)
  tol = tol or 1e-9
  ok(type(a) == "number" and math.abs(a - b) < tol,
    (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end

-- --------------------------------------------------------------------------- --
-- In-memory REAPER stub (track / item / take / source / fader surface)
-- --------------------------------------------------------------------------- --

local proj = {
  tracks = {},
  guid_seq = 0,
  -- The "filesystem": path -> { len = seconds }. Anything absent is a file REAPER
  -- cannot re-open -> PCM_Source_CreateFromFile returns nil.
  sources = {
    ["C:/takes/my_take.wav"] = { len = 1.5 },
    ["C:/takes/other.wav"]   = { len = 3.25 },
  },
  undo_begins = 0,
  undo_ends = 0,
  undo_descs = {},
  update_arrange_calls = 0,
  destroyed_sources = 0,
  fail_add_item = false,     -- AddMediaItemToTrack returns nil when true
  fail_add_take = false,     -- AddTakeToMediaItem returns nil when true
}

local function next_guid()
  proj.guid_seq = proj.guid_seq + 1
  return string.format("{LUACLONE-%04d}", proj.guid_seq)
end

local function new_track(name)
  local tr = { guid = next_guid(), name = name or "", items = {}, vol = 1.0, pan = 0.0 }
  proj.tracks[#proj.tracks + 1] = tr
  return tr
end

-- An audio item on `tr`: an active take whose source carries `file` (or no filename
-- when file is nil/"" — the in-project/section-source case).
local function add_audio_item(tr, file, pos, len, startoffs)
  local take = {
    is_midi = false,
    source = { file = file or "" },
    startoffs = startoffs or 0.0,
    playrate = nil,
    vol = nil,
  }
  local item = { track = tr, pos = pos or 0.0, len = len or 1.0,
    beat_attach_mode = nil, takes = { take } }
  take.item = item
  tr.items[#tr.items + 1] = item
  return item
end

local function add_midi_item(tr, pos)
  local take = { is_midi = true, source = nil }
  local item = { track = tr, pos = pos or 0.0, len = 1.0, takes = { take } }
  take.item = item
  tr.items[#tr.items + 1] = item
  return item
end

reaper = {
  GetExtState = function() return "" end,
  GetAppVersion = function() return "7.0/luatest" end,
  RecursiveCreateDirectory = function() end,
  ShowConsoleMsg = function() end,
  defer = function() end,
  EnumerateFiles = function() return nil end,

  CountTracks = function() return #proj.tracks end,
  GetTrack = function(_, i) return proj.tracks[i + 1] end,   -- 0-based in
  GetTrackGUID = function(tr) return tr.guid end,
  GetTrackName = function(tr) return true, tr.name end,
  InsertTrackAtIndex = function(idx)                          -- 0-based
    table.insert(proj.tracks, idx + 1,
      { guid = next_guid(), name = "", items = {}, vol = 1.0, pan = 0.0 })
  end,
  GetSetMediaTrackInfo_String = function(tr, key, val, set)
    if key == "P_NAME" and set then tr.name = val end
    return true, tr.name
  end,
  SetMediaTrackInfo_Value = function(tr, key, val)
    if key == "D_VOL" then tr.vol = val
    elseif key == "D_PAN" then tr.pan = val end
    return true
  end,
  GetMediaTrackInfo_Value = function(tr, key)
    if key == "D_VOL" then return tr.vol end
    if key == "D_PAN" then return tr.pan end
    return 0
  end,

  CountTrackMediaItems = function(tr) return #tr.items end,
  GetTrackMediaItem = function(tr, i) return tr.items[i + 1] end,   -- 0-based in
  GetActiveTake = function(item) return item.takes[1] end,
  TakeIsMIDI = function(take) return take.is_midi == true end,
  GetMediaItemTake_Source = function(take) return take.source end,
  GetMediaSourceFileName = function(src) return src.file or "" end,

  PCM_Source_CreateFromFile = function(file)
    local s = proj.sources[file]
    if not s then return nil end
    return { file = file, len = s.len }
  end,
  PCM_Source_Destroy = function() proj.destroyed_sources = proj.destroyed_sources + 1 end,

  AddMediaItemToTrack = function(tr)
    if proj.fail_add_item then return nil end
    local item = { track = tr, pos = 0, len = 0, takes = {} }
    tr.items[#tr.items + 1] = item
    return item
  end,
  AddTakeToMediaItem = function(item)
    if proj.fail_add_take then return nil end
    local take = { item = item, source = nil }
    item.takes[#item.takes + 1] = take
    return take
  end,
  SetMediaItemTake_Source = function(take, src) take.source = src end,
  SetMediaItemInfo_Value = function(item, key, val)
    if key == "D_POSITION" then item.pos = val
    elseif key == "D_LENGTH" then item.len = val
    elseif key == "C_BEATATTACHMODE" then item.beat_attach_mode = val end
    return true
  end,
  GetMediaItemInfo_Value = function(item, key)
    if key == "D_POSITION" then return item.pos or 0 end
    if key == "D_LENGTH" then return item.len or 0 end
    if key == "C_BEATATTACHMODE" then return item.beat_attach_mode end
    return 0
  end,
  SetMediaItemTakeInfo_Value = function(take, key, val)
    if key == "D_PLAYRATE" then take.playrate = val
    elseif key == "D_STARTOFFS" then take.startoffs = val
    elseif key == "D_VOL" then take.vol = val end
    return true
  end,
  GetMediaItemTakeInfo_Value = function(take, key)
    if key == "D_STARTOFFS" then return take.startoffs or 0.0 end
    if key == "D_PLAYRATE" then return take.playrate or 1.0 end
    return 0
  end,

  UpdateArrange = function() proj.update_arrange_calls = proj.update_arrange_calls + 1 end,
  Undo_BeginBlock = function() proj.undo_begins = proj.undo_begins + 1 end,
  Undo_EndBlock = function(desc, flags)
    proj.undo_ends = proj.undo_ends + 1
    proj.undo_descs[#proj.undo_descs + 1] = { desc = desc, flags = flags }
  end,
}

-- --------------------------------------------------------------------------- --
-- Load the bridge WITHOUT starting the defer loop
-- --------------------------------------------------------------------------- --

_G.ORPHEUS_NO_AUTORUN = true
local here = arg[0]:match("(.*[/\\])") or "./"
local M = assert(loadfile(here .. "../../src/orpheus_mcp/bridge/lua/orpheus_bridge.lua"))()

local function call(fn, params) return M.dispatch(fn, params or {}) end

local function reset()
  proj.tracks = {}
  proj.guid_seq = 0
  proj.undo_begins, proj.undo_ends, proj.undo_descs = 0, 0, {}
  proj.update_arrange_calls = 0
  proj.destroyed_sources = 0
  proj.fail_add_item = false
  proj.fail_add_take = false
end

-- --------------------------------------------------------------------------- --
-- 1. Happy path: two audio items -> ONE new appended track, both items cloned at
--    position + offset_s, D_LENGTH/D_STARTOFFS copied, take volume from
--    item_volume_db, and the f4394e9 tempo-immunity pins on every clone.
-- --------------------------------------------------------------------------- --
reset()
local src_tr = new_track("take")
add_audio_item(src_tr, "C:/takes/my_take.wav", 4.0, 1.5, 0.75)
add_audio_item(src_tr, "C:/takes/other.wav", 10.0, 3.25, 0.0)
local r = call("clone_track_items", {
  src_track = "take", new_track_name = "take (double)",
  offset_s = 0.02, item_volume_db = -6.0,
})
eq(r.ok, true, "clone_track_items ok")
eq(r.result.cloned, 2, "both audio items cloned")
eq(r.result.skipped_midi, 0, "no MIDI takes to skip")
eq(#r.result.failed, 0, "nothing failed")
eq(r.result.track_name, "take (double)", "result names the clone track")

-- 1a. ONE new track, appended at the end, named as asked; result carries its GUID.
eq(#proj.tracks, 2, "exactly one new track created")
local dest = proj.tracks[2]
eq(dest.name, "take (double)", "clone track named from new_track_name")
eq(r.result.track_guid, dest.guid, "result carries the created track GUID")

-- 1b. placement: position + offset_s; length + start offset copied verbatim.
eq(#dest.items, 2, "clone track carries both cloned items")
approx(dest.items[1].pos, 4.02, "item 1 D_POSITION == src + offset_s")
approx(dest.items[1].len, 1.5, "item 1 D_LENGTH copied")
approx(dest.items[1].takes[1].startoffs, 0.75, "item 1 D_STARTOFFS copied")
approx(dest.items[2].pos, 10.02, "item 2 D_POSITION == src + offset_s")
eq(dest.items[1].takes[1].source.file, "C:/takes/my_take.wav",
  "clone re-opens the SAME source file")

-- 1c. f4394e9 tempo-immunity pins: TIME base + native playrate on every clone.
eq(dest.items[1].beat_attach_mode, 0, "item 1 pinned to TIME base (C_BEATATTACHMODE=0)")
eq(dest.items[2].beat_attach_mode, 0, "item 2 pinned to TIME base (C_BEATATTACHMODE=0)")
approx(dest.items[1].takes[1].playrate, 1.0, "take 1 D_PLAYRATE == 1.0 (native rate)")

-- 1d. take volume: item_volume_db -> linear D_VOL (10^(db/20)).
approx(dest.items[1].takes[1].vol, 10.0 ^ (-6.0 / 20.0), "take D_VOL from item_volume_db", 1e-6)

-- 1e. ONE undo block for the WHOLE clone, and UpdateArrange once.
eq(proj.undo_begins, 1, "undo block opened exactly once")
eq(proj.undo_ends, 1, "undo block closed exactly once")
eq(proj.undo_descs[1].desc, "Orpheus: clone 2 items to take (double)",
  "undo description names the clone")
eq(proj.undo_descs[1].flags, -1, "Undo_EndBlock uses flags -1 (all)")
eq(proj.update_arrange_calls, 1, "UpdateArrange called ONCE")

-- --------------------------------------------------------------------------- --
-- 2. Defaults: offset_s -> 0.0, item_volume_db -> 0 dB (unity take volume).
-- --------------------------------------------------------------------------- --
reset()
add_audio_item(new_track("take"), "C:/takes/my_take.wav", 2.0, 1.5, 0.0)
local r2 = call("clone_track_items", { src_track = "take", new_track_name = "dbl" })
eq(r2.ok, true, "default clone ok")
approx(proj.tracks[2].items[1].pos, 2.0, "offset_s defaults to 0.0")
approx(proj.tracks[2].items[1].takes[1].vol, 1.0, "item_volume_db defaults to unity", 1e-9)

-- --------------------------------------------------------------------------- --
-- 3. MIDI takes are SKIPPED silently and counted, never cloned, never an error.
-- --------------------------------------------------------------------------- --
reset()
local tr3 = new_track("mixed")
add_audio_item(tr3, "C:/takes/my_take.wav", 0.0, 1.5, 0.0)
add_midi_item(tr3, 5.0)
local r3 = call("clone_track_items", { src_track = "mixed", new_track_name = "dbl" })
eq(r3.ok, true, "mixed-track clone ok")
eq(r3.result.cloned, 1, "only the audio item cloned")
eq(r3.result.skipped_midi, 1, "the MIDI take is counted under skipped_midi")
eq(#r3.result.failed, 0, "a MIDI take is not a failure")
eq(#proj.tracks[2].items, 1, "clone track carries only the audio clone")

-- --------------------------------------------------------------------------- --
-- 4. PER-ITEM FAILURES are recorded, never silent, and the rest still clone:
--    (a) a source with no filename (in-project/section source);
--    (b) a filename REAPER cannot re-open (PCM_Source_CreateFromFile nil).
-- --------------------------------------------------------------------------- --
reset()
local tr4 = new_track("take")
add_audio_item(tr4, "", 0.0, 1.0, 0.0)                        -- (a) no filename
add_audio_item(tr4, "C:/takes/gone.wav", 1.0, 1.0, 0.0)       -- (b) unopenable
add_audio_item(tr4, "C:/takes/my_take.wav", 2.0, 1.5, 0.0)    -- good
local r4 = call("clone_track_items", { src_track = "take", new_track_name = "dbl" })
eq(r4.ok, true, "per-item failures -> structured result, not a raise")
eq(r4.result.cloned, 1, "the good item still cloned")
eq(#r4.result.failed, 2, "both bad items are reported")
ok(tostring(r4.result.failed[1].error):find("no filename"),
  "no-filename failure carries its reason")
eq(r4.result.failed[1].item_index, 1, "no-filename failure names its 1-based item index")
ok(tostring(r4.result.failed[2].error):find("could not re%-open source file"),
  "unopenable-source failure carries its reason")
eq(#proj.tracks[2].items, 1, "only the good clone landed")
-- The undo block is balanced even on the partial-failure path.
eq(proj.undo_begins, 1, "failure path: undo block opened exactly once")
eq(proj.undo_ends, 1, "failure path: undo block closed exactly once")

-- --------------------------------------------------------------------------- --
-- 5. AddMediaItemToTrack / AddTakeToMediaItem returning nil: per-item failure,
--    and the re-opened source handle is released, not leaked.
-- --------------------------------------------------------------------------- --
reset()
add_audio_item(new_track("take"), "C:/takes/my_take.wav", 0.0, 1.5, 0.0)
proj.fail_add_item = true
local r5 = call("clone_track_items", { src_track = "take", new_track_name = "dbl" })
eq(r5.ok, true, "AddMediaItemToTrack nil -> structured result")
eq(r5.result.cloned, 0, "nothing cloned")
ok(tostring(r5.result.failed[1].error):find("could not create a media item"),
  "media-item failure carries its reason")
eq(proj.destroyed_sources, 1, "the re-opened source handle was destroyed, not leaked")
proj.fail_add_item = false

reset()
add_audio_item(new_track("take"), "C:/takes/my_take.wav", 0.0, 1.5, 0.0)
proj.fail_add_take = true
local r5b = call("clone_track_items", { src_track = "take", new_track_name = "dbl" })
eq(r5b.ok, true, "AddTakeToMediaItem nil -> structured result")
ok(tostring(r5b.result.failed[1].error):find("could not create a take"),
  "take failure carries its reason")
eq(proj.destroyed_sources, 1, "the source handle was destroyed on the take path too")
proj.fail_add_take = false

-- --------------------------------------------------------------------------- --
-- 6. Guards fire BEFORE any mutation: empty src track / missing new_track_name ->
--    clean error, no dest track, no undo block.
-- --------------------------------------------------------------------------- --
reset()
new_track("empty")
local r6 = call("clone_track_items", { src_track = "empty", new_track_name = "dbl" })
eq(r6.ok, false, "empty src track -> ok=false")
ok(tostring(r6.error):find("no items to clone"), "empty-track error message")
eq(#proj.tracks, 1, "no dest track created")
eq(proj.undo_begins, 0, "no undo block was ever opened")

reset()
add_audio_item(new_track("take"), "C:/takes/my_take.wav", 0.0, 1.5, 0.0)
local r6b = call("clone_track_items", { src_track = "take" })
eq(r6b.ok, false, "missing new_track_name -> ok=false")
ok(tostring(r6b.error):find("new_track_name is required"), "missing-name error message")
eq(proj.undo_begins, 0, "missing name: no undo block opened")

reset()
local r6c = call("clone_track_items", { src_track = "ghost", new_track_name = "dbl" })
eq(r6c.ok, false, "unknown src track -> ok=false")
ok(tostring(r6c.error):find("no track named"), "unknown-track error message")

-- --------------------------------------------------------------------------- --
-- 7. set_track_params: sets D_VOL (dB -> linear) and D_PAN (clamped), reads BOTH
--    back after the sets; a 0 linear gain reads back as the -150 dB floor.
-- --------------------------------------------------------------------------- --
reset()
local bus = new_track("bus")
local r7 = call("set_track_params", { track = "bus", volume_db = -6.0, pan = -0.4 })
eq(r7.ok, true, "set_track_params ok")
eq(r7.result.track_guid, bus.guid, "result carries the track GUID")
approx(r7.result.volume_db, -6.0, "volume_db round-trips through linear D_VOL", 1e-9)
approx(r7.result.pan, -0.4, "pan set and read back")
approx(bus.vol, 10.0 ^ (-6.0 / 20.0), "D_VOL holds the linear gain", 1e-9)

-- 7a. omitted fields are untouched (a call with neither is a pure read-back).
local r7a = call("set_track_params", { track = "bus" })
approx(r7a.result.volume_db, -6.0, "read-back call leaves volume untouched", 1e-9)
approx(r7a.result.pan, -0.4, "read-back call leaves pan untouched")

-- 7b. pan is clamped to [-1, 1].
local r7b = call("set_track_params", { track = "bus", pan = 2.5 })
approx(r7b.result.pan, 1.0, "pan clamped high to 1.0")
local r7c = call("set_track_params", { track = "bus", pan = -9.0 })
approx(r7c.result.pan, -1.0, "pan clamped low to -1.0")

-- 7d. a fader at 0 linear gain (-inf dB) reads back as the -150 floor, never -inf.
bus.vol = 0.0
local r7d = call("set_track_params", { track = "bus" })
approx(r7d.result.volume_db, -150.0, "zero linear gain floors at -150 dB")

-- 7e. unknown track is a clean error.
local r7e = call("set_track_params", { track = "ghost", pan = 0.0 })
eq(r7e.ok, false, "unknown track -> ok=false")
ok(tostring(r7e.error):find("no track named"), "unknown-track error message")


-- --------------------------------------------------------------------------- --
-- 8. nudge_take_offset — slide the audio inside items (record-latency repair)
-- --------------------------------------------------------------------------- --
reset()
local vox = new_track("vox")
local it1 = add_audio_item(vox, "C:/takes/my_take.wav", 2.0, 1.5, 0.1)
local r8 = call("nudge_take_offset", { track = "vox", delta_s = 0.26 })
eq(r8.ok, true, "nudge ok")
eq(r8.result.nudged, 1, "one item nudged")
eq(r8.result.clamped, 0, "nothing clamped")
approx(r8.result.delta_s, 0.26, "delta echoed")
approx(it1.takes[1].startoffs, 0.36, "D_STARTOFFS advanced by delta", 1e-9)
approx(it1.pos, 2.0, "item position UNTOUCHED (audio slides, item stays)", 1e-9)
eq(proj.undo_begins, 1, "one undo begin")
eq(proj.undo_ends, 1, "one undo end")
ok(proj.undo_descs[1].desc:find("nudge take offset"), "undo description names the verb")
eq(proj.update_arrange_calls, 1, "one UpdateArrange")

-- 8a. negative resulting offset clamps to 0 and is counted.
local r8a = call("nudge_take_offset", { track = "vox", delta_s = -5.0 })
eq(r8a.result.clamped, 1, "clamp counted")
approx(it1.takes[1].startoffs, 0.0, "offset clamped to zero", 1e-9)

-- 8b. MIDI takes are skipped and counted; audio still nudged.
add_midi_item(vox, 0.0)
local r8b = call("nudge_take_offset", { track = "vox", delta_s = 0.1 })
eq(r8b.result.nudged, 1, "audio item nudged alongside a MIDI item")
eq(r8b.result.skipped_midi, 1, "MIDI take counted, not touched")

-- 8c. guards: missing delta_s; empty track.
local r8c = call("nudge_take_offset", { track = "vox" })
eq(r8c.ok, false, "missing delta_s -> ok=false")
ok(tostring(r8c.error):find("delta_s is required"), "missing-delta message")
new_track("empty")
local r8d = call("nudge_take_offset", { track = "empty", delta_s = 0.1 })
eq(r8d.ok, false, "empty track -> ok=false")
ok(tostring(r8d.error):find("no items to nudge"), "empty-track message")


-- --------------------------------------------------------------------------- --
-- 9. clone_item_region — window a piece of a take onto a new track at dest_s
-- --------------------------------------------------------------------------- --
reset()
local voxr = new_track("vox")
-- item at 140.0, 8 s long, take already slid 0.26 s by a sync nudge
local itr = add_audio_item(voxr, "C:/takes/my_take.wav", 140.0, 8.0, 0.26)
local r9 = call("clone_item_region", { src_track = "vox", new_track_name = "chant",
  src_start_s = 142.9, src_end_s = 149.8, dest_s = 154.3, item_volume_db = -3.0 })
eq(r9.ok, true, "region clone ok")
eq(r9.result.track_name, "chant", "new track named")
approx(r9.result.cloned_start_s, 142.9, "clip start", 1e-9)
approx(r9.result.cloned_end_s, 148.0, "clip end clipped to item end", 1e-9)
approx(r9.result.dest_s, 154.3, "dest as asked", 1e-9)
approx(r9.result.length_s, 5.1, "clip length", 1e-9)
local chant = proj.tracks[#proj.tracks]
eq(chant.name, "chant", "appended track carries the name")
local citem = chant.items[1]
approx(citem.pos, 154.3, "item position at dest", 1e-9)
approx(citem.len, 5.1, "item length = clip length", 1e-9)
approx(citem.takes[1].startoffs, 0.26 + 2.9, "startoffs = take offs + in-item skip", 1e-9)
eq(citem.beat_attach_mode, 0, "f4394e9: TIME base pinned")
approx(citem.takes[1].playrate, 1.0, "f4394e9: native playrate pinned", 1e-9)
approx(citem.takes[1].vol, 10.0 ^ (-3.0 / 20.0), "take volume from item_volume_db", 1e-9)
eq(proj.undo_begins, 1, "one undo begin")
eq(proj.undo_ends, 1, "one undo end")

-- 9a. leading overhang: clip skips to item start, dest shifts by the same amount.
reset()
local vox2 = new_track("vox")
add_audio_item(vox2, "C:/takes/my_take.wav", 10.0, 5.0, 0.0)
local r9a = call("clone_item_region", { src_track = "vox", new_track_name = "clip",
  src_start_s = 8.0, src_end_s = 12.0, dest_s = 50.0 })
approx(r9a.result.cloned_start_s, 10.0, "clip start snapped to item start", 1e-9)
approx(r9a.result.dest_s, 52.0, "dest shifted by the clipped lead", 1e-9)
approx(r9a.result.length_s, 2.0, "clipped length", 1e-9)

-- 9b. guards: no overlap; inverted range; missing dest; unopenable source.
local r9b = call("clone_item_region", { src_track = "vox", new_track_name = "x",
  src_start_s = 100.0, src_end_s = 110.0, dest_s = 0.0 })
eq(r9b.ok, false, "no overlap -> ok=false")
ok(tostring(r9b.error):find("no audio item overlaps"), "no-overlap message")
local r9c = call("clone_item_region", { src_track = "vox", new_track_name = "x",
  src_start_s = 11.0, src_end_s = 11.0, dest_s = 0.0 })
ok(tostring(r9c.error):find("must be > src_start_s"), "inverted-range message")
local r9d = call("clone_item_region", { src_track = "vox", new_track_name = "x",
  src_start_s = 10.5, src_end_s = 11.0 })
ok(tostring(r9d.error):find("are required"), "missing dest_s message")
local vox3 = new_track("ghostsrc")
add_audio_item(vox3, "C:/takes/never_registered.wav", 0.0, 2.0, 0.0)
local r9e = call("clone_item_region", { src_track = "ghostsrc", new_track_name = "x",
  src_start_s = 0.0, src_end_s = 1.0, dest_s = 5.0 })
eq(r9e.ok, false, "unopenable source -> ok=false")
ok(tostring(r9e.error):find("could not re%-open source file"), "re-open failure message")
eq(proj.undo_begins, proj.undo_ends, "undo blocks balanced across failures")

print(string.format("lua clone handlers: %d passed, %d failed", passed, failed_count))
os.exit(failed_count == 0 and 0 or 1)
