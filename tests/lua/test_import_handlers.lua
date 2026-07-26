-- Lua-side tests for import_audio_files in orpheus_bridge.lua — the opt-in local-audio
-- import verb (drag-and-drop, programmatically).
-- Run: lua tests/lua/test_import_handlers.lua  (also driven in-process via lupa by
-- scripts/run_lua_tests.py).
--
-- Self-contained, same pattern as test_render_handlers.lua: stubs the `reaper` global with
-- an in-memory project modelling tracks, media items, takes, and a PCM-source registry
-- (path -> length in seconds; an unregistered path yields a nil source, exactly as
-- PCM_Source_CreateFromFile does for a file REAPER cannot open). This is the Lua half of
-- the contract; tests/test_import_audio.py is the Python half against the behavioural
-- fake — both must agree on the {imported, failed} shape and the one-undo-block promise.

local passed, failed_count = 0, 0
local function ok(cond, msg)
  if cond then passed = passed + 1
  else failed_count = failed_count + 1; io.stderr:write("FAIL: " .. tostring(msg) .. "\n") end
end
local function eq(a, b, msg)
  ok(a == b, (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end
local function approx(a, b, msg)
  ok(type(a) == "number" and math.abs(a - b) < 1e-9,
    (msg or "") .. " (got " .. tostring(a) .. ", want " .. tostring(b) .. ")")
end

-- --------------------------------------------------------------------------- --
-- In-memory REAPER stub (track / item / take / PCM-source surface)
-- --------------------------------------------------------------------------- --

local proj = {
  tracks = {},
  guid_seq = 0,
  -- The "filesystem": path -> { len = seconds, is_qn = false }. Anything absent is a
  -- file REAPER cannot open -> PCM_Source_CreateFromFile returns nil.
  sources = {
    ["C:/stems/vocals.wav"] = { len = 12.5 },
    ["C:/stems/drums.wav"]  = { len = 30.25 },
    ["/stems/bass.flac"]    = { len = 7.0 },
    ["C:/stems/empty.wav"]  = { len = 0.0 },   -- zero-length -> clean per-file error
  },
  undo_begins = 0,
  undo_ends = 0,
  undo_descs = {},
  update_arrange_calls = 0,
  destroyed_sources = 0,
  calls = {},           -- every source/item/take API name, in order
  fail_update_arrange = false,
}

local function record(name) proj.calls[#proj.calls + 1] = name end

local function next_guid()
  proj.guid_seq = proj.guid_seq + 1
  return string.format("{LUAIMP-%04d}", proj.guid_seq)
end

local function called(name)
  for _, c in ipairs(proj.calls) do if c == name then return true end end
  return false
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
    table.insert(proj.tracks, idx + 1, { guid = next_guid(), name = "", items = {} })
  end,
  GetSetMediaTrackInfo_String = function(tr, key, val, set)
    if key == "P_NAME" and set then tr.name = val end
    return true, tr.name
  end,

  -- PCM source: nil for an unknown path (the file REAPER cannot open).
  PCM_Source_CreateFromFile = function(file)
    record("PCM_Source_CreateFromFile")
    local s = proj.sources[file]
    if not s then return nil end
    return { file = file, len = s.len, is_qn = s.is_qn or false }
  end,
  -- Real signature: GetMediaSourceLength(src) -> (length, lengthIsQN).
  GetMediaSourceLength = function(src)
    record("GetMediaSourceLength")
    return src.len, src.is_qn
  end,
  PCM_Source_Destroy = function() proj.destroyed_sources = proj.destroyed_sources + 1 end,

  AddMediaItemToTrack = function(tr)
    record("AddMediaItemToTrack")
    local item = { track = tr, pos = 0, len = 0, takes = {} }
    tr.items[#tr.items + 1] = item
    return item
  end,
  AddTakeToMediaItem = function(item)
    record("AddTakeToMediaItem")
    local take = { item = item, name = "", source = nil }
    item.takes[#item.takes + 1] = take
    return take
  end,
  SetMediaItemTake_Source = function(take, src)
    record("SetMediaItemTake_Source")
    take.source = src
  end,
  GetSetMediaItemTakeInfo_String = function(take, key, val, set)
    record("GetSetMediaItemTakeInfo_String")
    if key == "P_NAME" and set then take.name = val end
    return true, take.name
  end,
  SetMediaItemInfo_Value = function(item, key, val)
    if key == "D_POSITION" then item.pos = val
    elseif key == "D_LENGTH" then item.len = val end
    return true
  end,
  GetMediaItemInfo_Value = function(item, key)
    if key == "D_POSITION" then return item.pos or 0 end
    if key == "D_LENGTH" then return item.len or 0 end
    return 0
  end,

  UpdateArrange = function()
    proj.update_arrange_calls = proj.update_arrange_calls + 1
    if proj.fail_update_arrange then error("stub: UpdateArrange blew up", 0) end
  end,
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
  proj.calls = {}
  proj.fail_update_arrange = false
end

-- --------------------------------------------------------------------------- --
-- 1. Happy path: two files -> two appended tracks, one item each, at at_s.
-- --------------------------------------------------------------------------- --
reset()
local r = call("import_audio_files", {
  files = { "C:/stems/vocals.wav", "C:/stems/drums.wav" },
  at_s = 4.5,
})
eq(r.ok, true, "import_audio_files ok")
eq(#r.result.imported, 2, "both files imported")
eq(#r.result.failed, 0, "nothing failed")

-- 1a. one NEW track per file, appended at the end, named from the basename sans extension.
eq(#proj.tracks, 2, "one track created per file")
eq(proj.tracks[1].name, "vocals", "track 1 named from basename sans extension")
eq(proj.tracks[2].name, "drums", "track 2 named from basename sans extension")
eq(r.result.imported[1].track_guid, proj.tracks[1].guid, "result carries the created track GUID")
eq(r.result.imported[1].file, "C:/stems/vocals.wav", "result echoes the source path")

-- 1b. the source/item/take chain really ran, in the documented order.
ok(called("PCM_Source_CreateFromFile"), "PCM_Source_CreateFromFile called")
ok(called("GetMediaSourceLength"), "GetMediaSourceLength called")
ok(called("AddMediaItemToTrack"), "AddMediaItemToTrack called")
ok(called("AddTakeToMediaItem"), "AddTakeToMediaItem called")
ok(called("SetMediaItemTake_Source"), "SetMediaItemTake_Source called")
ok(called("GetSetMediaItemTakeInfo_String"), "take name set via GetSetMediaItemTakeInfo_String")
eq(proj.calls[1], "PCM_Source_CreateFromFile", "source is created FIRST (before any mutation)")
eq(proj.calls[2], "GetMediaSourceLength", "length is read straight off the source")

-- 1c. D_POSITION == at_s and D_LENGTH == the source length, per item.
eq(#proj.tracks[1].items, 1, "track 1 carries exactly one media item")
approx(proj.tracks[1].items[1].pos, 4.5, "item 1 D_POSITION == at_s")
approx(proj.tracks[1].items[1].len, 12.5, "item 1 D_LENGTH == source length")
approx(proj.tracks[2].items[1].pos, 4.5, "item 2 D_POSITION == at_s")
approx(proj.tracks[2].items[1].len, 30.25, "item 2 D_LENGTH == source length")
approx(r.result.imported[2].length_s, 30.25, "result reports the source length in seconds")
eq(proj.tracks[1].items[1].takes[1].source.file, "C:/stems/vocals.wav",
  "the take's source is the file we opened")
eq(proj.tracks[1].items[1].takes[1].name, "vocals", "take name matches the track name")

-- 1d. ONE undo block for the WHOLE import (the one-Ctrl+Z promise), and UpdateArrange once.
eq(proj.undo_begins, 1, "undo block opened exactly once")
eq(proj.undo_ends, 1, "undo block closed exactly once")
eq(proj.undo_descs[1].desc, "Orpheus: import 2 audio files", "undo description names the import")
eq(proj.undo_descs[1].flags, -1, "Undo_EndBlock uses flags -1 (all)")
eq(proj.update_arrange_calls, 1, "UpdateArrange called ONCE, after every file")

-- --------------------------------------------------------------------------- --
-- 2. at_s defaults to 0.0 and name_prefix is applied verbatim.
-- --------------------------------------------------------------------------- --
reset()
local r2 = call("import_audio_files", {
  files = { "/stems/bass.flac" },
  name_prefix = "REF ",
})
eq(r2.ok, true, "prefixed import ok")
eq(proj.tracks[1].name, "REF bass", "name_prefix prepended to the basename")
approx(proj.tracks[1].items[1].pos, 0.0, "at_s defaults to 0.0")
approx(proj.tracks[1].items[1].len, 7.0, "flac source length honoured")
eq(proj.undo_descs[1].desc, "Orpheus: import 1 audio files", "undo description counts the files")

-- --------------------------------------------------------------------------- --
-- 3. PARTIAL FAILURE: an unopenable path is reported in `failed` (never silently
--    skipped), the other files still import, and the undo block is still balanced.
-- --------------------------------------------------------------------------- --
reset()
local r3 = call("import_audio_files", {
  files = { "C:/stems/vocals.wav", "C:/stems/missing.wav", "C:/stems/drums.wav" },
  at_s = 1.0,
})
eq(r3.ok, true, "partial failure still returns a structured result (not a raise)")
eq(#r3.result.imported, 2, "the two good files imported")
eq(#r3.result.failed, 1, "the missing file is reported as failed")
eq(r3.result.failed[1].file, "C:/stems/missing.wav", "the failure names the offending path")
ok(tostring(r3.result.failed[1].error):find("could not open audio file"),
  "the failure carries a clean error message")
ok(tostring(r3.result.failed[1].error):find("missing.wav"),
  "the error message names the path")
-- No stray track for the failed file: the source is validated BEFORE any mutation.
eq(#proj.tracks, 2, "no stray track created for the unopenable file")
eq(proj.tracks[1].name, "vocals", "good file 1 still landed")
eq(proj.tracks[2].name, "drums", "good file 2 still landed after the failure")
-- The undo block is opened and closed EXACTLY once even on the error path.
eq(proj.undo_begins, 1, "error path: undo block opened exactly once")
eq(proj.undo_ends, 1, "error path: undo block closed exactly once")
eq(proj.undo_descs[1].desc, "Orpheus: import 3 audio files",
  "error path: undo description counts every requested file")

-- --------------------------------------------------------------------------- --
-- 4. A ZERO-LENGTH source is a clean per-file error too (not a 0-length item), and the
--    source handle is released rather than leaked.
-- --------------------------------------------------------------------------- --
reset()
local r4 = call("import_audio_files", { files = { "C:/stems/empty.wav" } })
eq(r4.ok, true, "zero-length source -> structured result, not a raise")
eq(#r4.result.imported, 0, "nothing imported")
eq(#r4.result.failed, 1, "the zero-length file is reported as failed")
ok(tostring(r4.result.failed[1].error):find("empty or not a readable audio source"),
  "zero-length error message is the clean per-file one")
eq(#proj.tracks, 0, "no track created for a zero-length source")
eq(proj.destroyed_sources, 1, "the rejected source handle was destroyed, not leaked")
eq(proj.undo_begins, 1, "zero-length path: undo block opened once")
eq(proj.undo_ends, 1, "zero-length path: undo block closed once")

-- --------------------------------------------------------------------------- --
-- 5. An UNEXPECTED raise anywhere in the loop must still close the undo block exactly
--    once before the error surfaces — the block can never be left hanging open.
-- --------------------------------------------------------------------------- --
reset()
proj.fail_update_arrange = true
local r5 = call("import_audio_files", { files = { "C:/stems/vocals.wav" } })
eq(r5.ok, false, "an unexpected raise surfaces as ok=false")
ok(tostring(r5.error):find("UpdateArrange blew up"), "the original error is preserved")
eq(proj.undo_begins, 1, "unexpected raise: undo block opened exactly once")
eq(proj.undo_ends, 1, "unexpected raise: undo block CLOSED exactly once (never left open)")
proj.fail_update_arrange = false

-- --------------------------------------------------------------------------- --
-- 6. An empty file list is a clean error, and must NOT open an undo block at all.
-- --------------------------------------------------------------------------- --
reset()
local r6 = call("import_audio_files", { files = {} })
eq(r6.ok, false, "empty file list -> ok=false")
ok(tostring(r6.error):find("no files given"), "empty list error message")
eq(proj.undo_begins, 0, "empty list: no undo block was ever opened")
eq(proj.undo_ends, 0, "empty list: no undo block to close")
eq(#proj.tracks, 0, "empty list: no tracks created")

print(string.format("lua import handlers: %d passed, %d failed", passed, failed_count))
os.exit(failed_count == 0 and 0 or 1)
