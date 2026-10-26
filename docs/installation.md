# Installing Orpheus

> ⚠️ Early development — `v0.1` is not yet on PyPI. These are the intended install steps; until then, run from a clone (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

Orpheus has **two halves** that both need to be running:

1. The **MCP server** — an external Python process your AI client launches.
2. The **in-REAPER bridge** — a Lua script that runs inside REAPER. It starts with REAPER once
   installed.

They talk over a local folder of JSON files. No network, no ports, no cloud.

## 1. Install the server

```bash
# Zero-install run (once published):
uvx orpheus-mcp

# or from a clone:
uv sync
uv run orpheus-mcp
```

## 2. Install the in-REAPER bridge (one-time)

```bash
orpheus-mcp install-bridge
```

This copies `orpheus_bridge.lua` into REAPER's `Scripts/` folder (stable filename, so your config never changes) and adds a marked block to REAPER's `Scripts/__startup.lua`, which REAPER runs every time it opens, so the bridge starts with REAPER. Anything else in that file is left as it was. `orpheus-mcp install-bridge --no-autostart` skips the block; `orpheus-mcp remove-autostart` takes it out again.

To override the IPC directory:

```bash
export REAPER_MCP_BRIDGE_DIR=/path/to/bridge_dir
```

## 3. Point your MCP client at the server

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "orpheus": { "command": "uvx", "args": ["orpheus-mcp"] }
  }
}
```

Config file locations:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Restart the client. (For **Cursor / Claude Code**, use that client's MCP server config with the same command.)

## 4. Start REAPER

1. Open REAPER. The bridge starts with it and prints `Orpheus bridge listening in: …` to
   REAPER's console.
2. Was REAPER already open when you installed? Run `orpheus-mcp start-bridge`, or just ask
   Claude: the first Orpheus call starts the bridge in the open REAPER by itself
   (`REAPER.exe -nonewinst <script>`). Orpheus never opens REAPER for you.
3. In your AI client, ask: *"Check the Orpheus connection."* — it calls `get_connection_status`.

Starting the bridge twice is harmless: the newer copy takes over and the older one stops.
To start it by hand instead: `Actions → Show action list`, search `orpheus_bridge`, Run.

**Success looks like:** the client shows the Orpheus tools (the 🔨 hammer icon in Claude Desktop), and `get_connection_status` reports REAPER running + the bridge alive + a round-trip latency.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Tools don't appear in the client | Server command wrong, or client not restarted. Try `fastmcp dev` to test the server in the Inspector first. |
| `get_connection_status` says not connected | Its `autostart` field says why: REAPER isn't open, the bridge isn't installed (`orpheus-mcp install-bridge`), REAPER's program wasn't found (set `ORPHEUS_REAPER_EXE`), or REAPER didn't answer (restart REAPER). A `REAPER_MCP_BRIDGE_DIR` mismatch between the two halves looks the same. |
| Calls hang then time out | Bridge died (REAPER closed). Open REAPER again, or `orpheus-mcp start-bridge`. The heartbeat is designed to make this a clear error, not a silent hang. |
