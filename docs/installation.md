# Installing Orpheus

> ⚠️ Early development — `v0.1` is not yet on PyPI. These are the intended install steps; until then, run from a clone (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

Orpheus has **two halves** that both need to be running:

1. The **MCP server** — an external Python process your AI client launches.
2. The **in-REAPER bridge** — a Lua script you run inside REAPER once per session.

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

This copies `orpheus_bridge.lua` into REAPER's `Scripts/` folder (stable filename, so your config never changes). To override the IPC directory:

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

## 4. Run the bridge inside REAPER

1. Open REAPER and load (or create) a project.
2. `Actions → Show action list → ReaScript: Run…` → pick `orpheus_bridge.lua` (or assign it a toolbar button). It runs quietly in the background via a `defer` loop.
3. In your AI client, ask: *"Check the Orpheus connection."* — it calls `get_connection_status`.

**Success looks like:** the client shows the Orpheus tools (the 🔨 hammer icon in Claude Desktop), and `get_connection_status` reports REAPER running + the bridge alive + a round-trip latency.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Tools don't appear in the client | Server command wrong, or client not restarted. Try `fastmcp dev` to test the server in the Inspector first. |
| `get_connection_status` says "REAPER not listening" | The Lua bridge script isn't running in REAPER (re-run it), or `REAPER_MCP_BRIDGE_DIR` mismatch between the two halves. |
| Calls hang then time out | Bridge died (REAPER closed, project switched). Re-run the script. The heartbeat is designed to make this a clear error, not a silent hang. |
