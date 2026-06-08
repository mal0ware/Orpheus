"""Protocol-independent analysis core — pure functions, no REAPER, no MCP.

Everything here takes plain data (note lists, WAV paths, Specs) and returns plain data
(dataclasses / Pydantic models). That makes the hard musical logic fully unit-testable
against fixtures, and keeps the `tools/` layer a thin wrapper. (Pattern borrowed from
brightlikethelight/music21-mcp-server's "protocol-independent core".)
"""
