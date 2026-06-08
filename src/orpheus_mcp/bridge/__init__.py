"""The single point of contact with REAPER.

`client.BridgeClient` (Python side) and `lua/orpheus_bridge.lua` (in-REAPER side)
form a file-based JSON IPC channel. Nothing else in Orpheus may import `reaper.*`
or talk to REAPER directly — everything routes through the bridge.
"""

from orpheus_mcp.bridge.client import BridgeClient, BridgeError, BridgeTimeout

__all__ = ["BridgeClient", "BridgeError", "BridgeTimeout"]
