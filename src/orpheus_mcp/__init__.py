"""Orpheus — an MCP server that analyzes, explains, and reshapes music inside REAPER.

The public surface is the FastMCP server in :mod:`orpheus_mcp.server`. Everything
musical lives in protocol-independent pure functions under
:mod:`orpheus_mcp.analysis` and :mod:`orpheus_mcp.theory`; the :mod:`orpheus_mcp.tools`
package is a thin FastMCP wrapper over them, and :mod:`orpheus_mcp.bridge` is the
single point of contact with REAPER.
"""

__version__ = "0.1.0.dev0"
