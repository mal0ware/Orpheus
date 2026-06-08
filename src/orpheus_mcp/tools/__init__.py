"""FastMCP tool wrappers, grouped by domain.

Each module exposes ``register(mcp)``. Tools are thin: they validate inputs, call the
bridge and/or the pure-function analysis/theory layers, and return Pydantic models.
Musical logic belongs in ``orpheus_mcp.analysis`` / ``orpheus_mcp.theory``, not here.

Annotation convention (the approval gate, encoded in the protocol):
    * read-only tools  → annotations={"readOnlyHint": True}
    * mutating  tools  → annotations={"destructiveHint": True}
"""
