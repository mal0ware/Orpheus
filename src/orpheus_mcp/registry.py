"""Tool registration + toolset gating.

Each tool module exposes a ``register(mcp)`` function. We keep a category → register
map so the exposed tool surface can be filtered by *toolset* (à la shiehn's profiles /
github-mcp-server's ``--toolsets``). This dodges the 128-tool client cap and avoids
flooding the model with tools it doesn't need for the task at hand.

Default profile is small and NL-shaped; ``full`` exposes everything.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

# Imported lazily inside register_tools to keep import-time cheap and avoid hard
# failures when an optional dependency (e.g. librosa) is missing.
_CATEGORY_IMPORTS: dict[str, str] = {
    "bridge": "orpheus_mcp.tools.bridge_status",
    "project": "orpheus_mcp.tools.project",
    "transport": "orpheus_mcp.tools.transport",
    "tracks": "orpheus_mcp.tools.tracks",
    "midi": "orpheus_mcp.tools.midi",
    "theory": "orpheus_mcp.tools.theory",
    "analyze": "orpheus_mcp.tools.analyze",
    "style": "orpheus_mcp.tools.style",
    "apply": "orpheus_mcp.tools.apply",
    "render": "orpheus_mcp.tools.render",
    "instruments": "orpheus_mcp.tools.instruments",
    "dsl": "orpheus_mcp.tools.dsl",
    "compose": "orpheus_mcp.tools.compose",
    "mix": "orpheus_mcp.tools.mix",
}

# Toolset profiles → the categories they expose.
PROFILES: dict[str, tuple[str, ...]] = {
    # v0.1 "it explains your track": read + understand, no apply.
    "explain": ("bridge", "project", "transport", "theory", "analyze", "style", "render"),
    # v0.3 the full differentiator.
    "default": (
        "bridge", "project", "transport", "tracks", "midi",
        "theory", "analyze", "style", "apply", "render",
        "instruments", "compose",
    ),
    "full": tuple(_CATEGORY_IMPORTS.keys()),
}


def _import_register(module_path: str) -> Callable[..., None]:
    import importlib

    module = importlib.import_module(module_path)
    return module.register


def register_tools(mcp: object, profile: str = "default") -> list[str]:
    """Register every tool category in ``profile`` onto the FastMCP instance.

    HONESTY RULE: only the ``full`` profile exposes not-yet-implemented stub tools, and
    each stub's docstring starts with "[NOT IMPLEMENTED]". The explain/default surfaces
    advertise working tools only, so a client never plans around a tool that would just
    raise at call time. (Stubs raise when CALLED, not when registered — an
    except-NotImplementedError around registration can never catch them.)

    Returns the list of categories successfully registered. Categories whose optional
    dependencies are missing are skipped with a warning rather than crashing the server.
    """
    categories: Iterable[str] = PROFILES.get(profile, PROFILES["default"])
    include_stubs = profile == "full"
    registered: list[str] = []
    for category in categories:
        module_path = _CATEGORY_IMPORTS[category]
        try:
            _import_register(module_path)(mcp, include_stubs=include_stubs)
            registered.append(category)
        except ImportError as exc:  # optional dependency missing
            import warnings

            warnings.warn(f"Skipping toolset '{category}': {exc}", stacklevel=2)
    return registered
