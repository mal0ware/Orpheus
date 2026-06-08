# Contributing to Orpheus

Thanks for your interest! Orpheus is early and contributions are welcome — especially around the analysis brain, style fingerprints, and ReaScript bridge robustness.

## Ground rules

- **Reimplement, don't copy.** Orpheus stands on the shoulders of prior art (see [`docs/frontier-analysis.md`](docs/frontier-analysis.md)). We adopt *patterns and ideas*. We do **not** copy source from GPL-licensed projects (e.g. waveform-MCP) into this MIT codebase.
- **AI-assisted contributions are welcome** — disclose meaningful AI assistance in your PR description, and make sure you understand and have tested every line you submit. Judgment over volume.
- **The MIDI round-trip is sacred.** Any change touching note writing must keep `tests/test_midi_roundtrip.py` green. PPQ/tempo/take correctness is the load-bearing invariant.

## Dev setup

```bash
git clone https://github.com/mal0ware/Orpheus
cd Orpheus
uv sync --extra dev --extra analysis
uv run pytest
uv run ruff check .
```

The analysis and theory layers are **protocol-independent pure functions** (`src/orpheus_mcp/analysis/`, `src/orpheus_mcp/theory/`) — they're unit-testable without REAPER running. Most logic should live there; the `tools/` layer is a thin FastMCP wrapper.

## Architecture & tool conventions

- Read [`docs/architecture.md`](docs/architecture.md) first.
- Expose musical time to the model in **beats**, never PPQ/ticks.
- Mark read-only tools with `readOnlyHint=True` and mutating tools with `destructiveHint=True` — the human-approval gate is enforced through MCP tool annotations, not bolted on.
- Constrain numeric params with `Annotated[int, Field(ge=..., le=...)]` (velocity 0–127, etc.).
- New tools register through `registry.py` so they're toolset-gateable.

## PRs

- One focused change per PR. Include tests. Keep the README's tool table in sync.
- CI (ruff + pytest) must pass.
