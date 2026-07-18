"""Deterministic instrument-selection ladder: prefer the user's own good instruments,
guarantee sound with stock. Discovery only — this module installs nothing."""
from __future__ import annotations

# Curated per-role allowlist of quality instruments to prefer if already installed.
# Substring-matched against the installed-FX inventory (names vary by version/format).
ROLE_ALLOWLIST: dict[str, list[str]] = {
    "keys": ["Vital", "Surge XT", "Pianoteq", "Kontakt", "Decent Sampler"],
    "bass": ["Vital", "Surge XT", "Kontakt"],
    "drums": ["MT-PowerDrumKit", "Battery", "EZdrummer", "Superior Drummer", "Kontakt"],
}


def _match(inventory: list[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        for fx in inventory:
            if cand.lower() in fx.lower():
                return cand
    return None


def select_instrument(
    role: str,
    inventory: list[str],
    override: str | None = None,
    pack_installed: bool = False,
) -> dict:
    """Pick an instrument for `role` ('keys'|'bass'|'drums') against the installed
    inventory. Ladder: override -> installed allowlist match -> curated pack -> stock."""
    if override:
        return {"kind": "named", "name": override, "source": "override"}

    match = _match(inventory, ROLE_ALLOWLIST.get(role, []))
    if match:
        return {"kind": "named", "name": match, "source": "installed"}

    if pack_installed:
        return {"kind": "named", "name": "sfizz", "source": "pack"}

    if role == "drums":
        return {"kind": "drumkit", "name": None, "source": "stock"}
    return {"kind": "named", "name": "ReaSynth", "source": "stock"}
