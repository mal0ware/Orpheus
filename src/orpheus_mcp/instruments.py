"""Deterministic instrument-selection ladder: prefer the user's own good instruments, then the
optional sound pack, and guarantee sound with stock. This module installs nothing."""
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
    pack: dict | None = None,
) -> dict:
    """Pick an instrument for `role` ('keys'|'bass'|'drums'|'guitar') against the installed
    inventory. Ladder: override -> installed allowlist match -> sound pack -> stock.

    `pack` is the installed sound-pack instrument for this role (see
    `orpheus_mcp.soundpack.instrument_for_role`), or None."""
    if override:
        return {"kind": "named", "name": override, "source": "override"}

    match = _match(inventory, ROLE_ALLOWLIST.get(role, []))
    if match:
        return {"kind": "named", "name": match, "source": "installed"}

    if pack is not None:
        return {"kind": "pack", "name": pack["id"], "source": "pack"}

    if role == "drums":
        return {"kind": "drumkit", "name": None, "source": "stock"}
    return {"kind": "named", "name": "ReaSynth", "source": "stock"}


def choose_instrument(role: str, inventory: list[str], override: str | None = None) -> dict:
    """`select_instrument` with the installed sound pack looked up for you."""
    from orpheus_mcp import soundpack

    return select_instrument(role, inventory, override, pack=soundpack.instrument_for_role(role))


ZONES_PER_CALL = 32  # the bridge's add_sampler_zones cap


def load_instrument(bridge, track: str, choice: dict) -> dict:
    """Make `track` audible with a `select_instrument` choice: a named plugin, the stock
    drum kit, or a sound-pack instrument built as RS5k sampler zones."""
    if choice["kind"] == "drumkit":
        from orpheus_mcp.drumkit import load_drumkit

        return load_drumkit(bridge, track)
    if choice["kind"] == "pack":
        from orpheus_mcp import soundpack

        zones = soundpack.zones_for(choice["name"])
        added, failed = 0, []
        for i in range(0, len(zones), ZONES_PER_CALL):
            res = bridge.call("add_sampler_zones", track=track, zones=zones[i:i + ZONES_PER_CALL])
            added += res.get("zones_added", 0)
            failed += res.get("failed", [])
        return {"track": track, "loaded": f"sound pack: {choice['name']}",
                "zones_added": added, "failed": failed}
    return bridge.call("add_instrument", track=track, kind="named", name=choice["name"])
