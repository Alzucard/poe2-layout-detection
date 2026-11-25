"""Zones: comparing names, instances, towns, and the zone data (gates and points of interest).

Gates come from data/zones.json (shipped with the program, read only) plus the ones learned while you play
(learned_exits.json in the data folder). That's how a zone missing from zones.json gets its gates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import ZONES_PATH
from .files import read_json, write_json_atomic
from .logs import get_logger

log = get_logger("zones")

TOWN_NAMES = {"clearfell encampment", "ardura caravan", "ziggurat encampment", "kingsmarch"}


def normalize(s: str) -> str:
    """Lower case, only letters, digits and single spaces."""
    s = s.lower().replace("’", "'").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def zone_key(s: str) -> str:
    """The zone name as we compare it everywhere: normalize() without a leading "the"."""
    s = normalize(s)
    return s[4:] if s.startswith("the ") else s


@dataclass(frozen=True)
class Instance:
    """One generated copy of a zone, from the game log. Going back to it (through a portal) shows the same seed."""
    code: str          # the game's zone code, e.g. "G1_2"
    seed: int

    @property
    def id(self) -> str:
        """e.g. "G1_2_1628456714". Used as the run folder name and in layouts.json."""
        return f"{self.code}_{self.seed}"

    def to_json(self) -> list:
        return [self.code, self.seed]

    @classmethod
    def from_json(cls, data) -> Instance:
        return cls(str(data[0]), int(data[1]))


def is_town(zone: str | None, instance: Instance | None = None) -> bool:
    """Towns and hideouts have no layout worth recording. We go by the zone code (G1_town, HideoutSummit) when we
    know it, and by the name otherwise."""
    if instance is not None:
        code = instance.code.lower()
        if code.endswith("_town") or code.startswith("hideout"):
            return True
    if not zone:
        return False
    key = zone_key(zone)
    return key in TOWN_NAMES or "hideout" in key


class ZoneData:
    """Gates and points of interest of each zone. Not thread-safe; only the analysis thread uses it."""

    def __init__(self, zones_path: Path = ZONES_PATH, learned_path: Path | None = None):
        self._points: dict[str, list[str]] = {}     # zone_key -> point of interest names
        self._gates: dict[str, list[str]] = {}      # zone_key -> gate names (both directions)
        self._learned_path = learned_path
        self._learned: dict[str, list[str]] = {}    # zone_key -> gate names learned from your zone changes
        try:
            acts = read_json(zones_path, default={})
        except (OSError, ValueError) as e:
            log.warning("%s not loaded (%s) - only learned gates and Checkpoint/Waypoint are known", zones_path, e)
            acts = {}
        for zones in acts.values():
            for z in zones:
                self._points[zone_key(z["name"])] = [p["name"] for p in z.get("points_of_interest", [])]
                for other in z.get("connections", []):
                    # a passage has a gate at both ends, even where zones.json only lists one direction
                    self._add_gate(self._gates, z["name"], other)
                    self._add_gate(self._gates, other, z["name"])
        if learned_path is not None:
            try:
                self._learned = read_json(learned_path, default={})
            except (OSError, ValueError) as e:
                log.warning("%s unreadable (%s) - starting without learned gates", learned_path, e)

    @staticmethod
    def _add_gate(table: dict[str, list[str]], zone: str, gate: str) -> bool:
        names = table.setdefault(zone_key(zone), [])
        if zone_key(gate) in {zone_key(n) for n in names}:
            return False
        names.append(gate)
        return True

    def gates(self, zone: str | None) -> list[str]:
        """Names of the zones this zone connects to (its gate labels on the map)."""
        if not zone:
            return []
        k = zone_key(zone)
        out = list(self._gates.get(k, []))
        known = {zone_key(n) for n in out}
        return out + [n for n in self._learned.get(k, []) if zone_key(n) not in known]

    def exits(self, zone: str | None) -> set[str]:
        """zone_keys of the zones this zone connects to, or empty if we don't know."""
        return {zone_key(n) for n in self.gates(zone)}

    def points_of_interest(self, zone: str | None) -> list[str]:
        return list(self._points.get(zone_key(zone), [])) if zone else []

    def learn_exit(self, zone: str, came_from: str) -> bool:
        """You walked from came_from into zone through a gate (its label was read at the entrance), so each zone has
        a gate named after the other. Returns True when we learned something new."""
        if not zone or not came_from or zone_key(zone) == zone_key(came_from):
            return False
        if "hideout" in zone_key(zone) or "hideout" in zone_key(came_from):
            return False                    # you leave hideouts through a portal
        added = False
        for a, b in ((zone, came_from), (came_from, zone)):
            if zone_key(b) not in self.exits(a):
                added |= self._add_gate(self._learned, a, b)
        if added:
            log.info("learned the gates between %r and %r", zone, came_from)
            if self._learned_path is not None:
                try:
                    self._learned_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json_atomic(self._learned_path, self._learned, sort_keys=True)
                except OSError as e:
                    log.warning("could not save %s (%s) - kept for this session", self._learned_path, e)
        return added
