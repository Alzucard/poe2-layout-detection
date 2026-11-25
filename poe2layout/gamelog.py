"""The game's own log (Client.txt): where to find it, and the zone changes it records."""
from __future__ import annotations

import re
import winreg
from dataclasses import dataclass
from pathlib import Path

from .zones import Instance

# Only the game's own messages count, and those come right after the line's date, time and client tag. Chat goes into
# the same log ("[INFO Client 1] #Name: text"), so a chat message with the same text mustn't match.
_LINE_START = r"^\d{4}/\d\d/\d\d \d\d:\d\d:\d\d \d+ [0-9a-f]+ \[\w+ Client \d+\] "
_GENERATING_RE = re.compile(_LINE_START + r'Generating level (\d+) area "([^"]+)" with seed (\d+)')
_SCENE_RE = re.compile(_LINE_START + r"\[SCENE\] Set Source \[(.+)\]\s*$")


@dataclass(frozen=True)
class ZoneEvent:
    """You entered `zone`. None means you're back at the login or character select and in no zone."""
    zone: str | None
    instance: Instance | None
    level: int | None


def find_client_log(configured: str = "") -> Path | None:
    """Path of Exile 2's logs/Client.txt: the configured path, or else the standalone and Steam install folders."""
    if configured:
        p = Path(configured)
        return p if p.is_file() else None
    steam_roots = [Path(r"C:\Program Files (x86)\Steam"), Path(r"C:\Program Files\Steam")]
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            steam_roots.insert(0, Path(winreg.QueryValueEx(k, "SteamPath")[0]))
    except OSError:
        pass
    libraries = []
    for root in steam_roots:
        vdf = root / "steamapps" / "libraryfolders.vdf"
        if vdf.is_file():
            libraries += [Path(m.replace("\\\\", "\\")) for m in
                          re.findall(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="replace"))]
        libraries.append(root)
    candidates = [lib / "steamapps" / "common" / "Path of Exile 2" / "logs" / "Client.txt" for lib in libraries]
    candidates += [Path(r"C:\Program Files (x86)\Grinding Gear Games\Path of Exile 2\logs\Client.txt"),
                   Path(r"C:\Program Files\Grinding Gear Games\Path of Exile 2\logs\Client.txt")]
    found = [p for p in candidates if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


class GameLog:
    """Follows Client.txt for zone changes. For every zone load the game writes:
        Generating level 35 area "G3_2_1" with seed 3384606516
        [SCENE] Set Source [Infested Barrens]
    poll() returns the new ZoneEvents. The first poll only reports the zone you're in right now."""

    def __init__(self, path: Path, tail_bytes: int = 2_000_000):
        self.path = path
        self._pos = max(0, path.stat().st_size - tail_bytes)
        self._partial = b""                     # an unfinished last line, completed by the next read
        self._generating: tuple[int, str, int] | None = None    # (level, zone code, seed) waiting for its name
        self._first = True

    def poll(self) -> list[ZoneEvent]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._pos:                    # the log was cleared or replaced
            self._pos, self._partial = 0, b""
        if size == self._pos:
            self._first = False                 # nothing old to skip, so whatever comes next is new
            return []
        start = self._pos
        with open(self.path, "rb") as f:
            f.seek(start)
            data = f.read(size - start)
        self._pos = size
        if self._first and start > 0:
            data = data.split(b"\n", 1)[-1]     # we started mid-file, so drop the partial first line
        lines = (self._partial + data).split(b"\n")
        self._partial = lines.pop()
        events = [ev for raw in lines if (ev := self._parse(raw.decode("utf-8", errors="replace")))]
        if self._first:
            self._first = False
            return events[-1:]                  # history: all that matters is where you are now
        return events

    def _parse(self, line: str) -> ZoneEvent | None:
        m = _GENERATING_RE.search(line)
        if m:
            self._generating = (int(m.group(1)), m.group(2), int(m.group(3)))
            return None
        m = _SCENE_RE.search(line)
        if not m:
            return None
        name = m.group(1).strip()
        if name == "(unknown)":                 # the login or character select screen
            self._generating = None
            return ZoneEvent(None, None, None)
        if name == "(null)" or self._generating is None:
            return None                         # e.g. "Set Source [Act 1]" (the world map) has no Generating line
        level, code, seed = self._generating
        self._generating = None
        return ZoneEvent(name, Instance(code, seed), level)
