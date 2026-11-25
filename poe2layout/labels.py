"""Tab map labels: reading them (OCR on video frames), voting on them, and working out the entrance from them."""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

import numpy as np

from .logs import get_logger
from .ocr import WinOcr
from .zones import ZoneData, normalize, zone_key

log = get_logger("labels")

# Which parts of the screen we read for labels, and which we leave out: the chat (bottom left) and the zone panel (top
# right).
LABEL_ROWS = (0.06, 0.86)                                         # screen rows (fraction of the height)
LABEL_EXCLUDE = ((0.0, 0.62, 0.30, 1.0), (0.78, 0.0, 1.0, 0.25))  # (x0, y0, x1, y1) as screen fractions
FIXED_LABELS = ("Checkpoint", "Waypoint")
# small map labels get enlarged for OCR; 1.5x is about 40% faster than 2x and on 9 runs it confirmed 71 labels instead
# of 72
OCR_UPSCALE = 1.5

LABEL_MIN_HITS = 2         # how often a label has to be read before it counts
LABEL_MIN_SHARE = 0.25     # ...and in at least this share of the reads that had its spot on screen
LABEL_SAME_SPOT = 90       # the same name twice within this (canvas px) is one place read twice (shifted frames)
LABEL_MERGE_PX = 40        # reads of the same name this close together are one label
# Text that stays put on the screen while you walk isn't a map label. The quest tracker ("Ancient Beacons" matches
# "Ancient Ruin") ends up on the map all along your route, a new label every few steps (34 of them in Cemetery of the
# Eternals). So we drop a read when the same name was read at that screen spot for this many labels.
SCREEN_FIXED_PX = 12       # screen px; OCR boxes of the same text differ by a few px
SCREEN_FIXED_LABELS = 4    # real labels of one name land on the same screen spot by chance, but hardly ever four

ENTRANCE_NEAR = 250        # how far (canvas px) the Waypoint label may be from where the map began
# The gate named after the zone you came from is a much surer sign, so it may be farther away. When the map's first
# frames get thrown away (a shaky start), the map begins about 300 px in. In Hunting Grounds the gate was 312 px away,
# while a misread "Waypoint" at the far exit was 172 px away.
ENTRANCE_GATE_NEAR = 400
WAYPOINT_MIN_HITS = 3


def vocab_match(text: str, vocab: list[str]) -> str | None:
    """OCR of small map labels is patchy ("roubl d m" for Troubled Camp, "heckpoint"). So we accept a vocabulary
    entry when most of its letters show up in order AND the OCR text is mostly that entry, which keeps chat lines
    from matching. Entries shorter than 5 letters ("Keth", "Well") have to be read completely."""
    t = normalize(text)
    if len(t.replace(" ", "")) < 4:
        return None
    best, best_frac = None, 0.0
    for v in vocab:
        nv = normalize(v)
        sm = difflib.SequenceMatcher(None, t, nv, autojunk=False)
        matched = sum(b.size for b in sm.get_matching_blocks())
        frac_v, frac_t = matched / max(1, len(nv)), matched / max(1, len(t))
        need = min(5, len(nv.replace(" ", "")))
        if matched >= need and frac_v >= 0.6 and frac_t >= 0.6 and frac_v > best_frac:
            best, best_frac = v, frac_v
    return best


class LabelReader:
    """Reads the known label names in a video frame. Create it in the thread that uses it (because of WinOcr)."""

    def __init__(self, ocr: WinOcr, zones: ZoneData):
        self.ocr = ocr
        self.zones = zones

    def vocabulary(self, zone: str | None, came_from: str | None = None) -> list[str]:
        """Checkpoint, Waypoint, the zone's gates, the gate you came through (even for zones missing from
        zones.json), and the zone's points of interest."""
        words = list(FIXED_LABELS) + self.zones.gates(zone)
        if came_from and zone_key(came_from) not in {zone_key(w) for w in words}:
            words.append(came_from)
        return words + self.zones.points_of_interest(zone)

    def read(self, frame: np.ndarray, zone: str | None,
             came_from: str | None = None) -> list[tuple[str, float, float, float, float]]:
        """Returns [(label name, x, y, w, h)] in screen pixels."""
        h, w = frame.shape[:2]
        y0, y1 = int(h * LABEL_ROWS[0]), int(h * LABEL_ROWS[1])
        img = frame[y0:y1].copy()
        for ex0, ey0, ex1, ey1 in LABEL_EXCLUDE:
            img[max(0, int(ey0 * h) - y0):max(0, int(ey1 * h) - y0), int(ex0 * w):int(ex1 * w)] = 0
        vocab = self.vocabulary(zone, came_from)
        out = []
        for box in self.ocr.read_boxes(img, upscale=OCR_UPSCALE):
            name = vocab_match(box.text, vocab)
            if name:
                out.append((name, box.x, box.y + y0, box.w, box.h))
                log.debug("label %r read as %r at (%.0f, %.0f)", box.text, name, box.x, box.y + y0)
        return out


@dataclass
class VotedLabel:
    key: str           # normalized text, for comparing
    text: str
    x: float           # canvas px
    y: float
    hits: int = 1      # how often it was read
    looks: int = 0     # label reads that had this spot on screen
    screen: tuple = ()  # screen px (x, y) of each read, to spot screen-fixed text (not saved)

    def to_json(self) -> dict:
        return {"text": self.text, "x": round(self.x, 1), "y": round(self.y, 1), "hits": self.hits,
                "looks": self.looks}


class LabelVotes:
    """Map labels in canvas coordinates. When the same label is read in many frames, the reads are merged. Labels are
    voted on just like the outline: a frame placed in the wrong spot (a tracking glitch) puts its labels in the wrong
    spots too. Those spots get looked at many more times later without the label showing up, so the bad read is
    outvoted."""

    def __init__(self, merge_px: float = LABEL_MERGE_PX):
        self.items: list[VotedLabel] = []
        self.merge_px = merge_px

    def _near(self, it: VotedLabel, key: str, x: float, y: float) -> bool:
        return it.key == key and abs(it.x - x) < self.merge_px and abs(it.y - y) < self.merge_px

    def _merge(self, key: str, x: float, y: float, hits: int, looks: int, screen: tuple) -> bool:
        """Fold reads of a label at (x, y) into a nearby label of the same name, if there is one. Returns False when
        there's none, so the caller adds it as a new label."""
        for it in self.items:
            if self._near(it, key, x, y):
                n = it.hits
                it.x, it.y = (it.x * n + x * hits) / (n + hits), (it.y * n + y * hits) / (n + hits)
                it.hits = n + hits
                it.looks += looks
                it.screen += screen
                return True
        return False

    def look(self, x0: float, y0: float, x1: float, y1: float, skip=()) -> None:
        """One label read covered this canvas rectangle, minus the `skip` rectangles, which are never read."""
        for it in self.items:
            x, y = it.x, it.y
            if x0 <= x < x1 and y0 <= y < y1 and not any(a <= x < c and b <= y < d for a, b, c, d in skip):
                it.looks += 1

    def add(self, text: str, x: float, y: float, screen: tuple | None = None) -> None:
        """screen: where on the screen it was read (see without_screen_fixed)."""
        key = " ".join(text.split()).lower()
        if len(key) < 4:
            return
        read = (tuple(screen),) if screen is not None else ()
        if not self._merge(key, x, y, 1, 0, read):
            self.items.append(VotedLabel(key, text, x, y, screen=read))

    def absorb(self, items: list[VotedLabel], dx: float, dy: float) -> None:
        """Take over another map piece's labels, moved by (dx, dy) to where that piece sits now."""
        for o in items:
            x, y = o.x + dx, o.y + dy
            if not self._merge(o.key, x, y, o.hits, o.looks, o.screen):
                self.items.append(replace(o, x=x, y=y))

    def shift(self, dx: float, dy: float) -> None:
        for it in self.items:
            it.x += dx
            it.y += dy

    def shifted_copy(self, dx: float, dy: float) -> list[VotedLabel]:
        return [replace(it, x=it.x + dx, y=it.y + dy) for it in self.items]

    def to_json(self) -> list[dict]:
        """All the labels, including unconfirmed ones (meta.json keeps them all). trusted_labels() picks out the
        confirmed ones."""
        return [it.to_json() for it in self.items]


def without_screen_fixed(floors: list[list[VotedLabel]]) -> list[list[VotedLabel]]:
    """Each floor's labels without their reads of screen-fixed text (see SCREEN_FIXED_PX and SCREEN_FIXED_LABELS).
    The quest tracker sits at the same screen spot on every floor, so we count the labels of all floors together.
    A label that has no reads left is dropped.

    Reads go into screen cells of half SCREEN_FIXED_PX, and two reads count as the same spot when their cells are at
    most two apart. That catches everything within SCREEN_FIXED_PX (and some reads up to about twice as far), and
    it never compares every read with every other one, which took 216 MB for 3000 reads of one name."""
    cell = SCREEN_FIXED_PX / 2
    reads = Counter()           # (name, cell x, cell y, label) -> how many of the label's reads fell in that cell
    labels = defaultdict(set)   # (name, cell x, cell y) -> labels with a read in that cell
    for f, items in enumerate(floors):
        for i, it in enumerate(items):
            for x, y in it.screen:
                spot = (it.key, int(x // cell), int(y // cell))
                reads[(*spot, (f, i))] += 1
                labels[spot].add((f, i))
    fixed = Counter()           # label -> its reads of screen-fixed text
    for (key, cx, cy, label), n in reads.items():
        near = set().union(*(labels.get((key, cx + a, cy + b), ()) for a in range(-2, 3) for b in range(-2, 3)))
        if len(near) >= SCREEN_FIXED_LABELS:
            fixed[label] += n
    return [[replace(it, hits=it.hits - fixed[f, i]) if fixed[f, i] else it
             for i, it in enumerate(items) if it.hits > fixed[f, i]]
            for f, items in enumerate(floors)]


def label_ok(label: dict) -> bool:
    """True for a label that was read often enough, and in enough of the reads that looked at its spot."""
    return label["hits"] >= LABEL_MIN_HITS and label["hits"] >= LABEL_MIN_SHARE * label.get("looks", 0)


def one_per_spot(labels: list[dict]) -> list[dict]:
    """Keep the most-read label of each name at each spot."""
    out: list[dict] = []
    for lb in sorted(labels, key=lambda k: -k["hits"]):
        if not any(o["text"] == lb["text"] and np.hypot(o["x"] - lb["x"], o["y"] - lb["y"]) < LABEL_SAME_SPOT
                   for o in out):
            out.append(lb)
    return out


def trusted_labels(labels: list[dict]) -> list[dict]:
    """The confirmed labels of a run (meta.json "labels")."""
    return one_per_spot([lb for lb in labels if label_ok(lb)])


def _distance(label: dict, point) -> float:
    return float(np.hypot(label["x"] - point[0], label["y"] - point[1]))


def entrance_point(first, labels: list[dict], came_from: str | None = None,
                   exits: set[str] = frozenset()) -> list[float] | None:
    """Where you came into the zone. The map only begins once there's enough outline on screen, a few seconds in and
    100-300 px from the gate, so its first point isn't the entrance. The entrance label is nearby: the gate named
    after the zone you came from, or else the Waypoint (you teleported in). Two reads are enough here, because the
    Waypoint label is often half hidden by the player and fails the share vote (4 of 24 reads in Clearfell).

    If neither is nearby, we take any other gate of this zone (`exits`, as zone_keys) close to the first point. After
    a waypoint teleport, the zone you were in last tells us nothing, and the Waypoint label might not be read at all.
    In Grim Tangle, after a teleport from Red Vale, the Waypoint was never read but the Grelwood gate right next to
    it was read 8 times, 97 px away. If there's nothing at all (portal, town, no labels read), we use the first map
    point."""
    if first is None:
        return None

    def found(want, near: float) -> list[dict]:
        want = {want} if isinstance(want, str) else want
        return [lb for lb in labels if lb["hits"] >= LABEL_MIN_HITS and zone_key(lb["text"]) in want
                and _distance(lb, first) <= near]

    gate = zone_key(came_from) if came_from else None
    # First choice is a close gate. Second is the nearer of a farther gate and a well-read close Waypoint: a teleport
    # lands right at the Waypoint (read 4-10 times), while a misread "Waypoint" at another exit gets read about twice.
    # Last is any close Waypoint, since one half hidden by the player may only be read twice (2 of 21 in Clearfell).
    far_gate = found(gate, ENTRANCE_GATE_NEAR) if gate else []
    waypoints = found("waypoint", ENTRANCE_NEAR)
    candidates = ((found(gate, ENTRANCE_NEAR) if gate else [])
                  or (far_gate + [w for w in waypoints if w["hits"] >= WAYPOINT_MIN_HITS] if far_gate else waypoints)
                  or found(exits, ENTRANCE_NEAR))
    if candidates:
        best = min(candidates, key=lambda k: _distance(k, first) - 10 * k["hits"] ** 0.5)
        return [round(float(best["x"]), 1), round(float(best["y"]), 1)]
    return [float(first[0]), float(first[1])]


def came_through_gate(labels: list[dict], first, came_from: str | None) -> bool:
    """Was the gate named after the zone you came from confirmed near where the map began? Then you really walked
    through it (it wasn't a waypoint or portal trip), and the two zones are connected."""
    if first is None or not came_from:
        return False
    gate = zone_key(came_from)
    return any(zone_key(lb["text"]) == gate and _distance(lb, first) <= ENTRANCE_GATE_NEAR
               for lb in trusted_labels(labels))
