"""Pictures of a saved run, just for you to look at (the program never reads them back):
    path.png    where you walked. The line goes from blue (start) to red (end). The green dot is the entrance,
                joined by a thin green line to where the map began, and the red dot is the end. On other floors
                (path_floor2.png, ...) the green dot is where you arrived on that floor.
    labels.png  the confirmed labels (gates, checkpoints, ...) at their spot, one colour per name; the green
                ring is the entrance
    floors.png  for zones with several floors: the floors' maps stacked, with the first floor at the bottom
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .files import imwrite
from .labels import trusted_labels

FONT, FONT_SCALE, FONT_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
BACKGROUND_GRAY = 22
WALL_COLOR = (235, 190, 90)          # BGR, a light blue like the Tab map
ENTRANCE_COLOR = (0, 220, 0)
END_COLOR = (0, 0, 230)
# We don't draw a line between two route points where we don't know the route: a teleport (waypoint, checkpoint,
# portal), a frame that was found again far away, or a stretch without placed frames. Measured at 2 fps, walking moves
# about 17 map px per frame, sprinting got up to 85 (Sandswept Marsh), and teleports move hundreds.
MAX_STEP = 150         # map px per frame; anything faster is a jump, not walking
SPRINT_STEP = 100      # map px per frame not placed; at most the fastest sprint, so a teleport after a fade is a jump
MAX_MISSING = 4        # frames without a position (2 s) after which we don't know the route in between
# bright on the dark map, and easy to tell apart from the light blue walls and the green entrance
FIXED_LABEL_COLORS = {"Checkpoint": (0, 215, 255), "Waypoint": (245, 245, 245)}
PALETTE = [(40, 140, 255), (95, 95, 255), (215, 110, 255), (255, 140, 180), (110, 240, 210), (150, 170, 255),
           (0, 255, 190), (180, 220, 255), (60, 200, 240), (200, 120, 160)]


def label_colors(names: list[str]) -> dict[str, tuple]:
    """One colour per label name. Checkpoint and Waypoint always get the same one, the others each get their own."""
    out = {n: FIXED_LABEL_COLORS[n] for n in names if n in FIXED_LABEL_COLORS}
    for i, n in enumerate(sorted({n for n in names if n not in out})):
        out[n] = PALETTE[i % len(PALETTE)]
    return out


def map_picture(m: np.ndarray) -> np.ndarray:
    """The saved map as a dark picture with the walls in light blue."""
    img = np.full(m.shape + (3,), BACKGROUND_GRAY, np.uint8)
    img[m > 0] = WALL_COLOR
    return img


def _overlap(a, b) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _outlined_text(img: np.ndarray, text: str, org: tuple[int, int], color: tuple,
                   scale: float = FONT_SCALE) -> None:
    """Text with a dark outline, so it's readable on walls."""
    for c, w in (((0, 0, 0), 5), (color, FONT_THICKNESS)):
        cv2.putText(img, text, org, FONT, scale, c, w, cv2.LINE_AA)


def draw_labels(img: np.ndarray, labels: list[tuple]) -> None:
    """Dots with their names, with the names placed so they don't overlap each other or any dot. labels: [(text, x,
    y, color, dot radius)]. A negative radius draws a ring instead of a dot, for the entrance: it often sits on the
    gate it came through, and that gate's dot stays visible inside the ring. A name that got pushed away from its dot
    gets a thin line to it."""
    H, W = img.shape[:2]
    dots = [(x - abs(r) - 2, y - abs(r) - 2, x + abs(r) + 2, y + abs(r) + 2) for _, x, y, _, r in labels]
    placed: list[tuple] = []
    out = []
    for text, x, y, col, r in labels:
        (tw, th), base = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        bw, bh = tw + 8, th + base + 8                  # a box with a little room for the dark outline
        best = None
        for dist in range(0, 400, 20):
            g = abs(r) + 6 + dist
            spots = [(x + g, y - bh // 2), (x - g - bw, y - bh // 2), (x - bw // 2, y - g - bh),
                     (x - bw // 2, y + g), (x + g, y - g - bh), (x + g, y + g), (x - g - bw, y - g - bh),
                     (x - g - bw, y + g)]
            for bx, by in spots:
                box = (bx, by, bx + bw, by + bh)
                cost = sum(_overlap(box, o) for o in placed + dots)
                cost += 0 if bx >= 0 and bx + bw <= W and by >= 0 and by + bh <= H else 10 ** 6
                cost += dist        # closer to the dot is better when nothing overlaps either way
                if best is None or cost < best[0]:
                    best = (cost, box, dist)
            if best[0] - best[2] == 0:
                break
        _, box, dist = best
        placed.append(box)
        out.append((text, x, y, col, r, box, dist))
    for _text, x, y, col, _r, box, dist in out:            # leader lines first, so dots and names end up on top
        if dist:
            cx, cy = min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3])
            cv2.line(img, (x, y), (int(cx), int(cy)), col, 1, cv2.LINE_AA)
    for _text, x, y, col, r, _box, _dist in sorted(out, key=lambda o: o[4] >= 0):    # rings first, dots on top
        if r < 0:
            cv2.circle(img, (x, y), -r, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.circle(img, (x, y), -r, col, 3, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), r + 2, (0, 0, 0), -1)
            cv2.circle(img, (x, y), r, col, -1)
    for text, _x, _y, col, _r, box, _dist in out:
        (_, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        _outlined_text(img, text, (box[0] + 4, box[1] + 4 + th), col)


def render_floors(folder: Path, maps: list[np.ndarray], gap: int = 60) -> None:
    """floors.png: the floors' maps stacked, with the first floor at the bottom and the last one reached on top."""
    w = max(m.shape[1] for m in maps)
    rows = []
    for i, m in reversed(list(enumerate(maps, 1))):
        row = np.full((m.shape[0] + gap, w, 3), BACKGROUND_GRAY, np.uint8)
        x = (w - m.shape[1]) // 2
        row[gap:, x:x + m.shape[1]] = map_picture(m)
        _outlined_text(row, f"Floor {i}", (10, gap - 15), (245, 245, 245), FONT_SCALE * 1.5)
        rows.append(row)
    imwrite(folder / "floors.png", np.vstack(rows))


def _pt(p) -> tuple[int, int]:
    return int(p[0]), int(p[1])


def route_breaks(path: list, times: list[int]) -> list[bool]:
    """For each step path[i-1] -> path[i]: True where we don't know the route (see MAX_STEP, MAX_MISSING). times: the
    tick of each point (see Stitcher.path_t). In a thinned route a normal step spans several ticks."""
    dt = np.maximum(1, np.diff(np.asarray(times, np.int64)))
    dist = np.hypot(*np.diff(np.asarray(path, np.float64).reshape(-1, 2), axis=0).T)
    stride = float(np.median(dt)) if len(dt) else 1.0
    # every lost frame lets a step be one sprint longer, so a teleport after a short fade (a few frames lost) still
    # counts as a jump
    allowed = MAX_STEP * np.minimum(dt, stride) + SPRINT_STEP * np.maximum(0, dt - stride)
    return ((dt > stride + MAX_MISSING) | (dist > allowed)).tolist()


def render_run_images(folder: Path, m: np.ndarray, meta: dict, path: list | None = None,
                      times: list[int] | None = None, suffix: str = "") -> None:
    """Write path.png and labels.png next to a saved run. path, times: the full route and its frame numbers (defaults
    to meta's thinned ones). suffix is for another floor ("_floor2" gives path_floor2.png, and meta is then that
    floor's entry of other_floors)."""
    path = path or meta.get("path") or []
    times = times or meta.get("path_t")
    if not times or len(times) != len(path):      # runs saved before we kept the times: evenly spaced
        times = [i * max(1, round(meta.get("frames", len(path)) / max(1, len(path)))) for i in range(len(path))]
    breaks = route_breaks(path, times)
    entrance = meta.get("entrance") or (path[0] if path else None)
    img = map_picture(m)
    n = len(path)
    if n and entrance:
        cv2.line(img, _pt(entrance), _pt(path[0]), ENTRANCE_COLOR, 2, cv2.LINE_AA)
    for i in range(1, n):
        if breaks[i - 1]:
            continue
        t = i / max(1, n - 1)
        cv2.line(img, _pt(path[i - 1]), _pt(path[i]), (int(255 * (1 - t)), 60, int(255 * t)), 3, cv2.LINE_AA)
    if n:
        cv2.circle(img, _pt(entrance), 14, ENTRANCE_COLOR, -1)
        cv2.circle(img, _pt(path[-1]), 14, END_COLOR, -1)
    imwrite(folder / f"path{suffix}.png", img)

    img = map_picture(m)
    marks = []
    if entrance:
        marks.append(("entrance", *_pt(entrance), ENTRANCE_COLOR, -18))
    confirmed = trusted_labels(meta.get("labels", []))
    colors = label_colors([lb["text"] for lb in confirmed])
    for lb in confirmed:
        text = lb["text"].encode("ascii", "replace").decode()     # Hershey fonts only do ASCII
        marks.append((text, int(lb["x"]), int(lb["y"]), colors[lb["text"]], 10))
    draw_labels(img, marks)
    imwrite(folder / f"labels{suffix}.png", img)
