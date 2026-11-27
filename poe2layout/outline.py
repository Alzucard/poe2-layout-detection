"""Finds the Tab map's outline of the walkable area (thin blue lines) in a screen frame, and cleans up stitched maps."""
from __future__ import annotations

import cv2
import numpy as np

# The colour of the Tab map outline (HSV), measured on our own and on community screenshots: hue is about 100-145.
OUTLINE_HSV_LO = np.array([98, 25, 90])
OUTLINE_HSV_HI = np.array([146, 175, 255])

# Where we read the Tab map, as screen fractions (x, y). That's everything except the buff bar at the top, the stats and
# zone box at the top right, and the HUD at the bottom (life/mana text and orbs on the sides, the flask and skill bar
# panels; between those the map goes all the way down to the XP bar). Drawn on a 1440p screenshot. You can override it
# in config with "analysis": {"map_area": [[x, y], ...]}.
DEFAULT_MAP_AREA = ((0.0, 0.096), (0.69, 0.096), (0.69, 0.205), (1.0, 0.205), (1.0, 0.715), (0.895, 0.715),
                    (0.895, 0.833), (0.708, 0.833), (0.708, 0.91), (0.315, 0.91), (0.315, 0.72), (0.0, 0.72))
MapArea = tuple[tuple[float, float], ...]

# The bottom HUD (orb ornaments, skill bar) barely changes. Without a tint its B/R and G/R means are about 0.91 and
# 0.87. Blue-light filters (f.lux, Night Light) tint the whole capture warm, which turns the blue outline beige.
HUD_REFERENCE_RATIOS = (0.91, 0.87)
HUD_BANDS = ((0.86, 1.0, 0.0, 0.27), (0.86, 1.0, 0.73, 1.0))     # (y0, y1, x0, x1) as screen fractions

# A black screen is when this share of the map area is darker than BLACK_LEVEL in all channels. Across 1158 recorded
# frames, the darkest normal frame (the full-screen map view) came to 90%.
BLACK_LEVEL = 10
BLACK_SHARE = 0.97

# Side panels (inventory on the right; skills, options, market and so on on the left) push the game view sideways, and
# the player then stands in the middle of the part that's still visible. A panel's inner border is a straight vertical
# line 884.5 px from the screen edge at 1440p. We assume it scales with the screen height, like the rest of the UI. On
# 1213 kept frames the border scored at least 0.94 with a panel open and at most 0.70 without one.
PANEL_EDGE = 884.5 / 1440
PANEL_BANDS = ((0.08, 0.33), (0.33, 0.58), (0.58, 0.83))   # a tooltip can hide the border in one or two of them
PANEL_STEP = 30
PANEL_SCORE = 0.85

_area_masks: dict[tuple[int, int, MapArea], np.ndarray] = {}


def as_map_area(points) -> MapArea:
    return tuple((float(x), float(y)) for x, y in points)


def map_area_mask(h: int, w: int, area: MapArea = DEFAULT_MAP_AREA) -> np.ndarray:
    """A bool (h, w) mask that's True where we read the map, with the area scaled to this frame size. Memoized."""
    key = (h, w, area)
    if key not in _area_masks:
        m = np.zeros((h, w), np.uint8)
        pts = np.array([(round(x * w), round(y * h)) for x, y in area], np.int32)
        cv2.fillPoly(m, [pts], 1)
        _area_masks[key] = m > 0
    return _area_masks[key]


def is_black(frame: np.ndarray, area: MapArea = DEFAULT_MAP_AREA) -> bool:
    """True when the screen faded to black (stairs to another floor, or a teleport). The full-screen map view is
    black too, but only up to about 90% of the map area because of its lines and labels. Real ground never gets close
    to that."""
    small = frame[::8, ::8]
    return float((small.max(axis=2)[map_area_mask(*small.shape[:2], area)] < BLACK_LEVEL).mean()) >= BLACK_SHARE


def _panel_border(frame: np.ndarray, x: int) -> bool:
    """Is there a straight vertical border at column x (give or take 3 px) in at least one height band?"""
    h = frame.shape[0]
    strip = cv2.cvtColor(np.ascontiguousarray(frame[:, x - 3:x + 4]), cv2.COLOR_BGR2GRAY).astype(np.int16)
    steps = np.abs(np.diff(strip, axis=1)) > PANEL_STEP
    return any(float(steps[int(y0 * h):int(y1 * h)].mean(axis=0).max()) >= PANEL_SCORE for y0, y1 in PANEL_BANDS)


def panel_edge(h: int) -> int:
    """Distance (px) of a side panel's inner border from the screen edge, for a screen h px high."""
    return round(PANEL_EDGE * h)


def panel_sides(frame: np.ndarray) -> tuple[bool, bool]:
    """Returns (left, right): which side panels are open."""
    h, w = frame.shape[:2]
    edge = panel_edge(h)
    if not 4 <= edge <= w // 2 - 4:
        return False, False
    return _panel_border(frame, edge - 1), _panel_border(frame, w - edge)


def view_shift(frame: np.ndarray) -> int:
    """Screen px from the screen centre to the player, sideways. It's positive when a left panel is open, negative
    for a right one, and 0 when neither or both are open."""
    left, right = panel_sides(frame)
    return (panel_edge(frame.shape[0]) // 2) * (int(left) - int(right))


def color_gains(frame: np.ndarray) -> tuple[float, float]:
    """Per-channel gains (B, G) that undo a warm colour cast, estimated from the bottom HUD, which barely changes."""
    h, w = frame.shape[:2]
    px = np.concatenate([frame[int(y0 * h):int(y1 * h):4, int(x0 * w):int(x1 * w):4].reshape(-1, 3)
                         for y0, y1, x0, x1 in HUD_BANDS]).astype(np.float32)
    b, g, r = px.mean(0)
    if r < 8:   # a loading screen or black frame, nothing to judge
        return 1.0, 1.0
    gb = float(np.clip(HUD_REFERENCE_RATIOS[0] * r / max(b, 1e-3), 1.0, 3.0))
    gg = float(np.clip(HUD_REFERENCE_RATIOS[1] * r / max(g, 1e-3), 1.0, 2.0))
    # small deviations are normal (skill icons, life and mana levels), so only correct a clear cast
    return (gb if gb > 1.25 else 1.0), (gg if gb > 1.25 and gg > 1.08 else 1.0)


def neutralize_color_cast(frame: np.ndarray) -> np.ndarray:
    gb, gg = color_gains(frame)
    if gb == 1.0 and gg == 1.0:
        return frame

    def lut(k: float) -> np.ndarray:
        return np.clip(np.arange(256) * k, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(frame)
    return cv2.merge((cv2.LUT(b, lut(gb)), cv2.LUT(g, lut(gg)), r))


def outline_mask(frame: np.ndarray, area: MapArea = DEFAULT_MAP_AREA) -> np.ndarray:
    """The thin blue outline of the walkable area (uint8, 0 or 255). Glows and blobs (portals, skill effects), the
    HUD, the minimap and open side panels (blue gem sockets and mod text) are removed."""
    left, right = panel_sides(frame)
    frame = neutralize_color_cast(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, OUTLINE_HSV_LO, OUTLINE_HSV_HI)
    # On bright ground (sunlit sand) the see-through outline turns a pale lavender and falls outside the fixed colour
    # range, but it's still clearly bluer than the ground right next to it.
    b, _, r = cv2.split(frame)
    blue_minus_red = cv2.addWeighted(b, 1, r, -1, 128)            # b - r + 128, clipped to 0..255
    bluer = (blue_minus_red.astype(np.int16) - cv2.medianBlur(blue_minus_red, 21).astype(np.int16) > 22) & (b > 70)
    m = cv2.bitwise_or(m, bluer.astype(np.uint8) * 255)
    glow = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)), np.ones((5, 5), np.uint8))
    m = cv2.bitwise_and(m, cv2.bitwise_not(glow))
    m[~map_area_mask(*m.shape, area)] = 0
    edge = panel_edge(m.shape[0])
    if left:
        m[:, :edge] = 0
    if right:
        m[:, m.shape[1] - edge:] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    area_px = st[:, cv2.CC_STAT_AREA].astype(np.float32)
    span = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]).astype(np.float32)
    box_px = st[:, cv2.CC_STAT_WIDTH] * st[:, cv2.CC_STAT_HEIGHT]
    fill = area_px / np.maximum(1, box_px)
    # Outlines are thin curves or rings (pillars), only a few px thick however long they get. Terrain speckle is small
    # and compact, and glows (waypoints, portals, skills) are thick everywhere.
    thickness = area_px / np.maximum(span, 1)
    # Area over span overrates closed rings (small pillars come out at about 3x the line width). So a ring still passes
    # if its real width (area over half the contour length) looks like a line and its hole fills most of its box. Open
    # squiggles like lightning are thin too, but they have no hole, so they still get removed.
    keep = (area_px >= 25) & (span >= 14) & ((fill < 0.45) | (span >= 60))
    thin = (thickness <= 4.0) | (fill < 0.2)
    ring_candidate = keep & ~thin & (span >= 18)
    ring_candidate[0] = False
    ring = np.zeros(n, bool)
    if ring_candidate.any():
        contours, hier = cv2.findContours((ring_candidate[lab] * 255).astype(np.uint8), cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_NONE)
        # Perimeter (same as cv2.arcLength, closed) and area (same as cv2.contourArea) of all contours at once. Noisy
        # frames have thousands of tiny hole contours, and a Python loop over them cost more than the rest of this
        # function.
        lens = np.fromiter((len(c) for c in contours), np.int64, len(contours))
        pts = np.concatenate(contours).reshape(-1, 2).astype(np.float64)
        starts = np.r_[0, np.cumsum(lens)[:-1]]
        prev = np.arange(len(pts)) - 1
        prev[starts] = starts + lens - 1            # each contour is closed, so its first point follows its last
        q = pts[prev]
        perimeter = np.add.reduceat(np.hypot(pts[:, 0] - q[:, 0], pts[:, 1] - q[:, 1]), starts)
        contour_area = np.abs(np.add.reduceat(q[:, 0] * pts[:, 1] - pts[:, 0] * q[:, 1], starts)) / 2
        owner = lab[pts[starts, 1].astype(np.intp), pts[starts, 0].astype(np.intp)]
        is_hole = hier[0, :, 3] >= 0                # an inner contour is a hole in that component
        per_component = np.bincount(owner, perimeter, n)
        hole = np.bincount(owner[is_hole], contour_area[is_hole], n)
        width = area_px / np.maximum(per_component / 2, 1)
        ring = ring_candidate & (width <= 3.0) & (hole >= 0.3 * box_px)
    keep &= thin | ring
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def line_features(mask: np.ndarray) -> np.ndarray:
    """A blurred float version of a line mask. It gives smooth correlation peaks and tolerates 1-2 px of jitter."""
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 3)


def denoise_map(m: np.ndarray, join: int = 15, frac: float = 0.1, path=None, near: int = 150,
                min_px: int = 30) -> np.ndarray:
    """Drop scraps that aren't part of the map. Terrain and effect speckle at the screen edges passes the vote,
    because those spots only count as 'seen' when the speckle is there. Real walls, pillars and islands lie within
    `join` px of the connected body of the map, while scraps are small groups far away from it. path (map
    coordinates, optional): every real part of the map also comes within `near` px of where you walked (0 px in the
    Clearfell and Lightless runs), while junk placed by a bad frame, such as a shaky start, lies off the path (348
    px). So separate parts away from the path get dropped too."""
    if m.size == 0 or not m.any():
        return m
    n, lab = cv2.connectedComponents((cv2.dilate(m, np.ones((join, join), np.uint8)) > 0).astype(np.uint8), 8)
    px = np.bincount(lab[m > 0], minlength=n)
    px[0] = 0
    keep = px >= frac * px.max()
    if path is not None and len(path) and n > 2:
        # With a path, nearness decides. Small islands (Grelwood's pillars) and bits of wall cut off by a gap are real
        # if you walked past them, however small they are, and big junk away from the path isn't.
        walk = np.zeros(m.shape, np.uint8)
        cv2.polylines(walk, [np.asarray(path, np.int32).reshape(-1, 1, 2)], False, 255, 2 * near + 1)
        touched = np.zeros(n, bool)
        touched[np.unique(lab[(walk > 0) & (m > 0)])] = True
        keep = touched & (px >= min_px)
    keep[0] = False
    return np.where(keep[lab], m, 0).astype(np.uint8)
