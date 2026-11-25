"""Groups the saved runs of a zone into layouts, even when one is a mirrored or turned copy of another.

The Tab map shows the ground squashed vertically: walls that run along the world axes show up at about +-39.5 deg on
screen instead of +-45 (measured on saved runs). So a mirror or quarter turn in the world is only a plain flip along
the screen axes. Along the diagonals it's a transpose stretched by k = tan(39.5 deg). TRANSFORMS holds the world's 8
mirrors and rotations as they look on screen.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from .logs import get_logger
from .matching import locate

log = get_logger("layouts")

MAP_DIAGONAL_DEG = 39.5
_K = float(np.tan(np.radians(MAP_DIAGONAL_DEG)))
TRANSFORMS = {                      # name -> 2x2 matrix on (x, y) screen pixels; det < 0 means mirrored
    "id": ((1, 0), (0, 1)),
    "flip_h": ((-1, 0), (0, 1)),
    "flip_v": ((1, 0), (0, -1)),
    "rot180": ((-1, 0), (0, -1)),
    "diag": ((0, 1 / _K), (_K, 0)),
    "anti_diag": ((0, -1 / _K), (-_K, 0)),
    "rot90": ((0, -1 / _K), (_K, 0)),
    "rot270": ((0, 1 / _K), (-_K, 0)),
}


@dataclass(frozen=True)
class LayoutThresholds:
    """Two runs are the same layout in two cases. The first is when one map fits into the other with an outline
    agreement of at least `same` (same layout is about 0.85-0.95 in tests, different about 0.3) and their footprints
    overlap by at least `same_min_shape`. Zones built from shared pieces score 0.65-0.85 on a single common room or
    town alone (Excavation 0.66 / 0.20, Deshar's town up to 0.85 / 0.10). The second is when their footprints overlap
    by at least `same_shape` with an outline agreement of at least `shape_floor`. That covers zones that shuffle the
    obstacles inside one outer shape (Clearfell 0.79 / 0.49; different Lightless layouts reach 0.62 at most)."""
    same: float = 0.65
    same_min_shape: float = 0.5
    same_shape: float = 0.7
    shape_floor: float = 0.4

    def same_layout(self, agreement: float, shape: float) -> bool:
        return ((agreement >= self.same and shape >= self.same_min_shape)
                or (agreement >= self.shape_floor and shape >= self.same_shape))


DEFAULT_THRESHOLDS = LayoutThresholds()

# ---------------------------------------------------------------- orientations

def _matrix(name: str) -> np.ndarray:
    return np.array(TRANSFORMS[name], np.float64)


def transform_name(A) -> str:
    """The TRANSFORMS entry with matrix A."""
    return next(n for n in TRANSFORMS if np.allclose(_matrix(n), A, atol=1e-6))


def transform_then(first: str, second: str) -> str:
    """The orientation you get by applying `first`, then `second`."""
    return transform_name(_matrix(second) @ _matrix(first))


def transform_inverse(name: str) -> str:
    return transform_name(np.linalg.inv(_matrix(name)))


def is_mirrored(name: str) -> bool:
    return bool(np.linalg.det(_matrix(name)) < 0)


def transform_map(m: np.ndarray, name: str) -> np.ndarray:
    """The map image as seen through one of the TRANSFORMS."""
    if name == "id":
        return m
    if name in ("flip_h", "flip_v", "rot180"):     # exact, no resampling
        out = m[:, ::-1] if name == "flip_h" else m[::-1] if name == "flip_v" else m[::-1, ::-1]
        return np.ascontiguousarray(out)
    h, w = m.shape
    A = _matrix(name)
    corners = A @ np.array([[0, w - 1, 0, w - 1], [0, 0, h - 1, h - 1]], np.float64)
    lo = corners.min(axis=1)
    size = np.ceil(corners.max(axis=1) - lo).astype(int) + 1
    out = cv2.warpAffine(m, np.hstack([A, -lo[:, None]]), (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
    return (out > 64).astype(np.uint8) * 255 if m.max() > 1 else (out > 0).astype(np.uint8)


# ---------------------------------------------------------------- map comparison

class MapView:
    """One orientation of a saved map, plus what the comparisons compute from it (computed only once)."""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.coarse_cache: dict[int, np.ndarray] = {}     # see matching.coarse_features
        self._footprint: np.ndarray | None = None

    @property
    def footprint(self) -> np.ndarray:
        """The filled area the map covers: walls thickened, gaps closed and enclosed space filled in."""
        if self._footprint is None:
            f = cv2.dilate(self.image, np.ones((31, 31), np.uint8))
            f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8))
            f = cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
            outside = f.copy()
            cv2.floodFill(outside, None, (0, 0), 255)
            self._footprint = ((f > 0) | (outside == 0))[1:-1, 1:-1]
        return self._footprint


class MapVariants:
    """A saved map in all orientations, built the first time it's needed."""

    def __init__(self, image: np.ndarray):
        self._views = {"id": MapView(image)}
        self.pixels = int((image > 0).sum())

    @property
    def base(self) -> MapView:
        return self._views["id"]

    def view(self, name: str) -> MapView:
        if name not in self._views:
            self._views[name] = MapView(transform_map(self.base.image, name))
        return self._views[name]


def footprint_iou(fa: np.ndarray, fb: np.ndarray, dx: int, dy: int) -> float:
    """IoU of footprint fa placed at (dx, dy) in fb's coordinates, so fa pixel (x, y) is fb pixel (x+dx, y+dy)."""
    x0, y0 = min(0, dx), min(0, dy)
    x1, y1 = max(fb.shape[1], dx + fa.shape[1]), max(fb.shape[0], dy + fa.shape[0])
    A = np.zeros((y1 - y0, x1 - x0), bool)
    B = A.copy()
    A[dy - y0:dy - y0 + fa.shape[0], dx - x0:dx - x0 + fa.shape[1]] = fa
    B[-y0:-y0 + fb.shape[0], -x0:-x0 + fb.shape[1]] = fb
    union = int((A | B).sum())
    return float((A & B).sum()) / union if union else 0.0


def run_fit(a: MapVariants, b: MapVariants) -> tuple[float, float, str]:
    """Returns (outline agreement, shape overlap, orientation) of two maps at their best fit. We look for the smaller
    map inside the bigger one in every orientation. The orientation is the TRANSFORMS entry that turns a's map into
    b's.

    Shape overlap is the IoU of the two filled footprints at that placement. Some zones (Clearfell) keep their outer
    shape but shuffle the obstacles inside. Then the walls only agree about 0.5, just like different layouts that
    share one stretch of path, but the footprints still cover each other (0.79 against at most 0.62)."""
    # We refine one peak per orientation. In all 222 pairs of saved runs (Nov 2026), the winning placement of every
    # same-layout pair was its orientation's strongest coarse peak; weaker peaks only ever won between different
    # layouts. Doing 8 full-size refinements instead of 24 sorted a new Deshar run (with 18 saved) in about a third of
    # the time.
    swapped = a.pixels > b.pixels
    if swapped:
        a, b = b, a
    best = None
    for name in TRANSFORMS:
        v = a.view(name)
        dx, dy, ag = locate(v.image, b.base.image, peaks=1, cache=b.base.coarse_cache, partial_cache=v.coarse_cache)
        if best is None or ag > best[0]:
            best = (ag, v, dx, dy, name)
    ag, v, dx, dy, name = best
    shape = footprint_iou(v.footprint, b.base.footprint, dx, dy)
    return ag, shape, transform_inverse(name) if swapped else name


# ---------------------------------------------------------------- grouping

def layout_names(used: set[str]) -> Iterator[str]:
    """A, B, ..., Z, AA, AB, ..., skipping names that are taken."""
    i = 0
    while True:
        s, k = "", i
        while True:
            s = chr(65 + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        if s not in used:
            yield s
        i += 1


def layout_order(name: str) -> tuple[int, str]:
    """Sort key for the order the names were handed out in: A < Z < AA."""
    return len(name), name


def _same_screen(a: tuple | None, b: tuple | None) -> bool:
    """An unknown size (None) fits any size."""
    return a is None or b is None or tuple(a) == tuple(b)


def group_runs(zone: str, runs: list[tuple[str, MapVariants]], known: dict,
               thresholds: LayoutThresholds = DEFAULT_THRESHOLDS,
               screens: dict[str, tuple[int, int]] | None = None) -> tuple[dict[str, dict], bool]:
    """Sort runs [(instance id, maps)] into layouts. `known` is the previous result (layouts.json): runs found there
    keep their group, so each new run is only compared against the others once. A run that fits several groups joins
    them together (two runs that explored different halves get linked by a third one that saw both).

    screens: {instance id: screen size}. Runs filmed at different sizes are never compared. A run without a size
    (saved before we recorded it) is compared with all of them.

    Returns ({instance id: {"layout", "orientation", "mirrored"}}, changed)."""
    ids = {n for n, _ in runs}
    maps = dict(runs)
    screens = screens or {}
    groups = {n: g["layout"] for n, g in known.items()
              if n in ids and isinstance(g, dict) and g.get("orientation") in TRANSFORMS}
    orient = {n: known[n]["orientation"] for n in groups}
    gscreen: dict[str, tuple] = {}              # group -> screen size (a group never mixes sizes; None means unknown)
    for m, g in groups.items():
        if m in screens:
            gscreen.setdefault(g, tuple(screens[m]))
    changed = len(groups) != len(known)
    for n, variants in runs:
        if n in groups:
            continue
        hits: dict[str, str] = {}               # group -> orientation of this run against that group
        for m, g in list(groups.items()):
            if g in hits or not _same_screen(screens.get(n), gscreen.get(g)):
                continue
            ag, shape, tf = run_fit(variants, maps[m])
            if thresholds.same_layout(ag, shape):
                hits[g] = transform_then(tf, orient[m])
        if hits:
            keep = min(hits, key=layout_order)  # merge every group this run links to into the oldest one
            # ...of one screen size. A run without a size can fit groups of different sizes, so it only joins those of
            # keep's size, or else its own, or else the size of the oldest group with a size that it fits
            size = gscreen.get(keep) or screens.get(n) or next(
                (gscreen[g] for g in sorted(hits, key=layout_order) if g in gscreen), None)
            hits = {g: tf for g, tf in hits.items() if _same_screen(gscreen.get(g), size)}
            for g, tf in hits.items():
                if g == keep:
                    continue
                # turns group g's reference into keep's: undo this run's turn against g, then apply keep's
                d = transform_then(transform_inverse(tf), hits[keep])
                for m in [m for m, mg in groups.items() if mg == g]:
                    groups[m], orient[m] = keep, transform_then(orient[m], d)
                gscreen.pop(g, None)
            groups[n], orient[n] = keep, hits[keep]
        else:
            groups[n], orient[n] = next(layout_names(set(groups.values()))), "id"
            size = screens.get(n)
        if size:
            gscreen[groups[n]] = tuple(size)
        changed = True
        turn = f", {orient[n]}{' (mirrored)' if is_mirrored(orient[n]) else ''}" if orient[n] != "id" else ""
        log.info("run %s of %r is layout %s%s%s", n, zone, groups[n], "" if hits else " (new layout)", turn)
    out = {n: {"layout": groups[n], "orientation": orient[n], "mirrored": is_mirrored(orient[n])}
           for n in sorted(groups)}
    changed |= any(known.get(n) != g for n, g in out.items())
    return out, changed
