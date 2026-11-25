"""Shared test helpers: a synthetic Tab map (thin blue lines on dark ground) to pan a camera over."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

FRAME_W, FRAME_H = 1920, 1080


def outline_blue() -> tuple[int, int, int]:
    """BGR of a colour inside the outline's HSV range."""
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[115, 120, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])


def make_world(seed: int = 1, w: int = 4000, h: int = 3000, strokes: int = 500) -> np.ndarray:
    """A big BGR picture full of random blue polylines. It stands in for a zone's whole map."""
    rng = np.random.default_rng(seed)
    world = np.full((h, w, 3), 12, np.uint8)
    blue = outline_blue()
    for _ in range(strokes):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        pts = [(x, y)]
        for _ in range(6):
            x = int(np.clip(x + rng.integers(-150, 150), 0, w - 1))
            y = int(np.clip(y + rng.integers(-150, 150), 0, h - 1))
            pts.append((x, y))
        cv2.polylines(world, [np.array(pts)], False, blue, 2, cv2.LINE_AA)
    return world


def frame_at(world: np.ndarray, x: int, y: int) -> np.ndarray:
    return np.ascontiguousarray(world[y:y + FRAME_H, x:x + FRAME_W])


def make_map(seed: int, w: int = 700, h: int = 500, turns: int = 14) -> np.ndarray:
    """A small binary (0/255) map picture like a saved run's map.png: the outline of a winding corridor."""
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    x, y = w // 2, h // 2
    pts = [(x, y)]
    for _ in range(turns):
        if rng.random() < 0.5:
            x = int(np.clip(x + rng.integers(-180, 180), 30, w - 30))
        else:
            y = int(np.clip(y + rng.integers(-150, 150), 30, h - 30))
        pts.append((x, y))
    for (a, b), (c, d) in zip(pts, pts[1:], strict=False):
        cv2.rectangle(m, (min(a, c) - 18, min(b, d) - 18), (max(a, c) + 18, max(b, d) + 18), 255, 2)
    return m


@pytest.fixture(scope="session")
def world() -> np.ndarray:
    return make_world()
