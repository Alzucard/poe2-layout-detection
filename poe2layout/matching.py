"""Works out where one stitched map sits inside another (shift only, same scale)."""
from __future__ import annotations

import cv2
import numpy as np

from .outline import line_features


def agreement(a: np.ndarray, b: np.ndarray, tol: int = 7) -> float:
    """How well the outlines of two aligned maps agree (0..1), inside the area both of them have explored."""
    k = np.ones((tol, tol), np.uint8)
    fa = cv2.dilate(a, np.ones((41, 41), np.uint8)) > 0
    fb = cv2.dilate(b, np.ones((41, 41), np.uint8)) > 0
    both = fa & fb
    wa, wb = (a > 0) & both, (b > 0) & both
    if wa.sum() < 300 or wb.sum() < 300:
        return 0.0
    return float(min((wa & (cv2.dilate(b, k) > 0)).sum() / wa.sum(), (wb & (cv2.dilate(a, k) > 0)).sum() / wb.sum()))


def coarse_features(m: np.ndarray, coarse: int, cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Downscaled line_features of a map. `cache` (one per map that never changes) keeps them per `coarse`."""
    if cache is not None and coarse in cache:
        return cache[coarse]
    out = cv2.resize(line_features(m), None, fx=1 / coarse, fy=1 / coarse, interpolation=cv2.INTER_AREA)
    if cache is not None:
        cache[coarse] = out
    return out


def locate_ranked(partial: np.ndarray, full: np.ndarray, coarse: int = 4, peaks: int = 5,
                  cache: dict[int, np.ndarray] | None = None,
                  partial_cache: dict[int, np.ndarray] | None = None) -> list[tuple[float, int, int]]:
    """Possible placements of `partial` inside `full`, best first, as [(agreement, dx, dy)], where partial pixel (x,
    y) is full pixel (x + dx, y + dy). cache and partial_cache are the coarse feature caches of `full` and `partial`
    (see coarse_features)."""
    ph, pw = partial.shape
    pad = -(-max(ph, pw) // coarse) * coarse            # multiple of coarse, so padded coarse map == padded map
    padded = cv2.copyMakeBorder(full, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    p = pad // coarse
    full_small = cv2.copyMakeBorder(coarse_features(full, coarse, cache), p, p, p, p, cv2.BORDER_CONSTANT)
    partial_small = coarse_features(partial, coarse, partial_cache)
    res = cv2.matchTemplate(full_small, partial_small, cv2.TM_CCORR_NORMED)
    partial_feat = line_features(partial)
    out = []
    for _ in range(peaks):                              # refine the strongest coarse peaks
        _, _, _, loc = cv2.minMaxLoc(res)
        cx, cy = loc[0] * coarse, loc[1] * coarse
        cv2.circle(res, loc, 6, 0, -1)
        x0, y0 = max(0, cx - 2 * coarse), max(0, cy - 2 * coarse)
        window = line_features(padded[y0:y0 + ph + 4 * coarse, x0:x0 + pw + 4 * coarse])
        if window.shape[0] < ph or window.shape[1] < pw:
            continue
        fine = cv2.matchTemplate(window, partial_feat, cv2.TM_CCORR_NORMED)
        _, _, _, l2 = cv2.minMaxLoc(fine)
        fx, fy = x0 + l2[0], y0 + l2[1]
        out.append((agreement(partial, padded[fy:fy + ph, fx:fx + pw]), fx - pad, fy - pad))
    out.sort(key=lambda t: -t[0])
    return out


def locate(partial: np.ndarray, full: np.ndarray, coarse: int = 4, peaks: int = 3,
           cache: dict[int, np.ndarray] | None = None,
           partial_cache: dict[int, np.ndarray] | None = None) -> tuple[int, int, float]:
    """The best placement of `partial` (the smaller map) inside `full`, as (dx, dy, agreement). Each of the `peaks`
    strongest coarse matches gets refined at full size, which is the expensive part."""
    ag, dx, dy = locate_ranked(partial, full, coarse, peaks=peaks, cache=cache, partial_cache=partial_cache)[0]
    return dx, dy, ag
