r"""Outline detection on real game frames (tests/frames: training frames with the chat area blacked out).

Each <name>.jpg has a <name>_outline.png with the outline we found when the test was written, so any change to
outline.py that alters the result shows up here. If the change is intended, save the new masks (and have a look at
them) by running this file with the environment variable UPDATE_OUTLINE_FRAMES=1. In cmd:
    set UPDATE_OUTLINE_FRAMES=1
    .venv\Scripts\python -m pytest tests/test_outline_frames.py
"""
import os
from pathlib import Path

import numpy as np
import pytest

from poe2layout.files import imread, imwrite
from poe2layout.outline import DEFAULT_MAP_AREA, color_gains, map_area_mask, neutralize_color_cast, outline_mask

FRAMES = Path(__file__).parent / "frames"
NAMES = ["clearfell", "deshar_sand", "jungle_ruins", "excavation"]
UPDATE = os.environ.get("UPDATE_OUTLINE_FRAMES") == "1"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return float((a & b).sum() / max(1, (a | b).sum()))


def tinted(img: np.ndarray, kb: float, kg: float) -> np.ndarray:
    """A warm colour cast like a blue-light filter makes: blue and green turned down."""
    return (img.astype(np.float32) * np.array([kb, kg, 1.0], np.float32)).astype(np.uint8)


@pytest.fixture(scope="module", params=NAMES)
def frame(request):
    return request.param, imread(FRAMES / f"{request.param}.jpg")


def test_outline_matches_saved(frame):
    name, img = frame
    expected_path = FRAMES / f"{name}_outline.png"
    m = outline_mask(img)
    if UPDATE:
        imwrite(expected_path, m)
    expected = imread(expected_path)[:, :, 0]
    # a little slack, because JPEG decoding can differ by a few levels between OpenCV builds
    assert iou(m, expected) >= 0.97
    assert (m > 0).sum() > 20000                     # every frame shows plenty of map


def test_nothing_outside_the_map_area(frame):
    _, img = frame
    m = outline_mask(img)
    assert not m[~map_area_mask(*m.shape, DEFAULT_MAP_AREA)].any()


def test_untinted_frames_are_not_corrected(frame):
    _, img = frame
    assert color_gains(img) == (1.0, 1.0)


def test_strong_cast_is_corrected(frame):
    _, img = frame
    gb, gg = color_gains(tinted(img, 0.55, 0.75))
    assert abs(gb - 1 / 0.55) / (1 / 0.55) < 0.1 and abs(gg - 1 / 0.75) / (1 / 0.75) < 0.1


def test_cast_correction_brings_the_colours_back(frame):
    _, img = frame
    assert neutralize_color_cast(img) is img                  # nothing to correct
    fixed = neutralize_color_cast(tinted(img, 0.55, 0.75)).reshape(-1, 3).mean(0)
    want = img.reshape(-1, 3).mean(0)
    assert np.allclose(fixed[:2], want[:2], rtol=0.15) and fixed[2] == want[2]     # blue and green back, red kept


@pytest.mark.xfail(strict=True, reason="known weakness: the cast estimate from the HUD is off by 4-8% (the HUD's "
                   "own colours vary, e.g. the mana orb) and a mild cast is not corrected at all; with exact gains "
                   "the outline survives at IoU 0.85-0.96, with the estimate at 0.02-0.68")
@pytest.mark.parametrize("kb, kg", [(0.55, 0.75), (0.8, 0.9)])
def test_outline_survives_a_warm_cast(frame, kb, kg):
    _, img = frame
    assert iou(outline_mask(tinted(img, kb, kg)), outline_mask(img)) >= 0.8


@pytest.mark.parametrize("name, sides", [("panel_left", (True, False)), ("panels_both", (True, True)),
                                         *((n, (False, False)) for n in NAMES)])
def test_side_panels_on_real_frames(name, sides):
    # panel_left shows the market's filters (contents blacked out). panels_both is skills + inventory, with a tooltip
    # hiding part of the inventory's border
    from poe2layout.outline import panel_sides, view_shift
    img = imread(FRAMES / f"{name}.jpg")
    assert panel_sides(img) == sides
    assert view_shift(img) == (442 if sides == (True, False) else 0)


def test_no_outline_read_from_an_open_panel():
    from poe2layout.outline import PANEL_EDGE
    img = imread(FRAMES / "panel_left.jpg")
    img[300:1100, 200:700] = (0, 0, 0)
    for x in range(220, 680, 40):                       # blue lines on the panel (gem sockets, mod text)
        img[300:1100, x:x + 3] = (230, 140, 60)
    edge = round(PANEL_EDGE * img.shape[0])
    assert not outline_mask(img)[:, :edge].any()


def test_no_outline_read_from_an_open_right_panel():
    from poe2layout.outline import PANEL_EDGE
    img = imread(FRAMES / "panels_both.jpg")
    w = img.shape[1]
    img[300:1100, w - 700:w - 200] = (0, 0, 0)
    for x in range(w - 680, w - 220, 40):                   # blue lines on the inventory
        img[300:1100, x:x + 3] = (230, 140, 60)
    edge = round(PANEL_EDGE * img.shape[0])
    assert not outline_mask(img)[:, w - edge:].any()
