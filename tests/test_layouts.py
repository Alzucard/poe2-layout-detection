import itertools

import cv2
import numpy as np
import pytest

from conftest import make_map
from poe2layout.layouts import (
    DEFAULT_THRESHOLDS,
    TRANSFORMS,
    MapVariants,
    MapView,
    group_runs,
    is_mirrored,
    layout_names,
    layout_order,
    run_fit,
    transform_inverse,
    transform_map,
    transform_then,
)


def test_transforms_form_a_group():
    for a, b in itertools.product(TRANSFORMS, repeat=2):
        assert transform_then(a, b) in TRANSFORMS
    for a in TRANSFORMS:
        assert transform_then(a, transform_inverse(a)) == "id"
    assert [n for n in TRANSFORMS if is_mirrored(n)] == ["flip_h", "flip_v", "diag", "anti_diag"]
    assert transform_then("rot90", "rot90") == "rot180"


def test_transform_map_exact_flips():
    m = make_map(1)
    assert np.array_equal(transform_map(m, "flip_h"), m[:, ::-1])
    assert np.array_equal(transform_map(transform_map(m, "rot180"), "rot180"), m)
    assert transform_map(m, "rot90").dtype == np.uint8


@pytest.mark.parametrize("name", ["diag", "anti_diag", "rot90", "rot270"])
def test_transform_map_keeps_the_whole_map(name):
    m = np.zeros((200, 300), np.uint8)
    for y in (0, 197):
        for x in (0, 297):
            m[y:y + 3, x:x + 3] = 255                         # a dot in every corner
    out = transform_map(m, name)
    assert cv2.connectedComponents((out > 0).astype(np.uint8))[0] - 1 == 4      # no corner cut off
    assert out[0].any() and out[-1].any() and out[:, 0].any() and out[:, -1].any()   # cropped tight
    assert set(np.unique(out)) == {0, 255}
    assert set(np.unique(transform_map(m // 255, name))) == {0, 1}           # a 0/1 map stays 0/1


def test_footprint_fills_the_space_inside_the_walls():
    ring = np.zeros((400, 400), np.uint8)
    cv2.circle(ring, (200, 200), 150, 255, 3)
    f = MapView(ring).footprint
    assert f.shape == ring.shape
    assert f[200, 200] and f[200, 60]                         # inside the wall, far from it
    assert not f[5, 5] and not f[200, 5]                      # outside


def test_layout_names_and_order():
    names = list(itertools.islice(layout_names(set()), 28))
    assert names[:3] == ["A", "B", "C"] and names[25:28] == ["Z", "AA", "AB"]
    assert list(itertools.islice(layout_names({"A", "C"}), 2)) == ["B", "D"]
    assert sorted(["AA", "B", "Z", "A"], key=layout_order) == ["A", "B", "Z", "AA"]
    assert min({"AA", "Z"}, key=layout_order) == "Z"


def test_run_fit_same_mirrored_and_different():
    m = make_map(1)
    same, shape, orient = run_fit(MapVariants(m), MapVariants(m[:, ::-1].copy()))
    assert DEFAULT_THRESHOLDS.same_layout(same, shape)
    assert orient == "flip_h"
    other, shape2, _ = run_fit(MapVariants(m), MapVariants(make_map(2)))
    assert not DEFAULT_THRESHOLDS.same_layout(other, shape2)


def test_run_fit_either_way_round():
    m = make_map(1)
    small = transform_map(m[60:, 80:].copy(), "rot90")
    assert run_fit(MapVariants(m), MapVariants(small))[2] == "rot90"
    assert run_fit(MapVariants(small), MapVariants(m))[2] == "rot270"     # the turn back


@pytest.fixture(scope="module")
def maps():
    a, b = make_map(1), make_map(2)
    return {"r1": MapVariants(a), "r2": MapVariants(b), "r3": MapVariants(a[::-1].copy()),
            "r4": MapVariants(a[60:, 80:].copy())}


def test_group_runs_from_scratch(maps):
    out, changed = group_runs("Zone", list(maps.items()), {})
    assert changed
    assert out["r1"] == {"layout": "A", "orientation": "id", "mirrored": False}
    assert out["r2"]["layout"] == "B"
    assert out["r3"] == {"layout": "A", "orientation": "flip_v", "mirrored": True}
    assert out["r4"]["layout"] == "A" and out["r4"]["orientation"] == "id"


def test_group_runs_keeps_known(maps):
    known = {"r1": {"layout": "C", "orientation": "id", "mirrored": False},
             "r2": {"layout": "A", "orientation": "id", "mirrored": False}}
    out, changed = group_runs("Zone", [("r1", maps["r1"]), ("r2", maps["r2"])], known)
    assert not changed and out == known
    out, changed = group_runs("Zone", [("r1", maps["r1"]), ("r2", maps["r2"]), ("r3", maps["r3"])], known)
    assert changed and out["r3"]["layout"] == "C"
    # a run that's gone from disk gets dropped
    out, changed = group_runs("Zone", [("r1", maps["r1"])], known)
    assert changed and set(out) == {"r1"}


def test_run_linking_two_groups_merges_into_oldest():
    # two runs that explored different ends of one zone, and a third one that saw all of it
    full = np.hstack([make_map(5), make_map(6)])
    left, right = full[:, :1000].copy(), full[:, 400:].copy()
    assert not DEFAULT_THRESHOLDS.same_layout(*run_fit(MapVariants(left), MapVariants(right))[:2])
    known = {"L": {"layout": "B", "orientation": "id", "mirrored": False},
             "R": {"layout": "A", "orientation": "id", "mirrored": False}}
    runs = [("L", MapVariants(left)), ("R", MapVariants(right)), ("X", MapVariants(full))]
    out, _ = group_runs("Zone", runs, known)
    assert {g["layout"] for g in out.values()} == {"A"}


def test_runs_of_other_screen_sizes_are_not_compared(maps):
    runs = [("r1", maps["r1"]), ("r3", maps["r3"]), ("r4", maps["r4"])]
    screens = {"r1": (2560, 1440), "r3": (1920, 1080)}               # r4 was saved before we recorded sizes
    out, _ = group_runs("Zone", runs, {}, screens=screens)
    assert out["r1"]["layout"] != out["r3"]["layout"]
    assert out["r4"]["layout"] == out["r1"]["layout"]


def test_run_without_size_joins_groups_of_one_size_only(maps):
    # r4 (no size) is layout A, and r1 and r3 were filmed at other sizes, so they're B and C. A new run without a size
    # fits all three. It may link A to one sized group, but never to both, since that would mix 1440p and 1080p maps.
    known = {"r4": {"layout": "A", "orientation": "id", "mirrored": False},
             "r1": {"layout": "B", "orientation": "id", "mirrored": False},
             "r3": {"layout": "C", "orientation": "id", "mirrored": False}}
    runs = [("r4", maps["r4"]), ("r1", maps["r1"]), ("r3", maps["r3"]), ("r5", maps["r1"])]
    out, _ = group_runs("Zone", runs, known, screens={"r1": (2560, 1440), "r3": (1920, 1080)})
    assert out["r5"]["layout"] == out["r4"]["layout"] == out["r1"]["layout"] == "A"
    assert out["r3"]["layout"] == "C"
