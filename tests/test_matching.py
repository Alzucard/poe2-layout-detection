"""Finding where a smaller saved map sits inside a bigger one."""
import cv2
import numpy as np
import pytest

from conftest import make_map
from poe2layout.matching import agreement, coarse_features, locate, locate_ranked


@pytest.fixture(scope="module")
def full() -> np.ndarray:
    return make_map(3, w=900, h=700, turns=30)


def test_finds_a_cut_out_part(full):
    part = full[150:450, 200:550].copy()
    dx, dy, ag = locate(part, full)
    assert (dx, dy) == (200, 150)
    assert ag > 0.95


def test_part_reaching_past_the_edge(full):
    """The smaller map explored past the bigger map's top-left corner, so the shift is negative."""
    part = cv2.copyMakeBorder(full[0:300, 0:350], 40, 0, 60, 0, cv2.BORDER_CONSTANT)
    dx, dy, ag = locate(part, full)
    assert (dx, dy) == (-60, -40)
    assert ag > 0.95


@pytest.mark.parametrize("x, y", [(201, 151), (203, 149), (198, 147), (205, 153), (37, 61), (346, 299)])
def test_found_to_the_pixel_off_the_coarse_grid(full, x, y):
    dx, dy, _ = locate(full[y:y + 300, x:x + 350].copy(), full)
    assert (dx, dy) == (x, y)


def test_ranked_best_first(full):
    ranked = locate_ranked(full[150:450, 200:550].copy(), full, peaks=5)
    assert len(ranked) >= 2
    assert [r[0] for r in ranked] == sorted((r[0] for r in ranked), reverse=True)
    assert ranked[0][1:] == (200, 150)
    assert len({r[1:] for r in ranked}) == len(ranked)          # each candidate is another place


def test_caches_give_the_same_answer(full):
    part = full[150:450, 200:550].copy()
    cache, part_cache = {}, {}
    first = locate(part, full, cache=cache, partial_cache=part_cache)
    assert 4 in cache and 4 in part_cache
    assert locate(part, full, cache=cache, partial_cache=part_cache) == first == locate(part, full)


def test_coarse_features_cache_is_used():
    m = make_map(1)
    marker = np.zeros((1, 1), np.float32)
    assert coarse_features(m, 4, {4: marker}) is marker
    assert coarse_features(m, 4).shape == (m.shape[0] // 4, m.shape[1] // 4)


def square(x0: int, x1: int) -> np.ndarray:
    m = np.zeros((300, 300), np.uint8)
    cv2.rectangle(m, (x0, x0), (x1, x1), 255, 2)
    return m


def test_agreement_of_same_and_misplaced_maps():
    """Shifted by 20 px: still the same explored area, but the walls don't lie on top of each other anymore."""
    assert agreement(square(50, 250), square(50, 250)) == 1.0
    assert agreement(square(50, 250), square(70, 270)) < 0.2


def test_agreement_is_the_weaker_side():
    """b has every wall of a, plus one more inside the shared area, so a agrees fully and b only partly."""
    a = square(50, 250)
    b = a.copy()
    cv2.line(b, (60, 65), (240, 65), 255, 2)
    ag = agreement(a, b)
    assert 0.5 < ag < 0.95
    assert ag == agreement(b, a)


def test_agreement_tolerates_small_offsets():
    a = make_map(1)
    shifted = np.roll(a, 2, axis=1)
    assert agreement(a, shifted) > 0.95
    assert agreement(a, shifted, tol=1) < agreement(a, shifted)


def test_walls_only_one_map_explored_do_not_count():
    a = cv2.copyMakeBorder(square(50, 250), 0, 300, 0, 300, cv2.BORDER_CONSTANT)
    b = a.copy()
    cv2.rectangle(b, (400, 50), (550, 250), 255, 2)        # far from a's walls: b explored more, a never saw it
    cv2.rectangle(a, (50, 400), (250, 550), 255, 2)        # and the other way round
    assert agreement(a, b) == 1.0


def test_too_little_overlap_has_no_agreement():
    a = np.zeros((300, 300), np.uint8)
    cv2.line(a, (10, 10), (60, 10), 255, 1)               # well under 300 line pixels
    assert agreement(a, a) == 0.0
    b = a.copy()
    cv2.rectangle(b, (5, 5), (65, 15), 255, -1)            # enough pixels on b's side only
    assert agreement(a, b) == agreement(b, a) == 0.0
