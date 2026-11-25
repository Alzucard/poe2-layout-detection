"""Where the drawn route gets cut (teleports, gaps), and the label colours."""
import numpy as np
import pytest

from poe2layout.render import (
    FIXED_LABEL_COLORS,
    MAX_MISSING,
    MAX_STEP,
    PALETTE,
    label_colors,
    render_run_images,
    route_breaks,
)


def walk(n: int, step: int = 17) -> list[tuple[int, int]]:
    return [(i * step, 0) for i in range(n)]


def test_normal_walk_has_no_breaks():
    assert route_breaks(walk(6), list(range(6))) == [False] * 5


def test_teleport_breaks_the_route_once():
    path = walk(3) + [(5000, 0), (5017, 0)]
    assert route_breaks(path, list(range(5))) == [False, False, True, False]


def test_step_just_at_the_limit_is_walking():
    path = [(0, 0), (MAX_STEP, 0), (MAX_STEP + 1 + MAX_STEP + 1, 0)]
    assert route_breaks(path, [0, 1, 2]) == [False, True]


def test_frames_without_position_break_the_route():
    """A step normally spans 1 frame, and up to MAX_MISSING frames more is still the same walk."""
    assert route_breaks(walk(5), [0, 1, 2, 3, 4 + MAX_MISSING]) == [False] * 4
    assert route_breaks(walk(5), [0, 1, 2, 3, 5 + MAX_MISSING]) == [False] * 3 + [True]


def test_thinned_route_is_judged_by_its_stride():
    """With every 4th frame kept, steps of 4 frames and 4x the distance are normal walking."""
    times = [0, 4, 8, 12, 16]
    assert route_breaks(walk(5, step=4 * 17), times) == [False] * 4
    assert route_breaks(walk(5, step=4 * 17), [0, 4, 8, 12, 16 + MAX_MISSING + 1]) == [False] * 3 + [True]


def test_allowed_step_grows_with_the_frames_it_spans():
    """Over 4 frames a step may be 4x MAX_STEP long, so going well past MAX_STEP is still walking."""
    assert route_breaks(walk(4, step=4 * 30), [0, 4, 8, 12]) == [False] * 3


def test_stride_is_the_usual_step_not_the_shortest():
    """One short step (1 frame) mustn't make the normal 4-frame stride look like a gap, because the stride is the
    median. A 7-frame step is within stride + MAX_MISSING of 4, but not of 1."""
    assert route_breaks(walk(7), [0, 4, 8, 9, 13, 17, 24]) == [False] * 6


def test_repeated_frame_time_does_not_divide_by_zero():
    assert route_breaks([(0, 0), (50, 0), (50 + MAX_STEP + 1, 0)], [0, 0, 1]) == [False, True]


def test_short_routes():
    assert route_breaks([], []) == []
    assert route_breaks([(3, 4)], [7]) == []


@pytest.mark.parametrize("meta_times", [None, [0, 5, 10]])
def test_old_runs_without_matching_times_still_render(tmp_path, meta_times):
    """Runs saved before we kept path_t (missing, or thinned differently) get evenly spaced times."""
    meta = {"path": walk(5), "frames": 10, "path_t": meta_times}
    render_run_images(tmp_path, np.zeros((50, 100), np.uint8), meta)
    assert (tmp_path / "path.png").exists() and (tmp_path / "labels.png").exists()


def test_fixed_label_colors():
    colors = label_colors(["Waypoint", "Checkpoint", "Gate"])
    assert colors["Waypoint"] == FIXED_LABEL_COLORS["Waypoint"]
    assert colors["Checkpoint"] == FIXED_LABEL_COLORS["Checkpoint"]
    assert colors["Gate"] == PALETTE[0]


def test_other_labels_get_own_colors_whatever_the_order():
    names = ["Boss", "Gate", "Ladder", "Gate"]
    colors = label_colors(names)
    assert colors == label_colors(list(reversed(names)))
    assert len({colors["Boss"], colors["Gate"], colors["Ladder"]}) == 3
    assert set(colors.values()) <= set(PALETTE)


def test_palette_repeats_after_its_end():
    names = [f"L{i:02}" for i in range(len(PALETTE) + 1)]
    colors = label_colors(names)
    assert colors[names[-1]] == colors[names[0]] == PALETTE[0]


def test_fast_movement_is_still_walking():
    # Sandswept Marsh: sprinting moved 83 and 85 map px per frame, and the route got cut there. A step of 120 px per
    # frame is fast walking, while a teleport moves hundreds of px in one frame.
    assert route_breaks(walk(5, step=120), list(range(5))) == [False] * 4
    assert route_breaks([(0, 0), (85, 0), (400, 0)], [0, 1, 2]) == [False, True]


def test_teleport_after_lost_frames_breaks_the_route():
    # a checkpoint teleport: 2 frames lost in the fade, then 400 px away, which is more than fast walking covers
    path = walk(6) + [(17 * 5 + 400, 0)]
    assert route_breaks(path, [0, 1, 2, 3, 4, 5, 8]) == [False] * 5 + [True]


def test_sprint_across_lost_frames_is_still_walking():
    # sprinting at 85 px per frame while 3 or 4 frames weren't placed: we still know the route
    sprint = walk(6, step=85)
    assert route_breaks(sprint + [(85 * 5 + 340, 0)], [0, 1, 2, 3, 4, 5, 9]) == [False] * 6
    assert route_breaks(sprint + [(85 * 5 + 425, 0)], [0, 1, 2, 3, 4, 5, 10]) == [False] * 6
