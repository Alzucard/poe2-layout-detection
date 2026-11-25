import numpy as np
import pytest

from conftest import frame_at, make_world
from poe2layout.outline import PANEL_EDGE, outline_mask
from poe2layout.render import MAX_MISSING
from poe2layout.stitcher import MERGE_EVERY, MapPiece, StepResult, Stitcher

STEP = (25, 12)        # how far the camera moves per frame, in screen px


def pan(stitcher, world, n, start=(500, 400), step=STEP):
    return [stitcher.add(frame_at(world, start[0] + step[0] * i, start[1] + step[1] * i)) for i in range(n)]


def test_outline_mask_finds_blue_lines(world):
    m = outline_mask(frame_at(world, 0, 0))
    assert m.dtype == np.uint8 and set(np.unique(m)) <= {0, 255}
    assert (m > 0).sum() > 20000
    assert not outline_mask(np.full((1080, 1920, 3), 12, np.uint8)).any()


def test_tracks_a_panning_camera(world):
    s = Stitcher()
    results = pan(s, world, 30)
    assert results[0] is StepResult.STARTED
    assert all(r is StepResult.TRACKED for r in results[1:])
    moves = np.diff(np.array(s.path), axis=0)
    # the canvas runs at half the screen scale
    assert np.allclose(moves, [STEP[0] * s.scale, STEP[1] * s.scale], atol=1.0)
    m, _ = s.cropped_map()
    assert (m > 0).sum() > 5000


def test_dark_frames_are_skipped(world):
    s = Stitcher()
    assert s.add(np.full((1080, 1920, 3), 12, np.uint8)) is StepResult.SKIPPED
    assert not StepResult.SKIPPED.placed and StepResult.TRACKED.placed


def test_refinds_position_after_gap(world):
    s = Stitcher()
    pan(s, world, 10)
    end = s.path[-1]
    s.mark_gap()
    # come back after a while, somewhere the camera has already shown
    r = s.add(frame_at(world, 500 + STEP[0] * 4, 400 + STEP[1] * 4))
    assert r.placed
    assert abs(s.path[-1][0] - (end[0] - 5 * STEP[0] * s.scale)) <= 2
    assert abs(s.path[-1][1] - (end[1] - 5 * STEP[1] * s.scale)) <= 2


BLACK = np.zeros((1080, 1920, 3), np.uint8)


def test_black_screen_starts_a_new_floor(world):
    # Temple of Kopec: the stairs fade to black, and the next floor's map has nothing in common with this one
    s = Stitcher()
    pan(s, world, 40)
    assert s.add(BLACK) is StepResult.FADED
    pan(s, make_world(seed=5), 32)
    s.finalize()
    floors = s.floors(min_frames=30)
    assert [f.frames for f in floors] == [40, 32]        # in the order they were walked, and both are kept
    assert floors[0].path_t[0] < floors[1].path_t[0]


def test_black_screen_on_the_same_floor_merges_back(world):
    # a checkpoint teleport fades to black too, but it lands back on map we already know
    s = Stitcher()
    pan(s, world, 40)
    s.add(BLACK)
    pan(s, world, 25, start=(500 + STEP[0] * 10, 400 + STEP[1] * 10))
    s.finalize()
    assert [f.frames for f in s.floors(min_frames=1)] == [65]


def test_black_screen_before_a_map_changes_nothing(world):
    s = Stitcher()
    assert s.add(BLACK) is StepResult.SKIPPED
    pan(s, world, 5)
    assert s.frames == 5


def test_reset_and_other_world():
    s = Stitcher()
    pan(s, make_world(seed=3), 5)
    s.reset()
    assert s.frames == 0 and s.bbox() is None


def with_side_panel(frame, side):
    """The frame with a side panel (inventory, skills, market...) drawn over one side, the way the game does it."""
    out = frame.copy()
    h, w = out.shape[:2]
    edge = round(PANEL_EDGE * h)
    if side == "left":
        out[:, :edge] = 20
        out[:, edge - 2:edge] = 200
    else:
        out[:, w - edge:] = 20
        out[:, w - edge:w - edge + 2] = 200
    return out


def test_route_stays_with_the_player_when_a_side_panel_opens(world):
    # A side panel pushes the game view sideways, so the player stands in the middle of the part that's still visible
    # instead of the middle of the screen. The map still gets placed from its walls, but the route has to follow the
    # player, not the screen centre (otherwise path.png jumps straight sideways while a panel is open).
    s = Stitcher()
    pan(s, world, 30)
    x, y = 500 + STEP[0] * 29, 400 + STEP[1] * 29     # where the player stands now (screen centre, no panel)
    shift = round(PANEL_EDGE * 1080) // 2           # the view moves by half the panel's width
    assert s.add(with_side_panel(frame_at(world, x - shift, y), "left")) is StepResult.TRACKED
    assert s.add(with_side_panel(frame_at(world, x + shift, y), "right")) is StepResult.TRACKED
    assert s.add(frame_at(world, x, y)) is StepResult.TRACKED
    still = np.array(s.path[-4:])                      # the player didn't move in these four frames
    assert np.abs(still - still[0]).max() <= 2, still


def test_map_started_with_a_side_panel_open(world):
    s = Stitcher()
    x, y = 900, 700
    shift = round(PANEL_EDGE * 1080) // 2
    assert s.add(with_side_panel(frame_at(world, x - shift, y), "left")) is StepResult.STARTED
    assert s.add(frame_at(world, x, y)) is StepResult.TRACKED
    assert np.abs(np.array(s.path[1]) - s.path[0]).max() <= 2, s.path


def test_route_times_count_every_frame(world):
    s = Stitcher()
    pan(s, world, 5)
    s.add(np.full((1080, 1920, 3), 12, np.uint8))     # skipped, there's no map
    pan(s, world, 1, start=(500 + 25 * 5, 400 + 12 * 5))
    assert np.diff(s.path_t).tolist() == [1, 1, 1, 1, 2]     # the gap shows up in the times


@pytest.mark.xfail(strict=True, reason="known bug: a frame that motion puts up to SNAP_PX past the canvas edge is "
                   "snapped onto the edge (_match_in clips its window to the canvas) instead of moving the map first, "
                   "so it is painted up to SNAP_PX off")
def test_map_moves_back_when_it_reaches_the_canvas_edge(world):
    big, small = Stitcher(), Stitcher(canvas_size=1200)      # 960 px wide frames start 120 px from the edge
    pan(big, world, 12)
    assert all(r.placed for r in pan(small, world, 12))
    assert small.shifted[0] < 0                              # moved left, back into the middle
    assert np.array_equal(small.cropped_map()[0], big.cropped_map()[0])
    d = np.array(small.path) - np.array(big.path)           # the whole route moved along with the map
    assert (d == d[0]).all()
    assert tuple(np.subtract(small.offset, big.offset)) == tuple(d[0])


def test_map_bigger_than_the_canvas_is_not_placed(world):
    s = Stitcher(canvas_size=1000)
    results = pan(s, world, 12)
    assert results[0] is StepResult.STARTED and not results[-1].placed


def test_map_lost_for_good_is_not_another_floor(world):
    # tracking was lost without a black screen, so the pieces are parts of one floor and the biggest is its map
    s = Stitcher()
    pan(s, world, 40)
    results = pan(s, make_world(seed=5), 80)
    assert StepResult.NEW_PIECE in results
    s.finalize()
    floors = s.floors(min_frames=30)
    assert len(floors) == 1 and floors[0].frames == s.frames > 40


def test_gap_counts_as_time_without_a_position(world):
    s = Stitcher()
    pan(s, world, 10)
    s.mark_gap()
    s.add(frame_at(world, 500 + STEP[0] * 4, 400 + STEP[1] * 4))
    assert s.path_t[-1] - s.path_t[-2] > 1 + MAX_MISSING             # we don't draw the route across the gap


def test_floor_you_came_in_on_stays_first_after_it_merges_back(world):
    # Kopec: floor A, stairs up (black) to floor B, stairs down (black) back to A, so the third map is floor A again
    s = Stitcher()
    pan(s, world, 40)
    s.add(BLACK)
    pan(s, make_world(seed=5), 40)
    s.add(BLACK)
    pan(s, world, 25, start=(500 + STEP[0] * 10, 400 + STEP[1] * 10))
    s.finalize()
    floors = s.floors(min_frames=30)
    assert len(floors) == 2
    assert floors[0].path_t[0] == 1 and floors[0].frames == 65            # floor A, with the entrance, comes first


def test_resumed_instance_keeps_both_floors(world):
    # visit 1 ends on floor B, with floor A as the bigger map; then we're back in the instance on floor B (no black
    # screen)
    floor_b = make_world(seed=5)
    s = Stitcher()
    pan(s, world, 60)
    s.add(BLACK)
    pan(s, floor_b, 40)
    s.finalize()
    assert len(s.floors(min_frames=30)) == 2
    s.mark_gap()
    pan(s, floor_b, 50, start=(500 + STEP[0] * 39, 400 + STEP[1] * 39))
    s.finalize()
    floors = s.floors(min_frames=30)
    assert len(floors) == 2
    assert floors[0].path_t[0] == 1                                    # floor A, where you came in, is still map.png


def test_last_merge_runs_when_the_video_ends_on_a_black_screen(world):
    s = Stitcher()
    pan(s, world, 40, start=(100, 100))                                 # piece 1
    pan(s, world, 9, start=(2000, 1900), step=(0, 0))                   # lost for good: a new piece far away
    pan(s, world, 17, start=(1960, 1840), step=(-40, -60))              # walking back over piece 1
    assert s.pieces and s.frames < MERGE_EVERY
    s.add(BLACK)                                                        # leaving the zone fades to black
    assert s.finalize() == 1
    assert not s.pieces and s.frames > 40


def test_map_started_after_a_gap_can_use_motion(world):
    # a portal: the black screen gets filmed, then the town trip is a gap, and back in the zone the map starts over
    s = Stitcher()
    pan(s, world, 20)
    s.add(BLACK)
    s.mark_gap()
    results = pan(s, world, 8, start=(500 + STEP[0] * 19, 400 + STEP[1] * 19))
    assert results == [StepResult.STARTED] + [StepResult.TRACKED] * 7


def test_floor_map_is_the_piece_with_most_map_not_most_frames(world):
    # the position is lost for good while you stand still for a long time (a boss fight): lots of frames, little map
    s = Stitcher()
    pan(s, world, 30, step=(60, 30))                                    # exploring: lots of map in few frames
    pan(s, world, 60, start=(3000, 2200), step=(0, 0))                  # lost for good, then standing still
    assert len(s.pieces) == 1 and s.frames > s.pieces[0].frames
    s.finalize()
    explored = s.floors(min_frames=20)[0]
    assert explored.path_t[0] == 1


def test_floor_is_kept_when_its_biggest_piece_has_too_few_frames():
    # a short sprint can show more map than a longer piece of the same floor, but the longer one is still its map
    def piece(frames, px, floor):
        img = np.zeros((100, 100), np.uint8)
        img.flat[:px] = 255
        return MapPiece(img, img.copy(), img.copy(), frames, [(0, 0)], [frames], [], floor)

    s = Stitcher()
    s.pieces = [piece(60, 3000, 0), piece(40, 3000, 1), piece(25, 6000, 1)]
    floors = s.floors(min_frames=30)
    assert [(p.floor, p.frames) for p in floors] == [(1, 40), (0, 60)]
