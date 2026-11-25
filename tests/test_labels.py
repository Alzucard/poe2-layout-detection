import json
import tracemalloc

import numpy as np
import pytest

from poe2layout.labels import (
    ENTRANCE_GATE_NEAR,
    ENTRANCE_NEAR,
    LABEL_SAME_SPOT,
    OCR_UPSCALE,
    SCREEN_FIXED_LABELS,
    WAYPOINT_MIN_HITS,
    LabelReader,
    LabelVotes,
    came_through_gate,
    entrance_point,
    label_ok,
    trusted_labels,
    vocab_match,
    without_screen_fixed,
)
from poe2layout.ocr import TextBox
from poe2layout.zones import ZoneData

VOCAB = ["Checkpoint", "Waypoint", "The Grelwood", "Troubled Camp", "Keth", "Well"]


@pytest.mark.parametrize("text, expected", [
    ("Waypoint", "Waypoint"),
    ("heckpoint", "Checkpoint"),
    ("The Grelw00d", "The Grelwood"),
    ("roubl d Camp", "Troubled Camp"),
    ("Keth", "Keth"),
    ("Well", "Well"),
    ("Kexx", None),               # short names have to be read completely
    ("Wel", None),
    ("you have been kicked from the party Waypoint chat", None),
    ("ab", None),
])
def test_vocab_match(text, expected):
    assert vocab_match(text, VOCAB) == expected


def lb(text, x, y, hits=5, looks=5):
    return {"text": text, "x": x, "y": y, "hits": hits, "looks": looks}


def test_votes_merge_and_absorb():
    v = LabelVotes()
    v.add("Waypoint", 100, 100)
    v.add("Waypoint", 110, 104)        # the same label, read again nearby
    v.add("Waypoint", 400, 400)        # another one
    v.add("abc", 0, 0)                 # too short
    assert [(it.hits, round(it.x)) for it in v.items] == [(2, 105), (1, 400)]
    v.look(0, 0, 200, 200)
    v.look(0, 0, 200, 200, skip=[(0, 0, 200, 200)])
    assert v.items[0].looks == 1 and v.items[1].looks == 0
    other = LabelVotes()
    other.add("Waypoint", 0, 0)
    v.absorb(other.items, 105, 100)
    assert v.items[0].hits == 3
    v.shift(10, 0)
    assert round(v.items[0].x) == 115
    assert round(v.shifted_copy(-15, 0)[0].x) == 100 and round(v.items[0].x) == 115


def test_merged_reads_average_their_place():
    v = LabelVotes()
    v.add("Waypoint", 100, 100)
    v.add("Waypoint", 140, 100)                  # 40 px away, so another label
    v.add("Waypoint", 100, 139.9)                # within 40 px of the first
    assert [(it.hits, it.x, it.y) for it in v.items] == [(2, 100, pytest.approx(119.95)), (1, 140, 100)]
    v.add("Well", 10, 10)                         # 4 letters is long enough
    assert v.items[-1].key == "well"
    v.look(100, 100, 140, 200)                    # from the left/top edge up to the right one, not including it
    assert [it.looks for it in v.items] == [1, 0, 0]
    v.look(0, 0, 200, 200, skip=[(100, 100, 140, 120)])
    assert [it.looks for it in v.items] == [1, 1, 1]
    v.shift(0, 5)
    assert v.items[0].y == pytest.approx(124.95)


def test_absorbed_labels_are_weighted_by_their_reads():
    v, other = LabelVotes(), LabelVotes()
    v.add("Waypoint", 100, 100)
    v.add("Waypoint", 100, 100)
    v.add("Waypoint", 120, 100)                  # 2 reads at 100 and 1 at 120 make a label of 3 reads at 106.67
    other.add("Waypoint", 0, 0)
    other.add("Waypoint", 0, 0)
    other.items[0].looks = 3
    v.absorb(other.items, 130, 120)               # 2 reads at (130, 120)
    assert len(v.items) == 1
    it = v.items[0]
    assert (it.hits, it.looks) == (5, 3) and (it.x, it.y) == pytest.approx((116.0, 108.0))


def test_trusted_labels():
    assert label_ok(lb("Waypoint", 0, 0, hits=2, looks=8))
    assert not label_ok(lb("Waypoint", 0, 0, hits=2, looks=9))
    assert not label_ok(lb("Waypoint", 0, 0, hits=1, looks=1))
    out = trusted_labels([lb("Waypoint", 0, 0, hits=3), lb("Waypoint", 50, 0, hits=9), lb("Waypoint", 500, 0)])
    assert [(o["x"], o["hits"]) for o in out] == [(50, 9), (500, 5)]
    out = trusted_labels([lb("Waypoint", 1000, 0, hits=9), lb("Waypoint", 1000 + LABEL_SAME_SPOT, 0)])
    assert len(out) == 2                         # just far enough apart to be two places
    assert len(trusted_labels([lb("Waypoint", 1000, 500), lb("Waypoint", 1050, 500)])) == 1


def test_entrance_prefers_came_from_gate():
    labels = [lb("Waypoint", 120, 0), lb("The Grelwood", 0, 150)]
    assert entrance_point([0, 0], labels, "The Grelwood") == [0.0, 150.0]


def test_entrance_waypoint_after_teleport():
    labels = [lb("Waypoint", 120, 0), lb("Red Vale", 600, 0)]
    assert entrance_point([0, 0], labels, "Clearfell") == [120.0, 0.0]


def test_entrance_far_gate_vs_misread_waypoint():
    labels = [lb("Waypoint", 170, 0, hits=2), lb("The Grelwood", 310, 0, hits=8)]
    assert entrance_point([0, 0], labels, "The Grelwood") == [310.0, 0.0]


def test_entrance_other_gate_then_first_point():
    labels = [lb("Red Vale", 97, 0)]
    assert entrance_point([0, 0], labels, "Clearfell", exits={"red vale"}) == [97.0, 0.0]
    assert entrance_point([5, 6], [], None) == [5.0, 6.0]
    assert entrance_point(None, labels) is None


def test_entrance_scores_distance_against_reads():
    first = (1000, 500)
    labels = [lb("Waypoint", 1100, 500, hits=2), lb("Waypoint", 1000, 620.04, hits=25)]
    assert entrance_point(first, labels) == [1000.0, 620.0]       # 20 px farther away, but read far more often
    edge = [lb("Waypoint", 1000 + ENTRANCE_NEAR, 500, hits=2)]
    assert entrance_point(first, edge) == [1000.0 + ENTRANCE_NEAR, 500.0]
    assert entrance_point(first, [lb("Point", 1010, 500)]) == [1000.0, 500.0]   # not the Waypoint


def test_entrance_well_read_waypoint_beats_a_farther_gate():
    labels = [lb("Waypoint", 100, 0, hits=WAYPOINT_MIN_HITS), lb("The Grelwood", 310, 0, hits=8)]
    assert entrance_point([0, 0], labels, "The Grelwood") == [100.0, 0.0]


def test_came_through_gate():
    assert came_through_gate([lb("The Grelwood", 300, 0)], [0, 0], "Grelwood")
    assert came_through_gate([lb("The Grelwood", ENTRANCE_GATE_NEAR, 0)], [0, 0], "The Grelwood")
    assert not came_through_gate([lb("The Grelwood", ENTRANCE_GATE_NEAR + 1, 0)], [0, 0], "The Grelwood")
    assert not came_through_gate([lb("The Grelwood", 10, 0, hits=1)], [0, 0], "The Grelwood")   # unconfirmed
    assert not came_through_gate([lb("Waypoint", 10, 0)], [0, 0], "The Grelwood")
    assert not came_through_gate([lb("The Grelwood", 10, 0)], None, "The Grelwood")
    assert not came_through_gate([lb("The Grelwood", 10, 0)], [0, 0], None)


@pytest.fixture
def zones(tmp_path):
    f = tmp_path / "zones.json"
    f.write_text(json.dumps({"Act 1": [{"name": "The Grelwood", "connections": ["Red Vale"],
                                        "points_of_interest": [{"name": "Shrine"}]}]}), "utf-8")
    return ZoneData(f)


def test_vocabulary_has_gates_the_way_in_and_points_of_interest(zones):
    r = LabelReader(None, zones)
    assert r.vocabulary("The Grelwood", "Clearfell") == ["Checkpoint", "Waypoint", "Red Vale", "Clearfell", "Shrine"]
    assert r.vocabulary("The Grelwood", "the red vale") == ["Checkpoint", "Waypoint", "Red Vale", "Shrine"]
    assert r.vocabulary(None) == ["Checkpoint", "Waypoint"]


class FakeOcr:
    def __init__(self, boxes):
        self.boxes, self.calls = boxes, []

    def read_boxes(self, img, upscale):
        self.calls.append((img, upscale))
        return self.boxes


def test_read_keeps_known_names_in_screen_pixels(zones):
    ocr = FakeOcr([TextBox("Waypont", 100, 50, 80, 20), TextBox("selling my stash for chaos", 900, 300, 200, 20)])
    frame = np.full((1000, 2000, 3), 255, np.uint8)
    assert LabelReader(ocr, zones).read(frame, "The Grelwood") == [("Waypoint", 100, 110, 80, 20)]
    ((img, upscale),) = ocr.calls
    assert upscale == OCR_UPSCALE and img.shape == (800, 2000, 3)        # rows 60..860 of the screen
    assert img[700, 100].max() == 0 and img[100, 1900].max() == 0          # chat and zone panel blanked out
    assert img[100, 100].min() == 255 and img[700, 1000].min() == 255
    assert frame.min() == 255                                              # the frame itself isn't touched


def test_votes_to_json_has_all_labels_trusted_only_confirmed():
    v = LabelVotes()
    v.add("Waypoint", 10.04, 20.06)
    v.add("Waypoint", 10.04, 20.06)
    v.add("Shrine", 300, 300)
    assert v.to_json() == [{"text": "Waypoint", "x": 10.0, "y": 20.1, "hits": 2, "looks": 0},
                           {"text": "Shrine", "x": 300, "y": 300, "hits": 1, "looks": 0}]
    assert trusted_labels(v.to_json()) == v.to_json()[:1]


def test_screen_fixed_text_is_dropped():
    """The quest tracker ("Ancient Beacons", read as "Ancient Ruin") stays at one screen spot while you walk. Its
    reads land on the map all along your route, one label for every stretch you walk (34 of them in Cemetery of the
    Eternals)."""
    v = LabelVotes()
    tracker = (2330, 380)                                   # screen px, the same in every frame
    for step in range(6):                                   # walking: the frame's map offset moves 100 px per read
        ox = 100 * step
        v.add("Ancient Ruin", ox + tracker[0] * 0.5, 500 + tracker[1] * 0.5, screen=(tracker[0] + step % 3, tracker[1]))
        v.add("Ancient Ruin", ox + tracker[0] * 0.5, 500 + tracker[1] * 0.5, screen=(tracker[0] - 4, tracker[1] + 2))
        # the real label at map (900, 900) moves across the screen as you walk
        v.add("Ancient Ruin", 900, 900, screen=((900 - ox) / 0.5, (900 - 500) / 0.5))
    # if a real label is read once at the tracker's screen spot, only that read goes, not the label
    v.add("Ancient Ruin", 900, 900, screen=tracker)
    [kept] = without_screen_fixed([v.items])
    assert [(it.x, it.y, it.hits) for it in kept] == [(900, 900, 6)]


def test_same_screen_spot_of_a_few_labels_is_kept():
    """Real labels of one name (Checkpoints) are sometimes read at the same screen spot. They're kept as long as
    fewer than SCREEN_FIXED_LABELS labels share it, and reads without a screen spot (older code) always count."""
    v = LabelVotes()
    for i in range(SCREEN_FIXED_LABELS - 1):
        v.add("Checkpoint", 200 * i, 0, screen=(1000, 500))
        v.add("Checkpoint", 200 * i, 0, screen=(1000 + 300 * i, 600))
    v.add("Waypoint", 0, 0)
    assert [it.hits for it in without_screen_fixed([v.items])[0]] == [2] * (SCREEN_FIXED_LABELS - 1) + [1]


def test_screen_fixed_text_is_found_across_floors():
    """The quest tracker sits at the same screen spot on every floor. Two labels on each of two floors are four in
    all, so they go, even though neither floor has four on its own."""
    floors = [LabelVotes(), LabelVotes()]
    for votes in floors:
        for step in range(2):
            for _ in range(3):
                votes.add("Ancient Ruin", 100 * step, 0, screen=(2330, 380))
    assert without_screen_fixed([f.items for f in floors]) == [[], []]


def test_screen_fixed_check_keeps_memory_low_with_many_reads():
    """A long visit reads one name thousands of times. Comparing every read with every other one took 216 MB for
    3000 reads."""
    v = LabelVotes()
    for i in range(3000):
        v.add("Checkpoint", 0, 0, screen=(i % 300 * 7, i // 300 * 7))
    tracemalloc.start()
    try:
        [kept] = without_screen_fixed([v.items])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert [it.hits for it in kept] == [3000]
    assert peak < 5_000_000
