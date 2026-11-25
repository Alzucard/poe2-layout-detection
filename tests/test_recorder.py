import json
import time

import numpy as np
import pytest

from poe2layout import recorder as recorder_module
from poe2layout.analyzer import leftover_videos, read_visit_file
from poe2layout.config import config_from_dict
from poe2layout.recorder import ZoneRecorder, ZoneVideo
from poe2layout.zones import Instance

FRAME = np.zeros((64, 96, 3), np.uint8)
A = ("Clearfell", Instance("G1_2", 1))
A2 = ("Clearfell", Instance("G1_2", 2))
TOWN = ("Clearfell Encampment", Instance("G1_town", 9))


@pytest.fixture
def recorder(tmp_path):
    cfg, _ = config_from_dict({"data_dir": str(tmp_path), "recorder": {"manual": False}})   # films every zone
    done = []
    r = ZoneRecorder(cfg, done.append)
    yield r, done
    r._finish()


def film(r, n=2):
    for i in range(n):
        assert r.video.write(FRAME, float(i))


def test_visit_file_roundtrip(tmp_path):
    v = ZoneVideo(tmp_path, "HFYU", 2.0, "Clearfell", Instance("G1_2", 1), "Clearfell Encampment")
    v.write(FRAME, 1.0)
    v.gap()
    v.write(FRAME, 5.0)
    assert not v.write(np.zeros((10, 10, 3), np.uint8), 6.0)      # a different size
    path = v.close()
    assert path.suffix == ".avi" and path.exists()
    rec = read_visit_file(path.with_suffix(".jsonl"))
    assert (rec.zone, rec.instance, rec.came_from) == ("Clearfell", Instance("G1_2", 1), "Clearfell Encampment")
    assert rec.times == [1.0, 5.0] and rec.gaps == {1}


def test_empty_video_is_removed(tmp_path):
    v = ZoneVideo(tmp_path, "HFYU", 2.0, "Clearfell", Instance("G1_2", 1), None)
    assert v.close() is None
    assert list(tmp_path.iterdir()) == []


def test_visits_started_in_the_same_second_get_numbered(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_module.time, "strftime", lambda fmt: "20261122_120000")
    videos = [ZoneVideo(tmp_path, "HFYU", 2.0, "Clearfell", Instance("G1_2", 1), None) for _ in range(3)]
    assert [v.visit_file.name for v in videos] == [
        "20261122_120000_Clearfell.jsonl", "20261122_120000_Clearfell_2.jsonl",
        "20261122_120000_Clearfell_3.jsonl"]
    for v in videos:
        v.close()


def test_video_size_in_gb(tmp_path):
    v = ZoneVideo(tmp_path, "HFYU", 2.0, "Clearfell", Instance("G1_2", 1), None)
    assert v.gb() == 0.0                                         # nothing written yet, so no file
    v.write(FRAME, 0.0)
    v.close()
    assert v.gb() == v.path.stat().st_size / 1e9 > 0


def test_recorder_films_at_the_configured_fps(tmp_path):
    cfg, _ = config_from_dict({"data_dir": str(tmp_path), "recorder": {"fps": 4, "manual": False}})
    r = ZoneRecorder(cfg, lambda p: None)
    assert r.interval == 0.25
    r._change_visit(A)
    assert r.video.fps == 4.0
    assert r.get_state()["status"] == "recording" and r.get_state()["frames"] == 0
    r._finish()


def test_town_pauses_and_same_instance_continues(recorder):
    r, done = recorder
    r._change_visit((None, None))
    r._change_visit(A)
    first = r.video
    assert first is not None and first.visit == A
    film(r)
    r._change_visit(TOWN)
    assert r.video is first and done == []                       # paused, not finished
    r._change_visit((None, None))                                # character select
    assert r.video is first
    r._change_visit(A)
    assert r.video is first
    lines = first.visit_file.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1]) == {"gap": True}
    assert json.loads(lines[0])["came_from"] is None


def test_new_instance_finishes_video(recorder):
    r, done = recorder
    r._change_visit(TOWN)
    assert r.video is None                                       # towns aren't filmed
    r._change_visit(A)
    film(r)
    r._change_visit(A2)                                          # same zone, new instance
    assert len(done) == 1 and done[0].exists()
    assert r.video.visit == A2
    assert json.loads(r.video.visit_file.read_text(encoding="utf-8").splitlines()[0])["came_from"] == "Clearfell"
    assert read_visit_file(done[0].with_suffix(".jsonl")).came_from == "Clearfell Encampment"


def test_leftover_videos(tmp_path):
    for name in ("b", "a"):
        (tmp_path / f"{name}.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / f"{name}.avi").write_bytes(b"x")
    (tmp_path / "orphan.jsonl").write_text("{}\n", encoding="utf-8")
    assert [p.name for p in leftover_videos(tmp_path)] == ["a.avi", "b.avi"]
    assert not (tmp_path / "orphan.jsonl").exists()


def test_disk_full(recorder, tmp_path):
    r, _ = recorder
    r.folder = tmp_path
    r.min_free_gb = 1e12
    assert r._disk_full()
    r.min_free_gb = 0.0
    assert not r._disk_full()
    r.folder = tmp_path / "missing"                               # unreadable, keep filming
    r.min_free_gb = 1e12
    assert not r._disk_full()


class FakeGrabber:
    def __init__(self, monitor):
        self.monitor = monitor

    def grab(self):
        return FRAME


class FakeWindow:
    def __init__(self, rect):
        self.front, self.reason = True, "game not in front"

    def game_in_front(self):
        return self.front


def wait_for(r, **want):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        state = r.get_state()
        if all(state.get(k) == v for k, v in want.items()):
            return state
        time.sleep(0.01)
    raise AssertionError(f"state {r.get_state()} never had {want}")


def test_loop_films_only_with_the_game_in_front_and_space_left(recorder, monkeypatch):
    r, done = recorder
    monkeypatch.setattr(recorder_module, "Grabber", FakeGrabber)
    monkeypatch.setattr(recorder_module, "GameWindowCheck", FakeWindow)
    monkeypatch.setattr(recorder_module, "mss_rect", lambda monitor: (0, 0, 96, 64))
    monkeypatch.setattr(recorder_module, "DISK_CHECK_EVERY", 0.0)
    full = [False]
    r._disk_full = lambda: full[0]
    r.interval = 0.01
    r.start()
    try:
        wait_for(r, result="no zone", filming=False)
        r.set_zone(*A)
        wait_for(r, result="filming", filming=True)
        r._window.front = False
        wait_for(r, result="game not in front", filming=False)
        r._window.front = True
        full[0] = True
        wait_for(r, result="disk full", filming=False)
        full[0] = False
        wait_for(r, result="filming", filming=True)
        r.set_zone(*TOWN)
        wait_for(r, result="town", filming=False)
        assert done == []                                         # paused in town, not finished
    finally:
        r.stop()
        r.join(timeout=5)
    assert not r.is_alive()
    assert len(done) == 1 and read_visit_file(done[0].with_suffix(".jsonl")).zone == "Clearfell"


@pytest.mark.parametrize("cause", ["game not in front", "disk full"])
def test_pause_inside_a_visit_is_a_gap(recorder, monkeypatch, cause):
    r, done = recorder
    monkeypatch.setattr(recorder_module, "Grabber", FakeGrabber)
    monkeypatch.setattr(recorder_module, "GameWindowCheck", FakeWindow)
    monkeypatch.setattr(recorder_module, "mss_rect", lambda monitor: (0, 0, 96, 64))
    monkeypatch.setattr(recorder_module, "DISK_CHECK_EVERY", 0.0)
    monkeypatch.setattr(recorder_module, "GAP_AFTER", 0.0)
    full = [False]
    r._disk_full = lambda: full[0]
    r.interval = 0.01
    r.set_zone(*A)
    r.start()
    try:
        wait_for(r, result="filming", filming=True)
        if cause == "disk full":
            full[0] = True
        else:
            r._window.front = False
        wait_for(r, result=cause, filming=False)
        before = r.get_state()["frames"]
        full[0], r._window.front = False, True
        end = time.monotonic() + 5
        while r.get_state()["frames"] <= before and time.monotonic() < end:
            time.sleep(0.01)
    finally:
        r.stop()
        r.join(timeout=5)
    record = read_visit_file(done[0].with_suffix(".jsonl"))
    assert record.gaps == {before}                    # the first frame after the pause follows a gap


def test_short_focus_blip_is_not_a_gap(recorder, monkeypatch):
    # an alt-tab or overlay for a moment: a few frames are missing, but the route goes on
    r, done = recorder
    monkeypatch.setattr(recorder_module, "Grabber", FakeGrabber)
    monkeypatch.setattr(recorder_module, "GameWindowCheck", FakeWindow)
    monkeypatch.setattr(recorder_module, "mss_rect", lambda monitor: (0, 0, 96, 64))
    monkeypatch.setattr(recorder_module, "GAP_AFTER", 60.0)
    r._disk_full = lambda: False
    r.interval = 0.01
    r.set_zone(*A)
    r.start()
    try:
        wait_for(r, result="filming", filming=True)
        r._window.front = False
        wait_for(r, result="game not in front", filming=False)
        before = r.get_state()["frames"]
        r._window.front = True
        end = time.monotonic() + 5
        while r.get_state()["frames"] <= before and time.monotonic() < end:
            time.sleep(0.01)
    finally:
        r.stop()
        r.join(timeout=5)
    assert read_visit_file(done[0].with_suffix(".jsonl")).gaps == set()


class MonitorGrabber:
    """The monitor as the system reports it. It changes when the display resolution changes."""
    rect = (0, 0, 96, 64)

    def __init__(self, monitor):
        self.monitor = MonitorGrabber.rect
        self.size = (self.monitor[3], self.monitor[2])

    def grab(self):
        return np.zeros((*self.size, 3), np.uint8)

    def close(self):
        pass


class MonitorWindow:
    """The game is in front, on a monitor of the size given by `screen`."""
    screen = (0, 0, 96, 64)

    def __init__(self, rect):
        self.monitor, self.reason = rect, ""

    def game_in_front(self):
        ok = self.monitor == MonitorWindow.screen
        self.reason = "" if ok else "game on another monitor"
        return ok


def test_resolution_change_keeps_filming(recorder, monkeypatch):
    r, done = recorder
    monkeypatch.setattr(recorder_module, "Grabber", MonitorGrabber)
    monkeypatch.setattr(recorder_module, "GameWindowCheck", MonitorWindow)
    monkeypatch.setattr(recorder_module, "mss_rect", lambda monitor: monitor)
    monkeypatch.setattr(recorder_module, "MONITOR_CHECK_EVERY", 0.0)
    monkeypatch.setattr(MonitorGrabber, "rect", (0, 0, 96, 64))
    monkeypatch.setattr(MonitorWindow, "screen", (0, 0, 96, 64))
    r._disk_full = lambda: False
    r.interval = 0.01
    r.set_zone(*A)
    r.start()
    try:
        wait_for(r, result="filming", filming=True)
        MonitorGrabber.rect = MonitorWindow.screen = (0, 0, 48, 32)     # the game switched the resolution
        wait_for(r, result="game on another monitor", filming=False)
        wait_for(r, result="filming", filming=True)
    finally:
        r.stop()
        r.join(timeout=5)
    assert len(done) == 2                                     # carried on in a new video at the new size


def test_town_trip_keeps_where_you_came_from(recorder):
    # Clearfell entered from the encampment, a town trip, back in, and then the screen size changes. The new video still
    # came from the encampment, not from Clearfell itself.
    r, done = recorder
    r._change_visit(TOWN)
    r._change_visit(A)
    film(r)
    r._change_visit(TOWN)
    r._change_visit(A)
    r._film(np.zeros((32, 48, 3), np.uint8), 5.0)
    assert len(done) == 1
    assert json.loads(r.video.visit_file.read_text(encoding="utf-8").splitlines()[0])["came_from"] == \
        "Clearfell Encampment"


def test_video_that_could_not_start_is_tried_again(recorder, monkeypatch):
    r, _ = recorder
    real = recorder_module.ZoneVideo
    fails = [True]

    def flaky(*args, **kwargs):
        if fails[0]:
            raise OSError("drive gone")
        return real(*args, **kwargs)

    monkeypatch.setattr(recorder_module, "ZoneVideo", flaky)
    r._change_visit(TOWN)
    with pytest.raises(OSError):
        r._change_visit(A)
    assert r._visit != A                                  # the next round of the loop tries again
    fails[0] = False
    r._change_visit(A)
    assert r.video is not None and r.video.visit == A
    assert json.loads(r.video.visit_file.read_text(encoding="utf-8").splitlines()[0])["came_from"] == \
        "Clearfell Encampment"


def test_clock_set_back_does_not_stop_filming(recorder, monkeypatch):
    # the wall clock jumps back 1000 s between two reads (a time sync), and the next frame still comes on time
    r, _ = recorder
    monkeypatch.setattr(recorder_module, "Grabber", FakeGrabber)
    monkeypatch.setattr(recorder_module, "GameWindowCheck", FakeWindow)
    monkeypatch.setattr(recorder_module, "mss_rect", lambda monitor: (0, 0, 96, 64))
    clock = [1e9]

    def wall_clock():
        clock[0] -= 1000.0
        return clock[0]

    monkeypatch.setattr(recorder_module.time, "time", wall_clock)
    r._disk_full = lambda: False
    r.interval = 0.01
    r.set_zone(*A)
    r.start()
    try:
        end = time.monotonic() + 5
        while r.get_state().get("frames", 0) < 3 and time.monotonic() < end:
            time.sleep(0.01)
        assert r.get_state().get("frames", 0) >= 3
    finally:
        r.stop()
        r.join(timeout=5)


def test_alt_tab_before_a_town_trip_is_one_gap(recorder):
    r, _ = recorder
    r._change_visit(A)
    film(r)
    r._pause(time.monotonic() - 10)                       # alt-tabbed out, then off to town
    r._change_visit(TOWN)
    r._change_visit(A)                                    # back, which writes the gap
    r._film(FRAME, 5.0)
    lines = r.video.visit_file.read_text(encoding="utf-8").splitlines()
    assert sum(json.loads(line).get("gap", False) for line in lines) == 1


def test_video_that_could_not_restart_after_a_resize_is_tried_again(recorder, monkeypatch):
    r, done = recorder
    r._change_visit(TOWN)
    r._change_visit(A)
    film(r)
    real = recorder_module.ZoneVideo
    fails = [True]

    def flaky(*args, **kwargs):
        if fails[0]:
            raise OSError("drive gone")
        return real(*args, **kwargs)

    monkeypatch.setattr(recorder_module, "ZoneVideo", flaky)
    with pytest.raises(OSError):
        r._film(np.zeros((32, 48, 3), np.uint8), 5.0)     # the screen size changed: a new video that can't start
    assert r._visit != A                                  # the next round of the loop tries again
    fails[0] = False
    r._change_visit(A)
    assert r.video is not None and len(done) == 1
    assert json.loads(r.video.visit_file.read_text(encoding="utf-8").splitlines()[0])["came_from"] == \
        "Clearfell Encampment"


@pytest.fixture
def manual(tmp_path):
    cfg, _ = config_from_dict({"data_dir": str(tmp_path)})          # manual is the default
    done = []
    r = ZoneRecorder(cfg, done.append)
    yield r, done
    r._finish()


def came_from(video):
    return json.loads(video.visit_file.read_text(encoding="utf-8").splitlines()[0])["came_from"]


def test_manual_recording_films_only_between_start_and_stop(manual):
    """With recorder.manual, entering a zone doesn't film. The record hotkey starts and stops it."""
    r, done = manual
    r.set_zone(*TOWN)
    r._follow()
    r.set_zone(*A)
    r._follow()
    assert r.video is None                                       # not started yet
    r.set_recording(True)
    r._follow()
    assert r.video.visit == A
    assert came_from(r.video) == "Clearfell Encampment"          # started mid-zone, but it still knows the entrance
    film(r)
    r.set_zone(*A2)
    r._follow()
    assert len(done) == 1 and r.video.visit == A2               # while it's on, a new zone still gets its own video
    film(r)
    r.set_recording(False)
    r._follow()
    assert r.video is None and len(done) == 2                    # stopping hands the video to the analysis
    r.set_zone(*A)
    r._follow()
    assert r.video is None                                       # and it stays off in the next zone


def test_manual_recording_loop_says_how_to_start(manual, monkeypatch):
    r, done = manual
    monkeypatch.setattr(recorder_module, "Grabber", FakeGrabber)
    monkeypatch.setattr(recorder_module, "GameWindowCheck", FakeWindow)
    monkeypatch.setattr(recorder_module, "mss_rect", lambda monitor: (0, 0, 96, 64))
    r._disk_full = lambda: False
    r.interval = 0.01
    r.set_zone(*A)
    r.start()
    try:
        wait_for(r, result="press ctrl+alt+r to film", filming=False)
        r.set_recording(True)
        wait_for(r, result="filming", filming=True)
        r.set_recording(False)
        wait_for(r, result="press ctrl+alt+r to film", filming=False)
        assert len(done) == 1
    finally:
        r.stop()
        r.join(timeout=5)
