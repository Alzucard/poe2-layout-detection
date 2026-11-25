"""The whole analysis path, from a filmed zone video to a saved run (without OCR)."""
import json

import cv2
import numpy as np
import pytest

from conftest import frame_at
from poe2layout.analyzer import Analyzer, VideoReadError, frames_with_outlines, read_visit_file
from poe2layout.config import config_from_dict
from poe2layout.files import read_json
from poe2layout.outline import DEFAULT_MAP_AREA, outline_mask
from poe2layout.recorder import ZoneVideo
from poe2layout.zones import Instance, ZoneData

INSTANCE = Instance("G1_2", 77)


def film(folder, world, start, n, came_from="Clearfell Encampment"):
    v = ZoneVideo(folder, "HFYU", 2.0, "Clearfell", INSTANCE, came_from)
    for i in range(n):
        v.write(frame_at(world, start[0] + 25 * i, start[1] + 12 * i), 1000.0 + i / 2)
    return v.close()


@pytest.fixture
def analyzer(tmp_path):
    cfg, _ = config_from_dict({"data_dir": str(tmp_path)})
    an = Analyzer(cfg, ZoneData(tmp_path / "no_zones.json"))    # not started, since we call analyze() directly
    return an, cfg


def test_video_becomes_a_saved_run(analyzer, world):
    an, cfg = analyzer
    video = film(cfg.video_dir, world, (500, 400), 34)
    an.analyze(video)
    folder = cfg.runs_dir / "Clearfell" / INSTANCE.id
    meta = read_json(folder / "meta.json")
    assert meta["frames"] == 34 and meta["came_from"] == "Clearfell Encampment"
    assert read_json(cfg.runs_dir / "Clearfell" / "layouts.json")[INSTANCE.id]["layout"] == "A"
    assert list(cfg.video_dir.iterdir()) == []                  # the video and visit file are deleted
    assert ("Clearfell", INSTANCE) in an._parked


def test_coming_back_continues_the_map(analyzer, world):
    an, cfg = analyzer
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))
    # back through a portal: the second video carries on where the first one ended
    an.analyze(film(cfg.video_dir, world, (500 + 25 * 30, 400 + 12 * 30), 20, came_from="Clearfell"))
    meta = read_json(cfg.runs_dir / "Clearfell" / INSTANCE.id / "meta.json")
    assert meta["frames"] == 54
    assert meta["came_from"] == "Clearfell Encampment"           # still the way you first came in


def test_outlines_computed_ahead_match_frame_by_frame(tmp_path, world):
    video = film(tmp_path, world, (500, 400), 21)          # more frames than get computed ahead at once
    cap = cv2.VideoCapture(str(video))
    got = list(frames_with_outlines(cap, DEFAULT_MAP_AREA))
    cap.release()
    cap = cv2.VideoCapture(str(video))
    want = [cap.read()[1] for _ in range(21)]
    cap.release()
    assert len(got) == 21
    for (frame, outline), expected in zip(got, want, strict=True):
        assert np.array_equal(frame, expected)
        assert np.array_equal(outline, outline_mask(expected, DEFAULT_MAP_AREA))


class FlakyCapture:
    """read() of a capture whose third read fails, while a later one would work again."""

    def __init__(self, frame):
        self.reads = [(True, frame), (True, frame), (False, None), (True, frame)]

    def read(self):
        return self.reads.pop(0) if self.reads else (False, None)


def test_failed_read_ends_the_frames(world):
    frame = frame_at(world, 500, 400)
    assert len(list(frames_with_outlines(FlakyCapture(frame), DEFAULT_MAP_AREA))) == 2


def test_stopped_analysis_leaves_the_video(analyzer, world):
    an, cfg = analyzer
    video = film(cfg.video_dir, world, (500, 400), 34)
    an.stop()                                    # second Ctrl+C, so the program exits
    an.add(video)
    an.finish()
    an.run()
    assert video.exists() and video.with_suffix(".jsonl").exists() and an.get_state().get("failed") is None


def test_pool_refused_at_exit_does_not_set_the_video_aside(analyzer, world, monkeypatch):
    an, cfg = analyzer
    video = film(cfg.video_dir, world, (500, 400), 34)

    def refused(cap, area):              # what the thread pool does once the interpreter is shutting down
        an.stop()
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")
        yield

    monkeypatch.setattr("poe2layout.analyzer.frames_with_outlines", refused)
    an.add(video)
    an.finish()
    an.run()
    assert video.exists() and not cfg.failed_video_dir.exists() and an.get_state().get("failed") is None


def test_broken_last_line_is_skipped(tmp_path):
    p = tmp_path / "v.jsonl"
    p.write_text(json.dumps({"zone": "Clearfell", "instance": ["G1_2", 1], "came_from": None}) + "\n"
                 + '{"t": 1.0}\n{"t": 1.5}\n{"t": 2', encoding="utf-8")
    assert read_visit_file(p).times == [1.0, 1.5]


@pytest.mark.parametrize("text", ["", '{"zone": "Clea', '{"t": 1.0}\n', '{"zone": "Clearfell", "instance": null}\n'])
def test_broken_header_raises(tmp_path, text):
    p = tmp_path / "v.jsonl"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_visit_file(p)


def test_failed_video_is_set_aside_and_counted(analyzer):
    an, cfg = analyzer
    cfg.video_dir.mkdir(parents=True)
    bad = cfg.video_dir / "bad.avi"
    bad.write_bytes(b"not a video")
    bad.with_suffix(".jsonl").write_text("{broken\n", encoding="utf-8")
    an.add(bad)
    an.finish()
    an.run()                                                     # in this thread; it ends after the queue
    assert (cfg.failed_video_dir / "bad.avi").exists() and (cfg.failed_video_dir / "bad.jsonl").exists()
    state = an.get_state()
    assert state["failed"] == 1 and "bad.avi" in state["error"] and state["queued"] == 0


def visit_lines(n):
    head = {"zone": "Clearfell", "instance": INSTANCE.to_json(), "came_from": None}
    return "".join(json.dumps(r) + "\n" for r in [head, *({"t": 1000.0 + i / 2} for i in range(n))])


def test_unreadable_video_is_set_aside_not_deleted(analyzer):
    an, cfg = analyzer
    cfg.video_dir.mkdir(parents=True)
    bad = cfg.video_dir / "x.avi"
    bad.write_bytes(b"not a video" * 100)
    bad.with_suffix(".jsonl").write_text(visit_lines(40), encoding="utf-8")
    with pytest.raises(VideoReadError):
        an.analyze(bad)
    an.add(bad)
    an.finish()
    an.run()
    assert (cfg.failed_video_dir / "x.avi").exists() and an.get_state()["failed"] == 1


def test_cut_off_video(analyzer, world):
    an, cfg = analyzer
    video = film(cfg.video_dir, world, (500, 400), 34)
    visit = video.with_suffix(".jsonl")
    visit.write_text(visit_lines(100), encoding="utf-8")            # 66 frames lost in a crash
    with pytest.raises(VideoReadError):
        an.analyze(video)
    assert video.exists()
    visit.write_text(visit_lines(40), encoding="utf-8")             # only a few lost, so it still gets analyzed
    an.analyze(video)
    assert read_json(cfg.runs_dir / "Clearfell" / INSTANCE.id / "meta.json")["frames"] == 34


def test_failed_video_keeps_the_parked_map(analyzer, world):
    an, cfg = analyzer
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))
    bad = cfg.video_dir / "y.avi"
    bad.write_bytes(b"not a video" * 100)
    bad.with_suffix(".jsonl").write_text(visit_lines(40), encoding="utf-8")
    with pytest.raises(VideoReadError):
        an.analyze(bad)
    assert ("Clearfell", INSTANCE) in an._parked


def test_other_screen_size_starts_a_new_map(analyzer, world):
    an, cfg = analyzer
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))
    small = world[:, :, :]
    v = ZoneVideo(cfg.video_dir, "HFYU", 2.0, "Clearfell", INSTANCE, "Clearfell")
    for i in range(34):
        x, y = 500 + 25 * i, 400 + 12 * i
        v.write(np.ascontiguousarray(small[y:y + 900, x:x + 1600]), 2000.0 + i / 2)
    an.analyze(v.close())
    meta = read_json(cfg.runs_dir / "Clearfell" / INSTANCE.id / "meta.json")
    assert meta["screen"] == [1920, 1080]           # the first, bigger map is kept and not mixed with the small one


def test_label_reading_error_keeps_the_run(analyzer, world):
    an, cfg = analyzer

    class BrokenReader:
        def read(self, *args):
            raise OSError("image too large")

    an.reader = BrokenReader()
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))
    assert (cfg.runs_dir / "Clearfell" / INSTANCE.id / "meta.json").exists()


def test_failed_resumed_video_is_not_fed_twice(analyzer, world):
    an, cfg = analyzer
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))
    video = film(cfg.video_dir, world, (500 + 25 * 30, 400 + 12 * 30), 20, came_from="Clearfell")
    video.with_suffix(".jsonl").write_text(visit_lines(100), encoding="utf-8")      # cut off: 20 of 100 frames
    with pytest.raises(VideoReadError):
        an.analyze(video)
    run = an._parked[("Clearfell", INSTANCE)]                  # still parked, with the frames that were read
    frames = run.stitcher.frames
    # moved back from failed/ and tried again: it's analyzed on its own, and the parked map only counts its frames once
    video.with_suffix(".jsonl").write_text(visit_lines(20), encoding="utf-8")
    an.analyze(video)
    assert an._parked[("Clearfell", INSTANCE)] is run and run.stitcher.frames == frames


def test_save_error_keeps_the_parked_map(analyzer, world, monkeypatch):
    an, cfg = analyzer
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))

    def broken_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(an.library, "save", broken_save)
    video = film(cfg.video_dir, world, (500 + 25 * 30, 400 + 12 * 30), 20, came_from="Clearfell")
    with pytest.raises(OSError):
        an.analyze(video)
    assert an._parked[("Clearfell", INSTANCE)].stitcher.frames >= 50


class GateReader:
    """Stands in for the OCR. It reads the gate label "The Riverbank", which lies where the camera of frame 0 was
    centred (the map's first point). The video is lossless, so each frame tells us its camera position."""

    def __init__(self, world, start, n, fail=False):
        self.camera = {frame_at(world, start[0] + 25 * i, start[1] + 12 * i).tobytes(): i for i in range(n)}
        self.gate = (start[0] + 960, start[1] + 540)                  # world px
        self.start, self.fail, self.calls = start, fail, []

    def read(self, frame, zone, came_from):
        i = self.camera[frame.tobytes()]
        self.calls.append((i, zone, came_from))
        if self.fail:
            raise OSError("ocr broke")
        x = self.gate[0] - (self.start[0] + 25 * i)                   # the gate is on this frame's screen
        y = self.gate[1] - (self.start[1] + 12 * i)
        return [("The Riverbank", x - 60, y - 10, 120, 20)]


def test_labels_are_read_on_the_map_and_teach_the_gate(analyzer, world):
    an, cfg = analyzer
    an.reader = GateReader(world, (500, 400), 34)
    an.analyze(film(cfg.video_dir, world, (500, 400), 34, came_from="The Riverbank"))
    calls = an.reader.calls
    assert [i for i, _, _ in calls] == list(range(0, 34, 2))          # once per second of video (2 fps)
    assert calls[0][1:] == ("Clearfell", "The Riverbank")
    meta = read_json(cfg.runs_dir / "Clearfell" / INSTANCE.id / "meta.json")
    (label,) = meta["labels"]
    assert label["text"] == "The Riverbank" and label["hits"] == 17 and label["looks"] >= 17
    assert np.hypot(label["x"] - meta["first_point"][0], label["y"] - meta["first_point"][1]) <= 2
    assert "riverbank" in an.zones.exits("Clearfell")              # walked in through that gate


def test_failed_label_reading_still_saves_the_map(analyzer, world):
    an, cfg = analyzer
    an.reader = GateReader(world, (500, 400), 34, fail=True)
    an.analyze(film(cfg.video_dir, world, (500, 400), 34, came_from="The Riverbank"))
    assert len(an.reader.calls) == 1                                   # not tried again for this video
    meta = read_json(cfg.runs_dir / "Clearfell" / INSTANCE.id / "meta.json")
    assert meta["frames"] == 34 and meta["labels"] == []


def test_stitching_error_keeps_the_parked_map(analyzer, world, monkeypatch):
    an, cfg = analyzer
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))

    def broken_add(*args):
        raise RuntimeError("bad frame")

    monkeypatch.setattr(an.collector, "add", broken_add)
    video = film(cfg.video_dir, world, (500 + 25 * 30, 400 + 12 * 30), 20, came_from="Clearfell")
    with pytest.raises(RuntimeError):
        an.analyze(video)
    assert an._parked[("Clearfell", INSTANCE)].stitcher.frames >= 34      # the first video's map isn't lost


def test_black_screen_then_error_keeps_the_parked_map(analyzer, world, monkeypatch):
    # the resumed video goes black (its map is put aside as a piece), then fails before a new map starts
    an, cfg = analyzer
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))
    calls = []

    def add_then_fail(*args):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("bad frame")

    monkeypatch.setattr(an.collector, "add", add_then_fail)
    v = ZoneVideo(cfg.video_dir, "HFYU", 2.0, "Clearfell", INSTANCE, "Clearfell")
    for t in (1000.0, 1000.5):
        v.write(np.zeros((1080, 1920, 3), np.uint8), t)
    with pytest.raises(RuntimeError):
        an.analyze(v.close())
    assert an._parked[("Clearfell", INSTANCE)].stitcher.frames >= 34


def test_visit_file_of_the_old_recorder_is_read(tmp_path):
    p = tmp_path / "v.jsonl"
    head = {"area": "Clearfell", "entry": ["G1_2", 77], "came_from": "Clearfell Encampment", "started": 1.0}
    p.write_text(json.dumps(head) + '\n{"t": 1.0}\n{"gap": true}\n{"t": 9.0}\n', encoding="utf-8")
    rec = read_visit_file(p)
    assert (rec.zone, rec.instance, rec.came_from) == ("Clearfell", INSTANCE, "Clearfell Encampment")
    assert rec.times == [1.0, 9.0] and rec.gaps == {1}


def test_map_locked_by_another_program_keeps_the_video(analyzer, world, monkeypatch):
    an, cfg = analyzer

    def locked(*args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(an.library, "save", locked)
    video = film(cfg.video_dir, world, (500, 400), 34)
    an.add(video)
    an.finish()
    an.run()
    assert video.exists() and not (cfg.failed_video_dir / video.name).exists()    # analyzed again on the next start
    assert an.get_state()["failed"] == 1


class NoLabels:
    def read(self, frame, zone, came_from):
        return []


def test_gate_is_not_learned_when_its_label_was_not_seen(analyzer, world):
    # came by waypoint or portal, so the zone you were in before tells us nothing about this zone's gates
    an, cfg = analyzer
    an.reader = NoLabels()
    an.analyze(film(cfg.video_dir, world, (500, 400), 34, came_from="The Riverbank"))
    assert (cfg.runs_dir / "Clearfell" / INSTANCE.id / "meta.json").exists()
    assert "riverbank" not in an.zones.exits("Clearfell")


def test_error_while_parking_does_not_hide_the_analysis_error(analyzer, world, monkeypatch):
    # the resumed video goes black and then fails, and the last merge while parking fails too; the first error is the
    # one that counts
    an, cfg = analyzer
    an.analyze(film(cfg.video_dir, world, (500, 400), 34))
    st = an._parked[("Clearfell", INSTANCE)].stitcher
    calls = []

    def add_then_fail(*args):
        calls.append(1)
        if len(calls) > 1:
            raise PermissionError(13, "Access is denied")

    def broken_finalize():
        raise RuntimeError("merge failed")

    monkeypatch.setattr(an.collector, "add", add_then_fail)
    monkeypatch.setattr(st, "finalize", broken_finalize)
    v = ZoneVideo(cfg.video_dir, "HFYU", 2.0, "Clearfell", INSTANCE, "Clearfell")
    for t in (1000.0, 1000.5):
        v.write(np.zeros((1080, 1920, 3), np.uint8), t)
    with pytest.raises(PermissionError):
        an.analyze(v.close())
