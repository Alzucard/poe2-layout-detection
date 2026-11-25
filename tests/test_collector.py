"""Training frames kept during analysis: how often, where, the size limit and the frames.jsonl records."""
import json

import numpy as np
import pytest

from poe2layout.collector import TrainingFrameCollector
from poe2layout.config import TrainingFramesConfig
from poe2layout.stitcher import StepResult, Stitcher

FRAME = np.random.default_rng(0).integers(0, 255, (90, 160, 3), np.uint8)


def collector(tmp_path, **cfg) -> TrainingFrameCollector:
    c = TrainingFrameCollector(TrainingFramesConfig(**{"enabled": True, "every": 4.0, **cfg}), tmp_path)
    c.start("The Riverbank", "run1")
    return c


def records(folder) -> list[dict]:
    return [json.loads(line) for line in (folder / "frames.jsonl").read_text("utf-8").splitlines()]


def test_keeps_one_frame_every_few_seconds(tmp_path):
    c = collector(tmp_path)
    for t in (4.0, 5.0, 7.9, 8.0, 11.0, 12.5):
        c.add(FRAME, Stitcher(), StepResult.TRACKED, t)
    folder = tmp_path / "The Riverbank" / "run1"
    assert sorted(p.name for p in folder.glob("*.jpg")) == ["0000.jpg", "0001.jpg", "0002.jpg"]
    assert [r["t"] for r in records(folder)] == [4.0, 8.0, 12.5]


def test_record_describes_the_frame_and_the_stitcher(tmp_path):
    c = collector(tmp_path)
    c.add(FRAME, Stitcher(), StepResult.LOST, 4.123)
    (r,) = records(tmp_path / "The Riverbank" / "run1")
    assert r == {"file": "0000.jpg", "t": 4.12, "result": "lost", "method": "", "offset": None,
                 "shifted": [0, 0], "pieces": 0, "frames": 0, "scale": 0.5, "screen": [160, 90]}


def test_skipped_frames_are_not_kept(tmp_path):
    c = collector(tmp_path)
    c.add(FRAME, Stitcher(), StepResult.SKIPPED, 10.0)
    assert not (tmp_path / "The Riverbank").exists()
    c.add(FRAME, Stitcher(), StepResult.TRACKED, 10.0)    # a skipped frame didn't use up the slot
    assert (tmp_path / "The Riverbank" / "run1" / "0000.jpg").is_file()


def test_disabled_keeps_nothing(tmp_path):
    c = collector(tmp_path, enabled=False)
    c.add(FRAME, Stitcher(), StepResult.TRACKED, 100.0)
    assert list(tmp_path.iterdir()) == []


def test_nothing_kept_before_start(tmp_path):
    c = TrainingFrameCollector(TrainingFramesConfig(enabled=True), tmp_path)
    c.add(FRAME, Stitcher(), StepResult.TRACKED, 100.0)
    assert list(tmp_path.iterdir()) == []


def test_same_run_again_appends_numbering(tmp_path):
    c = collector(tmp_path)
    c.add(FRAME, Stitcher(), StepResult.TRACKED, 4.0)
    c.start("The Riverbank", "run1")                       # a resumed visit, so video time starts over
    c.add(FRAME, Stitcher(), StepResult.TRACKED, 4.0)
    folder = tmp_path / "The Riverbank" / "run1"
    assert sorted(p.name for p in folder.glob("*.jpg")) == ["0000.jpg", "0001.jpg"]
    assert [r["file"] for r in records(folder)] == ["0000.jpg", "0001.jpg"]


def test_zone_names_are_made_safe_for_folders(tmp_path):
    c = TrainingFrameCollector(TrainingFramesConfig(enabled=True), tmp_path)
    c.start('Bad:Zone?"', "run/1")
    c.add(FRAME, Stitcher(), StepResult.TRACKED, 4.0)
    (jpg,) = tmp_path.rglob("*.jpg")
    assert jpg.parent.parent.parent == tmp_path              # exactly <root>/<zone>/<run>/, no extra levels


def test_stops_at_max_mb_and_warns_once(tmp_path, caplog):
    (tmp_path / "old.bin").write_bytes(b"x" * 2_000_000)   # space that's already used counts against the limit
    c = collector(tmp_path, max_mb=2.0)
    with caplog.at_level("WARNING", logger="poe2.collector"):
        for t in (4.0, 8.0, 12.0):
            c.add(FRAME, Stitcher(), StepResult.TRACKED, t)
    assert not list(tmp_path.rglob("*.jpg"))
    assert sum("max_mb" in r.getMessage() for r in caplog.records) == 1


def test_limit_counts_the_frames_it_keeps(tmp_path):
    c = collector(tmp_path, max_mb=1e-9)                     # the first frame alone goes over
    for t in (4.0, 8.0):
        c.add(FRAME, Stitcher(), StepResult.TRACKED, t)
    assert len(list(tmp_path.rglob("*.jpg"))) == 1


@pytest.mark.parametrize("quality", [10, 95])
def test_jpeg_quality_is_used(tmp_path, quality):
    c = collector(tmp_path, jpeg_quality=quality)
    c.add(FRAME, Stitcher(), StepResult.TRACKED, 4.0)
    size = (tmp_path / "The Riverbank" / "run1" / "0000.jpg").stat().st_size
    assert (size < 8000) if quality == 10 else (size > 15000)


def test_numbering_continues_after_a_deleted_frame(tmp_path):
    c = collector(tmp_path, every=1.0)
    for t in (1.0, 2.0, 3.0):
        c.add(FRAME, Stitcher(), StepResult.TRACKED, t)
    folder = tmp_path / "The Riverbank" / "run1"
    (folder / "0001.jpg").unlink()                         # you threw a frame away
    c.start("The Riverbank", "run1")
    c.add(FRAME, Stitcher(), StepResult.TRACKED, 1.0)
    assert sorted(p.name for p in folder.glob("*.jpg")) == ["0000.jpg", "0002.jpg", "0003.jpg"]
    assert [r["file"] for r in records(folder)] == ["0000.jpg", "0001.jpg", "0002.jpg", "0003.jpg"]
