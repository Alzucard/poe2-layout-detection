"""Zone analysis. Takes the video of a finished visit, turns it into a saved run and then deletes the video.

For every frame we stitch the Tab map outline into the zone's map (Stitcher) and keep track of the route you walked.
About once a second of video time we also read the map labels. Then the run is saved (RunLibrary) and sorted into a
layout. Videos are analyzed one after another, in the order they were filmed, in the background while you play.
Anything left over from an earlier start (closed early, or a crash) goes first. If a video's analysis fails, or we
can read far fewer frames than its visit file lists, we move it to <video_dir>/failed/ instead of retrying it on
every start.
"""
from __future__ import annotations

import json
import queue
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .collector import TrainingFrameCollector
from .config import VIDEO_CODECS, Config
from .labels import LABEL_EXCLUDE, LABEL_ROWS, LabelReader, came_through_gate, trusted_labels
from .library import RunLibrary
from .logs import get_logger
from .ocr import WinOcr
from .outline import DEFAULT_MAP_AREA, MapArea, as_map_area, outline_mask
from .stitcher import Stitcher
from .threads import StatefulThread
from .winapi import lower_thread_priority
from .zones import Instance, ZoneData, is_town

log = get_logger("analyzer")

MIN_RUN_FRAMES = 30            # we don't save a map made from fewer placed frames
LABEL_EDGE_MARGIN = 20         # canvas px; labels cut off at the screen edge can't be read, so their spots don't count
MIN_READ_SHARE = 0.5           # if we can read less of the visit file's frames than this, the video is set aside
OUTLINE_THREADS = 4            # frames whose outline we compute at once (the slowest step, about 3x faster with 4)


class VideoReadError(RuntimeError):
    """The video has far fewer readable frames than its visit file lists."""


class AnalysisStopped(Exception):
    """The program is shutting down (stop()), so the video stays in the folder for the next start."""


@dataclass
class VisitRecord:
    """What's in a visit file (the .jsonl next to the video)."""
    zone: str
    instance: Instance
    came_from: str | None
    times: list[float] = field(default_factory=list)     # the time of each frame
    gaps: set[int] = field(default_factory=set)          # frames that come right after a gap (you were away)


def read_visit_file(path: Path) -> VisitRecord:
    """Raises ValueError if we can't read the first line (zone, instance). Other broken lines, say from a crash while
    a line was being written, are skipped, since the frames are still in the video."""
    lines = []
    for i, s in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not s.strip():
            continue
        try:
            lines.append(json.loads(s))
        except ValueError:
            if i == 0:
                raise
            log.warning("%s: line %d unreadable - skipped", path.name, i + 1)
    head = lines[0] if lines else {}
    if "area" in head and "zone" not in head:            # the first version wrote these: area, entry
        head = {"zone": head["area"], "instance": head.get("entry"), "came_from": head.get("came_from")}
    if "zone" not in head or not head.get("instance"):
        raise ValueError(f"{path.name} has no visit header (zone and instance)")
    record = VisitRecord(head["zone"], Instance.from_json(head["instance"]), head.get("came_from"))
    for line in lines[1:]:
        if line.get("gap"):
            record.gaps.add(len(record.times))
        elif "t" in line:
            record.times.append(line["t"])
    return record


def frames_with_outlines(cap: cv2.VideoCapture, area: MapArea):
    """Yields (frame, outline_mask) for every frame of the video, in order. The outlines are computed a few frames
    ahead on worker threads, because OpenCV runs outline_mask on one core and releases the GIL."""
    with ThreadPoolExecutor(OUTLINE_THREADS, initializer=lower_thread_priority) as pool:
        ahead = deque()
        more = True
        while True:
            while more and len(ahead) <= OUTLINE_THREADS:
                more, frame = cap.read()             # the first read that fails ends the video
                if more:
                    ahead.append((frame, pool.submit(outline_mask, frame, area)))
            if not ahead:
                return
            frame, outline = ahead.popleft()
            yield frame, outline.result()


def leftover_videos(folder: Path) -> list[Path]:
    """Videos still waiting in the folder, oldest first. A visit file without its video gets removed."""
    out = []
    for visit_file in sorted(folder.glob("*.jsonl")):
        videos = [visit_file.with_suffix(ext) for ext in VIDEO_CODECS.values() if visit_file.with_suffix(ext).exists()]
        if videos:
            out.append(videos[0])
        else:
            visit_file.unlink(missing_ok=True)
    return out


@dataclass
class ZoneRun:
    """An instance being analyzed. We keep it after its video is done, in case you come back to it."""
    zone: str
    instance: Instance
    came_from: str | None
    stitcher: Stitcher
    videos: set[str] = field(default_factory=set)        # names of the videos that went into this map


class Analyzer(StatefulThread):
    """Background thread. add() hands it a finished video, and get_state() tells you how far along it is. The state
    has status ("idle" or "analyzing"), zone, done, total, queued, failed (videos moved to failed/) and error (the
    last failure)."""

    def __init__(self, cfg: Config, zones: ZoneData):
        super().__init__("analyzer", {"status": "idle"})
        self.cfg = cfg
        self.zones = zones
        self.map_area = as_map_area(cfg.analysis.map_area) if cfg.analysis.map_area else DEFAULT_MAP_AREA
        self.collector = TrainingFrameCollector(cfg.training_frames, cfg.dataset_dir)
        self.library = RunLibrary(cfg.runs_dir, diagnostics_dir=cfg.snapshots_dir)
        self.reader: LabelReader | None = None
        self._parked: dict[tuple[str, Instance], ZoneRun] = {}   # instances you might come back to
        self._queue: queue.Queue[Path | None] = queue.Queue()
        self._waiting = 0                                         # videos handed in that haven't started yet
        for video in leftover_videos(cfg.video_dir):
            log.info("video left from an earlier start: %s", video.name)
            self.add(video)

    # ---- thread-safe API
    def add(self, video: Path) -> None:
        with self._state_lock:
            self._waiting += 1
        self._queue.put(video)

    def finish(self) -> None:
        """Stop once every video handed in so far has been analyzed."""
        self._queue.put(None)

    def get_state(self) -> dict:
        with self._state_lock:
            return dict(self._state, queued=self._waiting)

    # ---- loop
    def run(self) -> None:
        lower_thread_priority()                  # the game gets the CPU first
        try:
            self.reader = LabelReader(WinOcr(), self.zones)    # WinOcr belongs to the thread that uses it
        except (ImportError, RuntimeError, OSError) as e:
            log.warning("label OCR not available (%s) - runs are saved without labels", e)
        while True:
            video = self._queue.get()
            if video is None:
                break
            with self._state_lock:
                self._waiting -= 1
            try:
                self.analyze(video)
            except Exception as e:               # one bad video shouldn't stop the others from being analyzed
                if self.stopping:                # shutting down; the thread pool refuses work then too (RuntimeError)
                    log.info("analysis of %s stopped - it stays in the video folder", video.name)
                    break
                log.exception("analysis of %s failed", video.name)
                if isinstance(e, PermissionError):   # another program (an image viewer, say) has the file open
                    log.warning("%s stays in the video folder - it is analyzed again at the next start", video.name)
                else:
                    self._set_aside(video)
                with self._state_lock:
                    self._state["failed"] = self._state.get("failed", 0) + 1
                    self._state["error"] = f"{video.name}: {e}"
            self._publish(status="idle", zone=None, done=0, total=0)
        if self.reader is not None:
            self.reader.ocr.close()

    def _set_aside(self, video: Path) -> None:
        """Move a video whose analysis failed, along with its visit file, to failed/ so we don't retry it on every
        start."""
        target = self.cfg.failed_video_dir
        try:
            target.mkdir(parents=True, exist_ok=True)
            for f in (video, video.with_suffix(".jsonl")):
                if f.exists():
                    shutil.move(str(f), str(target / f.name))
            log.warning("%s moved to %s - move it back to the video folder to try again", video.name, target)
        except OSError as e:
            log.warning("could not move %s to %s (%s)", video.name, target, e)

    # ---- one video
    def analyze(self, video: Path) -> None:
        visit_file = video.with_suffix(".jsonl")
        record = read_visit_file(visit_file)
        zone, instance = record.zone, record.instance
        if is_town(zone, instance):
            log.info("%s is a town/hideout - deleted without analysis", video.name)
            video.unlink(missing_ok=True)
            visit_file.unlink(missing_ok=True)
            return
        cap = cv2.VideoCapture(str(video))
        try:
            run, resumed, alone = self._run_for(video.name, cap, record)
        except BaseException:
            cap.release()
            raise
        frames = 0
        try:
            try:
                self.collector.start(zone, instance.id)
                self._publish(status="analyzing", zone=zone, done=0, total=len(record.times))
                run.videos.add(video.name)
                frames = self._stitch_video(video.name, cap, record, run)
            finally:
                cap.release()
            listed = len(record.times)
            if frames != listed:
                log.warning("%s: %d frames read, the visit file lists %d", video.name, frames, listed)
            if frames < listed * MIN_READ_SHARE:
                # we can't open it, or it's cut off (a crash while filming), so keep it around to look at instead of
                # deleting it
                raise VideoReadError(f"only {frames} of the {listed} frames in the visit file could be read")
            if frames == 0:
                log.info("%s has no frames - deleted", video.name)
            else:
                self._save_run(run)
        finally:
            # Park the run whether or not this video or its save failed, so its map (plus the earlier videos' map if
            # this was a resumed instance) isn't lost. A map that was made on its own never replaces the parked one.
            if not alone and (resumed or frames > 0):
                self._park(run)
        video.unlink(missing_ok=True)
        visit_file.unlink(missing_ok=True)

    def _run_for(self, name: str, cap: cv2.VideoCapture, record: VisitRecord) -> tuple[ZoneRun, bool, bool]:
        """Returns (run, resumed, alone): the parked map of this instance if there is one to continue, otherwise a
        new one. alone means a new map that must not replace the parked one. That happens when the video is already
        in it (its first try failed and it was moved back from failed/, so feeding it again would count its frames
        twice), or when it was filmed at a different screen size (the stitcher's thresholds assume one size)."""
        zone, instance = record.zone, record.instance
        run = self._parked.get((zone, instance))
        alone = False
        if run is not None and name in run.videos:
            log.info("%s is already in the map of %r (an earlier try failed) - analyzed on its own", name, zone)
            run, alone = None, True
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if run is not None and run.stitcher.screen and cap.isOpened() and size != run.stitcher.screen:
            log.info("%s: the screen size changed since the last visit of %r - starting a new map", name, zone)
            run, alone = None, True
        if run is not None:
            del self._parked[(zone, instance)]
            run.stitcher.mark_gap()              # we were away, so find the position again before trusting motion
            log.info("analyzing %s: back in the same instance of %r, continuing its map (%d frames)",
                     name, zone, run.stitcher.frames)
            return run, True, False
        snapshots = self.cfg.snapshots_dir if self.cfg.debug_snapshots else None
        run = ZoneRun(zone, instance, record.came_from, Stitcher(map_area=self.map_area, snapshot_dir=snapshots))
        log.info("analyzing %s: %r, %d frames (%d saved runs so far)", name, zone, len(record.times),
                 self.library.count(zone))
        return run, False, alone

    def _stitch_video(self, name: str, cap: cv2.VideoCapture, record: VisitRecord, run: ZoneRun) -> int:
        """Feed every frame of the video to the run's stitcher. Returns how many frames were read."""
        st = run.stitcher
        n, last_ocr, ocr_failed = 0, -1e9, False
        # closed right away if this raises, so we don't compute outlines ahead for nothing
        with closing(frames_with_outlines(cap, st.map_area)) as frames:
            for frame, outline in frames:
                if self.stopping:
                    raise AnalysisStopped
                t = record.times[n] if n < len(record.times) else (record.times[-1] if record.times else 0.0)
                if n in record.gaps:
                    st.mark_gap()
                    log.info("%s: back from town at frame %d", name, n)
                n += 1
                result = st.add(frame, outline)
                self.collector.add(frame, st, result, t)
                if (self.reader and result.placed and not ocr_failed
                        and t - last_ocr >= self.cfg.analysis.ocr_every):
                    last_ocr = t
                    try:
                        self._read_labels(run, frame)
                    except Exception:            # labels are a bonus; the map of this video still gets saved
                        log.exception("%s: reading the map labels failed - no more labels read from it", name)
                        ocr_failed = True
                if n % 10 == 0:
                    self._publish(done=n)
        return n

    def _read_labels(self, run: ZoneRun, frame: np.ndarray) -> None:
        """Read the frame's map labels with OCR and put them on the canvas where this frame sits."""
        st = run.stitcher
        ox, oy = st.offset
        s = st.scale
        for name, x, y, w, h in self.reader.read(frame, run.zone, run.came_from):
            st.labels.add(name, ox + (x + w / 2) * s, oy + (y + h / 2) * s, screen=(x + w / 2, y + h / 2))
        # the part of the screen we read, on the canvas: labels there had a chance to be seen (for the vote)
        fh, fw = frame.shape[:2]
        e = LABEL_EDGE_MARGIN
        skip = [(ox + ex0 * fw * s, oy + ey0 * fh * s, ox + ex1 * fw * s, oy + ey1 * fh * s)
                for ex0, ey0, ex1, ey1 in LABEL_EXCLUDE]
        st.labels.look(ox + e, oy + LABEL_ROWS[0] * fh * s + e, ox + fw * s - e, oy + LABEL_ROWS[1] * fh * s - e,
                       skip)

    def _park(self, run: ZoneRun) -> None:
        """Keep the map of an instance you might come back to after another zone. (A town trip on its own keeps the
        video going, see recorder.py.) Once there are more than keep_instances, the oldest parked map is dropped."""
        if run.stitcher.offset is None:
            try:
                run.stitcher.finalize()          # the screen went black last, so continue from its largest piece
            except Exception:                    # parking runs in analyze's finally, so never hide its own error
                log.exception("%r: the last merge before parking failed", run.zone)
        if run.stitcher.offset is None:
            return
        self._parked[(run.zone, run.instance)] = run
        while len(self._parked) > self.cfg.analysis.keep_instances:
            self._parked.pop(next(iter(self._parked)))

    def _save_run(self, run: ZoneRun) -> None:
        st = run.stitcher
        st.finalize()
        if st.frames < MIN_RUN_FRAMES:
            log.info("run in %r not saved: largest map piece has only %d frames", run.zone, st.frames)
            return
        saved = self.library.save(run.zone, st, run.instance, run.came_from, self.zones.exits(run.zone),
                                  min_floor_frames=MIN_RUN_FRAMES)
        if saved is None:
            return
        meta = saved.meta
        floors = 1 + len(meta["other_floors"])
        log.info("run saved: %s (%d frames, %d floors, %d labels read, %d confirmed, %d pieces left out)",
                 saved.folder, meta["frames"], floors, len(meta["labels"]), len(trusted_labels(meta["labels"])),
                 len(st.pieces) + 1 - floors)
        # only learned from a gate seen at the entrance; a waypoint or portal trip tells us nothing about gates
        if came_through_gate(meta["labels"], meta["first_point"], run.came_from):
            self.zones.learn_exit(run.zone, run.came_from)
        try:
            self.library.layouts(run.zone)
        except Exception:                        # grouping is redone next time, and the run itself is saved
            log.exception("grouping the saved runs into layouts failed")
