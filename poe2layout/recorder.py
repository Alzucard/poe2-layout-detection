"""Zone recorder. Films the screen while you play, one lossless video per zone visit.

Every 1/fps seconds, while the game is in front, the screen gets written to <video_dir>/<stamp>_<zone>.avi (.mkv with
FFV1). Next to it sits <same name>.jsonl, the visit file, with what the analysis needs: a first line with the zone,
the instance and the zone you came from, and then one line per frame with its time.

When you enter another zone, the video is closed and handed to the analysis, which stitches the map from it and then
deletes it. Towns and hideouts aren't filmed. A trip to town (or to the character select) only pauses the video. When
you're back in the same instance it carries on with the same video and writes a "gap" line, so the analysis knows to
find the position again.

With recorder.manual on, nothing is filmed until you press the record hotkey, and pressing it again finishes the
video. While it's on, everything above works the same (a new zone still gets its own video).

The pause hotkey stops filming without finishing the video, and pressing it again carries on with the same video
(with a "gap" line, like a town trip). Starting a recording ends a pause.

It's lossless on purpose, because normal video compression wipes out a lot of the thin blue Tab map outline.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .capture import Grabber
from .config import VIDEO_CODECS, Config
from .files import safe_name
from .logs import get_logger
from .threads import StatefulThread
from .winapi import GameWindowCheck, mss_rect
from .zones import Instance, is_town

log = get_logger("recorder")

Visit = tuple[str | None, Instance | None]       # (zone, instance); (None, None) when you're in no zone
DISK_CHECK_EVERY = 10.0                          # seconds
MONITOR_CHECK_EVERY = 5.0                        # seconds between checks for the game moving to another monitor
GAP_AFTER = 2.0          # seconds not filmed: longer is a gap, shorter (alt-tab, overlay) only misses frames


class VideoWriteError(RuntimeError):
    pass


class ZoneVideo:
    """One visit being filmed: the video file and its visit file (.jsonl)."""

    def __init__(self, folder: Path, codec: str, fps: float, zone: str, instance: Instance,
                 came_from: str | None):
        folder.mkdir(parents=True, exist_ok=True)
        self.visit: Visit = (zone, instance)
        stem = base = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(zone)}"
        n = 1
        while (folder / (stem + ".jsonl")).exists():      # two visits started in the same second
            n += 1
            stem = f"{base}_{n}"
        self.path = folder / (stem + VIDEO_CODECS[codec])
        self.visit_file = folder / (stem + ".jsonl")
        self.codec, self.fps = codec, fps
        self.frames = 0
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._lines = open(self.visit_file, "w", encoding="utf-8")   # noqa: SIM115 - open for the whole visit, closed by close()
        self._write_line({"zone": zone, "instance": instance.to_json(),
                          "came_from": came_from, "started": time.time()})

    def _write_line(self, record: dict) -> None:
        self._lines.write(json.dumps(record) + "\n")
        self._lines.flush()

    def write(self, frame: np.ndarray, t: float) -> bool:
        """False when the frame's size doesn't match the video (the screen resolution changed)."""
        h, w = frame.shape[:2]
        if self._writer is None:
            self._size = (w, h)
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*self.codec), self.fps, self._size)
            if not self._writer.isOpened():
                raise VideoWriteError(f"cannot write {self.codec} video to {self.path}")
        elif (w, h) != self._size:
            return False
        self._writer.write(frame)
        self._write_line({"t": round(t, 3)})
        self.frames += 1
        return True

    def gap(self) -> None:
        """Some frames are missing before the next one (you were away, e.g. in town)."""
        self._write_line({"gap": True})

    def gb(self) -> float:
        try:
            return self.path.stat().st_size / 1e9
        except OSError:
            return 0.0

    def close(self) -> Path | None:
        """Finish the files. Returns the video path, or None when nothing was filmed (the files are removed then)."""
        if self._writer is not None:
            self._writer.release()
        self._write_line({"end": time.time()})
        self._lines.close()
        if self.frames == 0:
            self.visit_file.unlink(missing_ok=True)
            self.path.unlink(missing_ok=True)
            return None
        return self.path


class ZoneRecorder(StatefulThread):
    """Background thread. The app tells it which zone you're in (set_zone), and every finished video goes to on_done.
    The state has status, filming, result (why it isn't filming), frames, gb and error."""

    def __init__(self, cfg: Config, on_done: Callable[[Path], None]):
        super().__init__("recorder", {"status": "idle", "filming": False})
        self.cfg = cfg
        self.on_done = on_done
        self.interval = 1.0 / cfg.recorder.fps
        self.codec = cfg.recorder.video_codec
        self.folder = cfg.video_dir
        self.min_free_gb = cfg.recorder.min_free_gb
        self.video: ZoneVideo | None = None
        self._requested: Visit = (None, None)     # set_zone writes it from the other thread, under the state lock
        self._wanted = not cfg.recorder.manual    # set_recording writes it the same way
        self._on = self._wanted                   # whether we film zones right now (follows _wanted, see _follow)
        self._off_text = f"press {cfg.hotkeys.record} to film"
        self._hold_wanted = False                 # toggle_pause writes it the same way
        self._held = False                        # paused by the hotkey right now (follows _hold_wanted)
        self._held_text = f"press {cfg.hotkeys.pause} to go on"
        self._visit: Visit = (None, None)
        self._broken_visit: Visit | None = None   # its video couldn't be written, so we don't retry on every frame
        self._last_zone: str | None = None
        self._came_from: str | None = None
        self._paused_since: float | None = None   # frames of this visit not filmed since then (see GAP_AFTER)
        self._window: GameWindowCheck | None = None     # made in run(), for the monitor the grabber films

    # ---- thread-safe API
    def set_zone(self, zone: str | None, instance: Instance | None) -> None:
        with self._state_lock:
            self._requested = (zone, instance)

    def set_recording(self, on: bool) -> None:
        """Start or stop filming (recorder.manual). Starting also ends a pause."""
        with self._state_lock:
            self._wanted = on
            if on:
                self._hold_wanted = False

    def toggle_recording(self) -> None:
        """Start filming, or stop it (the record hotkey)."""
        with self._state_lock:
            on = not self._wanted
        self.set_recording(on)

    def toggle_pause(self) -> None:
        """Pause filming, or go on after a pause. The video stays open meanwhile."""
        with self._state_lock:
            self._hold_wanted = not self._hold_wanted

    # ---- visits
    def _finish(self) -> None:
        if self.video is None:
            return
        video, self.video = self.video, None
        path = video.close()
        if path:
            log.info("video of %r finished: %d frames, %.1f GB - handed to the analysis", video.visit[0],
                     video.frames, video.gb())
            self.on_done(path)

    def _change_visit(self, visit: Visit) -> None:
        zone, instance = visit
        if self.video is not None:
            if zone is None or is_town(zone, instance):
                log.info("left %r for %s - its video is paused until you come back", self.video.visit[0],
                         repr(zone) if zone else "the character select")
            elif self.video.visit == visit:
                log.info("back in the same instance of %r - continuing its video", zone)
                self.video.gap()
                self._paused_since = None              # an alt-tab before the trip falls into this gap
            else:
                self._finish()
        self._visit = visit
        if zone:
            if is_town(zone, instance):
                log.info("%r is a town/hideout - not filmed", zone)
            elif self.video is None:
                self._came_from = self._last_zone      # its gate label marks the entrance
                if self._on:
                    self._start()
            self._last_zone = zone

    def _follow(self) -> None:
        """Catch up with what set_zone, set_recording and toggle_pause asked for."""
        with self._state_lock:
            requested, wanted, hold = self._requested, self._wanted, self._hold_wanted
        if hold != self._held:
            self._held = hold
            log.info("filming paused" if hold else "filming goes on")
        if wanted != self._on:
            self._on = wanted
            if not wanted and self.video is not None:
                log.info("recording stopped")
                self._finish()
            elif wanted:
                log.info("recording started")
        if requested != self._visit:
            self._change_visit(requested)
        elif (self._on and self.video is None and self._visit[0] and not is_town(*self._visit)
              and self._visit != self._broken_visit):
            self._start()                              # started inside a zone: _came_from is from when you came in

    def _start(self) -> None:
        zone, instance = self._visit
        try:
            self.video = ZoneVideo(self.folder, self.codec, 1.0 / self.interval, zone, instance, self._came_from)
        except Exception:
            # we try again next round (the error is shown in the meantime), from the same entrance
            self._visit, self._last_zone = (None, None), self._came_from
            raise
        log.info("filming %r into %s", zone, self.video.path.name)
        self._publish(status="recording", frames=0, gb=0.0)

    def _disk_full(self) -> bool:
        try:
            return shutil.disk_usage(self.folder).free / 1e9 < self.min_free_gb
        except OSError:
            return False

    # ---- loop
    def run(self) -> None:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            grabber = Grabber(self.cfg.monitor)
            self._window = GameWindowCheck(mss_rect(grabber.monitor))
        except Exception as e:                       # mss or folder errors: report them, there's nothing to film
            log.exception("recorder failed to start")
            self._publish(status="error", error=str(e))
            return
        last_disk_check, disk_full, last_monitor_check = -1e9, False, -1e9
        while not self.stopping:
            t0 = time.time()                         # frame time (wall clock); waiting uses the monotonic clock
            m0 = time.monotonic()
            try:
                self._follow()
                if not self._on:
                    self._publish(filming=False, result=self._off_text)
                elif self._visit == self._broken_visit:
                    self._publish(filming=False, result="video cannot be written")
                elif self.video is None or self.video.visit != self._visit:
                    # no zone, a town, or the character select (a paused video waits for you to come back)
                    self._publish(filming=False, result="town" if is_town(*self._visit) else "no zone")
                elif self._held:
                    self._not_filming(m0, self._held_text)   # a gap when you go on, just like after an alt-tab
                elif not self._window.game_in_front():
                    # you alt-tabbed out, or the game is on another monitor; never film other windows
                    self._not_filming(m0, self._window.reason)
                    moved = self._window.reason == "game on another monitor"
                    if moved and m0 - last_monitor_check > MONITOR_CHECK_EVERY:
                        last_monitor_check = m0
                        grabber = self._new_monitor(grabber)
                else:
                    if m0 - last_disk_check > DISK_CHECK_EVERY:
                        last_disk_check, was_full = m0, disk_full
                        disk_full = self._disk_full()
                        if disk_full and not was_full:
                            log.warning("less than %.0f GB free in %s - filming paused", self.min_free_gb, self.folder)
                    if disk_full:
                        self._not_filming(m0, "disk full")
                    else:
                        self._film(grabber.grab(), t0, m0)
                self._loop_ok()
            except VideoWriteError as e:
                log.error("%s - this visit is not filmed", e)
                self._broken_visit = self._visit
                self._finish()
                self._publish(filming=False, error=str(e))
            except Exception as e:                   # keep filming, whatever happened to one frame
                self._loop_failed(e)
            self._stop_event.wait(max(0.0, self.interval - (time.monotonic() - m0)))
        try:
            self._finish()
        except Exception:
            log.exception("finishing the last video failed")

    def _pause(self, t: float) -> None:
        if self._paused_since is None:
            self._paused_since = t

    def _not_filming(self, t: float, result: str) -> None:
        """This frame isn't filmed (paused, game not in front, disk full). t: monotonic time, for GAP_AFTER."""
        self._pause(t)
        self._publish(filming=False, result=result)

    def _new_monitor(self, grabber: Grabber) -> Grabber:
        """The filmed monitor as the system reports it right now, since its resolution or position may have changed."""
        new = Grabber(self.cfg.monitor)
        rect = mss_rect(new.monitor)
        if rect == self._window.monitor:
            new.close()
            return grabber
        log.info("the filmed monitor changed from %s to %s", self._window.monitor, rect)
        self._window.monitor = rect
        grabber.close()
        return new

    def _film(self, frame: np.ndarray, t: float, now: float | None = None) -> None:
        """t is the frame's time (wall clock). now is the monotonic clock (default: now), used to tell how long a
        pause was."""
        if self._paused_since is not None:
            now = time.monotonic() if now is None else now
            away, self._paused_since = now - self._paused_since, None
            if self.video.frames and away >= GAP_AFTER:
                self.video.gap()
        if not self.video.write(frame, t):
            log.info("screen size changed - continuing in a new video")
            self._finish()
            self._start()
            self.video.write(frame, t)
        self._publish(filming=True, result="filming", frames=self.video.frames, gb=self.video.gb(), error=None)
