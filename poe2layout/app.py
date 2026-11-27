"""Entry points: the capture (films zones while you play and analyzes them in the background), and the standalone
analysis of videos left in the video folder."""
from __future__ import annotations

import time

from . import __version__
from .analyzer import Analyzer
from .config import Config, ConfigError, load_config
from .detector import Detector
from .hotkey import Hotkey
from .logs import get_logger, setup_logging, status_logger
from .recorder import ZoneRecorder
from .winapi import AlreadyRunning, SingleInstance, set_dpi_aware
from .zones import ZoneData

log = get_logger("app")
STATUS_EVERY = 5.0         # seconds between status lines while only the counters change


def analysis_text(an: dict) -> str:
    waiting = f", {an['queued']} waiting" if an.get("queued") else ""
    failed = f", {an['failed']} failed (see the log)" if an.get("failed") else ""
    if an.get("status") == "analyzing":
        return f"analyzing {an.get('zone')} {an.get('done', 0)}/{an.get('total', 0)} frames{waiting}{failed}"
    return f"analysis idle{waiting}{failed}"


def status_line(detector: dict, recorder: dict, analysis: dict) -> str:
    if detector.get("status") == "error":
        return f"zone detection error: {detector.get('error', '')}"
    if recorder.get("status") == "error" or recorder.get("error"):
        return f"recorder error: {recorder.get('error', '')} | {analysis_text(analysis)}"
    zone = detector.get("zone")
    if not zone:
        return f"waiting for a zone | {analysis_text(analysis)}"
    what = "filming" if recorder.get("filming") else f"paused ({recorder.get('result', 'waiting')})"
    return (f"{zone}: {what} | {recorder.get('frames', 0)} frames, {recorder.get('gb', 0.0):.1f} GB | "
            f"{analysis_text(analysis)}")


def status_events(detector: dict, recorder: dict) -> tuple:
    """The parts of the status line that get printed right away when they change: zone, filming or not (and why), and
    errors. If only the counters change (frames, GB, analysis progress), we print at most every STATUS_EVERY seconds."""
    return (detector.get("status"), detector.get("zone"), detector.get("error"),
            recorder.get("filming"), recorder.get("result"), recorder.get("error"))


def was_pressed(key: Hotkey | None) -> bool:
    """True once for each press of the hotkey."""
    if key is None or not key.pressed.is_set():
        return False
    key.pressed.clear()
    return True


def capture(cfg: Config) -> None:
    """Film zone visits until Ctrl+C or the quit hotkey. The videos still waiting get analyzed before we exit."""
    set_dpi_aware()
    zones = ZoneData(learned_path=cfg.learned_exits_path)
    detector = Detector(cfg)
    analyzer = Analyzer(cfg, zones)
    recorder = ZoneRecorder(cfg, analyzer.add)
    hotkey = Hotkey(cfg.hotkeys.quit, "quit")
    record = Hotkey(cfg.hotkeys.record, "record") if cfg.recorder.manual else None
    pause = Hotkey(cfg.hotkeys.pause, "pause")
    hotkeys = (hotkey, record, pause)
    for t in (detector, analyzer, recorder, *hotkeys):
        if t:
            t.start()
    log.info("capture running - Ctrl+C or %s stops it (the current zone is analyzed first)", hotkey.spec)
    if record:
        log.info("manual recording: %s starts filming, and pressing it again stops it", record.spec)
    log.info("%s pauses filming, and pressing it again goes on with the same video", pause.spec)
    status = status_logger()
    last_line, last_events, last_t = None, None, 0.0
    try:
        while not hotkey.pressed.is_set():
            ds = detector.get_state()
            # the game log's instance (zone code + seed) lets a portal trip back continue the same map
            recorder.set_zone(ds.get("zone"), ds.get("instance"))
            if was_pressed(record):
                recorder.toggle_recording()
            if was_pressed(pause):
                recorder.toggle_pause()
            rs = recorder.get_state()
            line, events = status_line(ds, rs, analyzer.get_state()), status_events(ds, rs)
            now = time.time()
            if line != last_line and (events != last_events or now - last_t >= STATUS_EVERY):
                status.info(line)
                last_line, last_events, last_t = line, events, now
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
        for key in hotkeys:
            if key:
                key.stop()
        recorder.stop()
        try:
            recorder.join(timeout=10)            # closes the current video and hands it to the analysis
        except KeyboardInterrupt:
            analyzer.stop()
            log.info("stopping right away - the videos stay in the video folder and are analyzed on the next start")
        else:
            finish_analysis(analyzer)


def finish_analysis(analyzer: Analyzer) -> None:
    """Analyze the videos still waiting before we exit. A second Ctrl+C leaves them for the next start."""
    analyzer.finish()
    an = analyzer.get_state()
    if an.get("status") == "analyzing" or an.get("queued"):
        log.info("stopping - analyzing the videos still waiting (Ctrl+C again leaves them for the next start)")
    status = status_logger()
    try:
        last = 0.0
        while analyzer.is_alive():
            analyzer.join(timeout=0.5)
            if time.time() - last >= STATUS_EVERY and analyzer.is_alive():
                status.info(analysis_text(analyzer.get_state()))
                last = time.time()
    except KeyboardInterrupt:
        analyzer.stop()
        log.info("analysis interrupted - the videos stay in the video folder and are analyzed on the next start")


def analyze_leftovers(cfg: Config) -> None:
    """Analyze the videos left in the video folder, then exit."""
    zones = ZoneData(learned_path=cfg.learned_exits_path)
    analyzer = Analyzer(cfg, zones)
    if analyzer.get_state()["queued"] == 0:
        log.info("no videos to analyze in %s", cfg.video_dir)
    analyzer.start()
    finish_analysis(analyzer)


def main(command: str = "capture") -> int:
    try:
        cfg, problems = load_config()
    except ConfigError as e:
        print(f"config.json: {e}")
        return 2
    setup_logging(cfg.log_level, cfg.logs_dir)
    log.info("poe2layout %s", __version__)
    for problem in problems:
        log.warning(problem)
    try:
        with SingleInstance():
            if command == "analyze":
                analyze_leftovers(cfg)
            else:
                capture(cfg)
    except AlreadyRunning as e:
        log.error("not started: %s", e)
        return 1
    except KeyboardInterrupt:                    # Ctrl+C while starting up
        return 130
    except Exception:
        log.exception("capture crashed")
        raise
    finally:
        log.info("stopped")
    return 0
