"""Zone detection from game log events, and the status line."""
import os
import time

import pytest
from test_gamelog import enter

from poe2layout.app import analysis_text, status_events, status_line
from poe2layout.config import Config
from poe2layout.detector import Detector
from poe2layout.gamelog import ZoneEvent
from poe2layout.hotkey import Hotkey
from poe2layout.winapi import GameWindowCheck, _exe_name, mss_rect, user32, window_monitor_rect
from poe2layout.zones import Instance

A = ZoneEvent("Clearfell", Instance("G1_2", 1), 3)


def test_detector_follows_zone_changes():
    d = Detector(Config())
    d._handle(A)
    assert d.get_state() == {"status": "in zone", "zone": "Clearfell", "instance": Instance("G1_2", 1)}
    d._handle(ZoneEvent(None, None, None))                       # character select
    assert d.get_state()["zone"] is None and d.get_state()["status"] == "waiting"
    d._handle(A)
    assert d.get_state()["zone"] == "Clearfell"
    d._handle(ZoneEvent("Clearfell", Instance("G1_2", 2), 3))   # same zone, new instance
    assert d.get_state()["instance"] == Instance("G1_2", 2)


def test_detector_logs_each_visit_once(caplog):
    d = Detector(Config())
    caplog.set_level("INFO", logger="poe2.detector")
    d._handle(A)
    d._handle(A)                                                 # the same scene loaded again, nothing new
    d._handle(ZoneEvent("Clearfell Encampment", Instance("G1_town", 9), 3))
    d._handle(A)
    entered = [r.getMessage() for r in caplog.records if r.getMessage().startswith("entered")]
    assert len(entered) == 3
    assert "new instance" in entered[0] and "same instance as before" in entered[2]


def test_detector_waits_for_the_first_zone(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text("", encoding="utf-8")
    d = Detector(Config(client_log=str(p), poll_interval=0.01))
    d.start()
    try:
        wait_for(d, status="waiting")
    finally:
        d.stop()
        d.join(timeout=5)


def wait_for(t, **want):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        if all(t.get_state().get(k) == v for k, v in want.items()):
            return
        time.sleep(0.01)
    raise AssertionError(f"state {t.get_state()} never had {want}")


def test_detector_loop_follows_the_game_log(tmp_path):
    p = tmp_path / "Client.txt"
    p.write_text(enter("G1_2", 2, "Clearfell", 3), encoding="utf-8")
    d = Detector(Config(client_log=str(p), poll_interval=0.01))
    d.start()
    try:
        wait_for(d, status="in zone", zone="Clearfell", instance=Instance("G1_2", 2))
        with open(p, "a", encoding="utf-8") as f:
            f.write(enter("G1_town", 9, "Clearfell Encampment"))
        wait_for(d, status="in zone", zone="Clearfell Encampment", instance=Instance("G1_town", 9))
    finally:
        d.stop()
        d.join(timeout=5)
    assert not d.is_alive()


def test_detector_without_a_game_log(tmp_path):
    d = Detector(Config(client_log=str(tmp_path / "missing.txt")))
    d.run()                                                      # returns right away
    assert d.get_state()["status"] == "error" and "Client.txt" in d.get_state()["error"]


def test_analysis_text():
    assert analysis_text({"status": "idle"}) == "analysis idle"
    assert analysis_text({"status": "analyzing", "zone": "Deshar", "done": 10, "total": 50, "queued": 2}) == \
        "analyzing Deshar 10/50 frames, 2 waiting"
    assert analysis_text({"status": "idle", "failed": 1}) == "analysis idle, 1 failed (see the log)"


def test_status_line():
    idle = {"status": "idle"}
    assert status_line({"status": "waiting"}, {}, idle) == "waiting for a zone | analysis idle"
    assert status_line({"zone": "Clearfell"}, {"filming": True, "frames": 4, "gb": 0.02}, idle) == \
        "Clearfell: filming | 4 frames, 0.0 GB | analysis idle"
    assert status_line({"zone": "Clearfell"}, {"filming": False, "result": "town"}, idle).startswith(
        "Clearfell: paused (town)")
    assert status_line({"status": "error", "error": "no log"}, {}, idle) == "zone detection error: no log"
    assert status_line({"zone": "X"}, {"error": "disk"}, idle) == "recorder error: disk | analysis idle"


def test_status_events_ignore_counters():
    zone = {"status": "in zone", "zone": "Clearfell"}
    filming = {"filming": True, "result": "filming", "frames": 4, "gb": 0.02}
    assert status_events(zone, filming) == status_events(zone, dict(filming, frames=90, gb=0.5))
    assert status_events(zone, filming) != status_events(zone, dict(filming, filming=False, result="town"))
    assert status_events(zone, filming) != status_events(dict(zone, zone="Grelwood"), filming)


def test_exe_name_of_this_process():
    assert _exe_name(os.getpid()).lower().startswith("python")
    assert _exe_name(0) is None


def test_window_monitor_is_one_of_the_monitors():
    import mss
    with mss.MSS() as sct:
        rects = [mss_rect(m) for m in sct.monitors[1:]]
    assert window_monitor_rect(user32.GetDesktopWindow()) in rects
    assert mss_rect({"left": -1920, "top": 0, "width": 1920, "height": 1080}) == (-1920, 0, 0, 1080)


def test_game_check_says_why_not(monkeypatch):
    check = GameWindowCheck(monitor=(0, 0, 1, 1))
    monkeypatch.setattr(check, "_is_game", lambda hwnd: True)
    assert not check.game_in_front() and check.reason == "game on another monitor"
    monkeypatch.setattr(check, "_is_game", lambda hwnd: False)
    assert not check.game_in_front() and check.reason == "game not in front"


def test_hotkeys_start_and_stop():
    """The quit and record hotkeys are registered at the same time, each in its own thread."""
    keys = [Hotkey("ctrl+alt+shift+f24", "quit"), Hotkey("ctrl+alt+shift+f23", "record")]
    for hk in keys:
        hk.start()
    for hk in keys:
        for _ in range(50):
            if hk._thread_id:
                break
            time.sleep(0.02)
    time.sleep(0.1)
    assert all(hk.is_alive() for hk in keys)                     # a failed RegisterHotKey ends the thread
    for hk in keys:
        hk.stop()
        hk.join(timeout=2)
    assert not any(hk.is_alive() or hk.pressed.is_set() for hk in keys)


def test_repeated_loop_error_is_logged_once(caplog):
    from poe2layout.threads import StatefulThread
    t = StatefulThread("worker", {})
    t.stop()                                                     # no waiting in _loop_failed
    with caplog.at_level("DEBUG", logger="poe2.worker"):
        for _ in range(3):
            t._loop_failed(OSError("grab failed"))
    assert [r.levelname for r in caplog.records] == ["ERROR", "DEBUG", "DEBUG"]
    assert t.get_state()["error"] == "grab failed"
    t._loop_ok()
    assert t.get_state()["error"] is None
    t._loop_failed(OSError("grab failed"))                       # after recovering it's reported in full again
    assert caplog.records[-1].levelname == "ERROR"


def test_unknown_program_in_front_is_logged_once(monkeypatch, caplog):
    import poe2layout.winapi as winapi
    monkeypatch.setattr(winapi, "_exe_name", lambda pid: None)
    check = GameWindowCheck()
    assert not check._is_game(12345) and not check._is_game(12345)
    assert sum("cannot tell which program" in r.message for r in caplog.records) == 1


def test_failing_window_check_means_do_not_film(monkeypatch):
    import poe2layout.winapi as winapi

    def broken():
        raise OSError("no desktop")

    monkeypatch.setattr(winapi.user32, "GetForegroundWindow", broken)
    check = GameWindowCheck()
    assert not check.game_in_front() and check.reason == "game not in front"


def test_only_one_copy_runs_at_a_time():
    from poe2layout.winapi import AlreadyRunning, SingleInstance
    name = rf"Local\poe2layout-test-{os.getpid()}"
    with SingleInstance(name), pytest.raises(AlreadyRunning), SingleInstance(name):
        pass
    with SingleInstance(name):                                     # free again once the first one ended
        pass


def test_windows_calls_declare_their_types():
    # without argtypes, ctypes passes 32-bit ints and cuts 64-bit handles short
    from ctypes import wintypes

    from poe2layout.winapi import kernel32
    assert kernel32.OpenProcess.restype is wintypes.HANDLE
    assert kernel32.CloseHandle.argtypes == [wintypes.HANDLE]
    assert user32.GetWindowThreadProcessId.argtypes[0] is wintypes.HWND


class StubbornAnalyzer:
    """Its analysis takes a long time, and a second Ctrl+C interrupts join()."""

    def __init__(self):
        self.stopped = False

    def finish(self):
        pass

    def get_state(self):
        return {"status": "analyzing", "queued": 1}

    def is_alive(self):
        return not self.stopped

    def join(self, timeout=None):
        raise KeyboardInterrupt

    def stop(self):
        self.stopped = True


def test_second_ctrl_c_stops_the_analysis():
    from poe2layout.app import finish_analysis
    an = StubbornAnalyzer()
    finish_analysis(an)                                             # returns instead of raising
    assert an.stopped


def test_ctrl_c_while_starting_exits_quietly(monkeypatch, tmp_path):
    import poe2layout.app as app
    from poe2layout.config import config_from_dict
    cfg, _ = config_from_dict({"data_dir": str(tmp_path)})
    monkeypatch.setattr(app, "load_config", lambda: (cfg, []))
    monkeypatch.setattr(app, "setup_logging", lambda *a: None)          # leave the test run's logging alone
    monkeypatch.setattr(app, "SingleInstance", lambda: __import__("contextlib").nullcontext())

    def interrupted(cfg):
        raise KeyboardInterrupt

    monkeypatch.setattr(app, "capture", interrupted)
    assert app.main() == 130
