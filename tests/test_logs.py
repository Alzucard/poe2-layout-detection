"""Logging setup: events go to logs/capture.log, and the status line only goes to the console."""
import logging

import pytest

from poe2layout import logs


@pytest.fixture
def fresh_logging(monkeypatch):
    """setup_logging like on a fresh start. The handlers it adds are removed afterwards."""
    monkeypatch.setattr(logs, "_ready", False)
    loggers = [logging.getLogger("poe2"), logging.getLogger("poe2status")]
    before = [(lg, list(lg.handlers), lg.level, lg.propagate) for lg in loggers]
    yield
    for lg, handlers, level, propagate in before:
        for h in lg.handlers:
            if h not in handlers:
                h.close()
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate


def file_text(logs_dir) -> str:
    for h in logging.getLogger("poe2").handlers:
        h.flush()
    return (logs_dir / "capture.log").read_text("utf-8")


def test_events_reach_the_log_file(tmp_path, fresh_logging):
    logs.setup_logging("info", tmp_path / "logs")
    logs.get_logger("stitcher").info("map started")
    logs.get_logger("stitcher").debug("hidden detail")
    text = file_text(tmp_path / "logs")
    assert "poe2.stitcher" in text and "map started" in text
    assert "hidden detail" not in text


def test_status_line_stays_out_of_the_log_file(tmp_path, fresh_logging):
    logs.setup_logging("debug", tmp_path / "logs")
    logs.status_logger().info("tracking 123 frames")
    assert "tracking 123 frames" not in file_text(tmp_path / "logs")


def test_status_line_goes_to_the_console_once(tmp_path, fresh_logging, capsys, caplog):
    logs.setup_logging("info", tmp_path / "logs")
    logs.status_logger().info("tracking 123 frames")
    assert capsys.readouterr().err.count("tracking 123 frames") == 1
    assert "tracking 123 frames" not in caplog.text            # not passed on to the root logger


def test_debug_goes_to_file_but_console_stays_at_info(tmp_path, fresh_logging):
    logs.setup_logging("DEBUG", tmp_path / "logs")
    logs.get_logger("x").debug("fine detail")
    assert "fine detail" in file_text(tmp_path / "logs")
    console = [h for h in logging.getLogger("poe2").handlers if type(h) is logging.StreamHandler]
    assert [h.level for h in console] == [logging.INFO]


def test_unknown_level_falls_back_to_info(tmp_path, fresh_logging):
    logs.setup_logging("chatty", tmp_path / "logs")
    assert logging.getLogger("poe2").level == logging.INFO


def test_second_call_adds_no_handlers(tmp_path, fresh_logging):
    logs.setup_logging("info", tmp_path / "logs")
    n = len(logging.getLogger("poe2").handlers), len(logging.getLogger("poe2status").handlers)
    logs.setup_logging("debug", tmp_path / "other")
    assert (len(logging.getLogger("poe2").handlers), len(logging.getLogger("poe2status").handlers)) == n
    assert not (tmp_path / "other").exists()
