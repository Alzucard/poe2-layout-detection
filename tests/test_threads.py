"""The worker base class: state copies and stopping (loop error logging is tested in test_app.py)."""
import time

from poe2layout.threads import StatefulThread


def test_state_is_a_copy():
    initial = {"status": "waiting"}
    t = StatefulThread("worker", initial)
    initial["status"] = "changed by the caller"
    state = t.get_state()
    state["status"] = "changed by a reader"
    assert t.get_state() == {"status": "waiting"}


def test_publish_merges_changes():
    t = StatefulThread("worker", {"status": "waiting", "zone": None})
    t._publish(zone="The Riverbank")
    assert t.get_state() == {"status": "waiting", "zone": "The Riverbank"}


def test_stop_request():
    t = StatefulThread("worker", {})
    assert not t.stopping and t.daemon
    t.stop()
    assert t.stopping


def test_loop_error_wait_ends_on_stop():
    t = StatefulThread("worker", {})
    t.stop()
    start = time.monotonic()
    t._loop_failed(OSError("grab failed"))
    assert time.monotonic() - start < 0.5


def test_loop_error_waits_a_second():
    t = StatefulThread("worker", {})
    waits = []
    t._stop_event.wait = waits.append                       # record the wait instead of sleeping
    t._loop_failed(OSError("grab failed"))
    assert waits == [1.0]


def test_loop_ok_without_an_error_changes_nothing():
    t = StatefulThread("worker", {"status": "running"})
    t._loop_ok()
    assert t.get_state() == {"status": "running"}
