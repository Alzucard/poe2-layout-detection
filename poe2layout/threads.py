"""Base class for the background workers: a state dict others can read safely, a stop request, and handling for a
round of the worker's loop that failed."""
from __future__ import annotations

import threading

from .logs import get_logger


class StatefulThread(threading.Thread):
    def __init__(self, name: str, state: dict):
        super().__init__(daemon=True, name=name)
        self._state_lock = threading.Lock()
        self._state = dict(state)
        # not "_stop": up to Python 3.12, threading.Thread has a _stop() method that join() calls
        self._stop_event = threading.Event()
        self._log = get_logger(name)
        self._last_loop_error: str | None = None

    def get_state(self) -> dict:
        """A copy of the current state (thread-safe)."""
        with self._state_lock:
            return dict(self._state)

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def _publish(self, **changes) -> None:
        with self._state_lock:
            self._state.update(changes)

    def _loop_failed(self, e: Exception) -> None:
        """A round of the loop failed. The first time we log the traceback; if the same error repeats we only log it
        at DEBUG, otherwise a failing screen grab would fill the log every second. Then we wait a second, or less if
        we're asked to stop."""
        msg = f"{type(e).__name__}: {e}"
        if msg != self._last_loop_error:
            self._log.exception("%s loop error", self.name)
            self._last_loop_error = msg
        else:
            self._log.debug("%s loop error again: %s", self.name, msg)
        self._publish(error=str(e))
        self._stop_event.wait(1.0)

    def _loop_ok(self) -> None:
        """A round of the loop went through, so any earlier loop error is over."""
        if self._last_loop_error is not None:
            self._log.info("%s working again after: %s", self.name, self._last_loop_error)
            self._last_loop_error = None
            self._publish(error=None)
