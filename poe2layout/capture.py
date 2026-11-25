"""Screen capture of one monitor."""
from __future__ import annotations

import mss
import numpy as np

from .logs import get_logger

log = get_logger("capture")


class Grabber:
    """Create it in the thread that uses it (mss handles are per thread)."""

    def __init__(self, monitor: int = 1):
        self._sct = mss.MSS()
        monitors = self._sct.monitors                 # [0] is all monitors together, [1] is the main one, ...
        if not 1 <= monitor < len(monitors):
            log.warning("monitor %d does not exist (%d found) - using the main monitor", monitor, len(monitors) - 1)
            monitor = 1
        self.monitor = monitors[monitor]

    def grab(self) -> np.ndarray:
        """The whole monitor as a BGR image."""
        shot = self._sct.grab(self.monitor)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def close(self) -> None:
        self._sct.close()
