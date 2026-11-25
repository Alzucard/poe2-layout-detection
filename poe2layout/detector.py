"""Background thread that knows which zone you're in, from the game log (Client.txt).

We never run OCR on the live game screen (game ToS); only on the recorded videos, after the fact. The zone name is
taken exactly as the game log spells it.
"""
from __future__ import annotations

import time

from .config import Config
from .gamelog import GameLog, ZoneEvent, find_client_log
from .logs import get_logger
from .threads import StatefulThread
from .zones import Instance

log = get_logger("detector")


class Detector(StatefulThread):
    """The state has status ("starting", "waiting", "in zone", "error"), zone, instance and error."""

    def __init__(self, cfg: Config):
        super().__init__("detector", {"status": "starting", "zone": None, "instance": None})
        self.cfg = cfg
        self.zone: str | None = None
        self.instance: Instance | None = None
        self._seen_instances: set[Instance] = set()
        self._game_log: GameLog | None = None

    def run(self) -> None:
        path = find_client_log(self.cfg.client_log)
        if path is None:
            msg = "game log (Client.txt) not found - set client_log in config.json"
            log.error(msg)
            self._publish(status="error", error=msg)
            return
        try:
            self._game_log = GameLog(path)
        except OSError as e:
            log.exception("cannot open the game log %s", path)
            self._publish(status="error", error=str(e))
            return
        log.info("zone changes come from the game log: %s", path)
        self._publish(status="waiting")
        while not self.stopping:
            t0 = time.monotonic()
            try:
                for event in self._game_log.poll():
                    self._handle(event)
                self._loop_ok()
            except Exception as e:                     # keep following the log, whatever happened to one read
                self._loop_failed(e)
            self._stop_event.wait(max(0.0, self.cfg.poll_interval - (time.monotonic() - t0)))

    def _handle(self, event: ZoneEvent) -> None:
        if event.zone is None:
            if self.zone:
                log.info("left the game world (character select / login)")
            self.zone, self.instance = None, None
            self._publish(zone=None, instance=None, status="waiting")
            return
        if event.zone == self.zone and event.instance == self.instance:
            return
        again = "same instance as before" if event.instance in self._seen_instances else "new instance"
        self._seen_instances.add(event.instance)
        log.info("entered %r (game log: %s level %s seed %s, %s)", event.zone, event.instance.code, event.level,
                 event.instance.seed, again)
        self.zone, self.instance = event.zone, event.instance
        self._publish(zone=event.zone, instance=event.instance, status="in zone")
