"""Global hotkeys (quit, and start/stop recording). RegisterHotKey works while the game has focus and
doesn't need admin rights."""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import itertools
import threading

from .logs import get_logger
from .winapi import kernel32, user32

log = get_logger("hotkey")

MODIFIERS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
VK_F1 = 0x70
_ids = itertools.count(1)   # each hotkey gets its own id, so two can be registered at once


def parse_hotkey(spec: str) -> tuple[int, int]:
    """"ctrl+alt+q" -> (modifier flags, virtual key code). Supports letters, digits and F1-F24."""
    mods, vk = MOD_NOREPEAT, None
    for part in spec.lower().replace(" ", "").split("+"):
        if part in MODIFIERS:
            mods |= MODIFIERS[part]
        elif len(part) == 1 and part.isascii() and part.isalnum():
            vk = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
            vk = VK_F1 + int(part[1:]) - 1
        else:
            raise ValueError(f"unsupported key {part!r} in hotkey {spec!r}")
    if vk is None:
        raise ValueError(f"hotkey {spec!r} has no key")
    return mods, vk


class Hotkey(threading.Thread):
    """Sets `pressed` when the hotkey is pressed, wherever you are. `name` is its key under "hotkeys" in the
    config, used in the log."""

    def __init__(self, spec: str, name: str = "quit"):
        super().__init__(daemon=True, name=f"hotkey-{name}")
        self.spec, self.what = spec, name
        self.pressed = threading.Event()
        self._id = next(_ids)
        self._thread_id: int | None = None

    def run(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        try:
            mods, vk = parse_hotkey(self.spec)
        except ValueError as e:
            log.warning("hotkeys.%s: %s", self.what, e)
            return
        if not user32.RegisterHotKey(None, self._id, mods, vk):
            log.warning("could not register the %s hotkey %s (used by another program?)", self.what, self.spec)
            return
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam == self._id:
                self.pressed.set()
        user32.UnregisterHotKey(None, self._id)

    def stop(self) -> None:
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
