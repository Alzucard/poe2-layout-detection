"""Windows calls: DPI awareness, "is the game in front, on the filmed monitor?", only one running copy at a time, and
thread priority."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path

from .logs import get_logger

log = get_logger("winapi")

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _declare(fn, restype, *argtypes) -> None:
    """Sets the argument and result types of a Windows function. Without them ctypes passes every number as a 32-bit
    int, which cuts 64-bit handles short."""
    fn.restype = restype
    fn.argtypes = list(argtypes)


_declare(kernel32.CreateMutexW, wintypes.HANDLE, ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
_declare(kernel32.OpenProcess, wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_declare(kernel32.QueryFullProcessImageNameW, wintypes.BOOL, wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
         ctypes.POINTER(wintypes.DWORD))
_declare(kernel32.CloseHandle, wintypes.BOOL, wintypes.HANDLE)
_declare(kernel32.GetCurrentThread, wintypes.HANDLE)
_declare(kernel32.SetThreadPriority, wintypes.BOOL, wintypes.HANDLE, ctypes.c_int)
_declare(kernel32.GetCurrentThreadId, wintypes.DWORD)
_declare(user32.GetForegroundWindow, wintypes.HWND)
_declare(user32.GetDesktopWindow, wintypes.HWND)
_declare(user32.GetWindowThreadProcessId, wintypes.DWORD, wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
_declare(user32.SetProcessDPIAware, wintypes.BOOL)
_declare(user32.RegisterHotKey, wintypes.BOOL, wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
_declare(user32.UnregisterHotKey, wintypes.BOOL, wintypes.HWND, ctypes.c_int)
_declare(user32.GetMessageW, wintypes.BOOL, ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT,
         wintypes.UINT)
_declare(user32.PostThreadMessageW, wintypes.BOOL, wintypes.DWORD, wintypes.UINT, wintypes.WPARAM,
         wintypes.LPARAM)


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD)]


_declare(user32.MonitorFromWindow, wintypes.HANDLE, wintypes.HWND, wintypes.DWORD)
_declare(user32.GetMonitorInfoW, wintypes.BOOL, wintypes.HANDLE, ctypes.POINTER(MONITORINFO))

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_ALREADY_EXISTS = 183
THREAD_PRIORITY_BELOW_NORMAL = -1
MONITOR_DEFAULTTONEAREST = 2

Rect = tuple[int, int, int, int]          # left, top, right, bottom (screen pixels)


def set_dpi_aware() -> None:
    """Make screen coordinates real pixels, so captures line up on scaled displays."""
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def lower_thread_priority() -> None:
    """Run the calling thread below normal priority, so the game gets the CPU first."""
    try:
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)
    except OSError:
        pass


def _exe_name(pid: int) -> str | None:
    """The file name of a process's program, or None when we can't tell."""
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        buf, n = ctypes.create_unicode_buffer(520), wintypes.DWORD(520)
        if not kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            return None
    finally:
        kernel32.CloseHandle(h)
    return Path(buf.value).name


def window_monitor_rect(hwnd) -> Rect | None:
    """The monitor a window is (mostly) on, or None when we can't tell."""
    mon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    if not mon or not user32.GetMonitorInfoW(mon, ctypes.byref(info)):
        return None
    r = info.rcMonitor
    return r.left, r.top, r.right, r.bottom


def mss_rect(monitor: dict) -> Rect:
    """An mss monitor entry ({"left", "top", "width", "height"}) as a Rect."""
    return monitor["left"], monitor["top"], monitor["left"] + monitor["width"], monitor["top"] + monitor["height"]


class GameWindowCheck:
    """Is Path of Exile the foreground window, on the filmed monitor? If we can't tell, the answer is no: we only
    film the screen when it definitely shows the game, never other windows. `reason` says why not."""

    def __init__(self, monitor: Rect | None = None):
        self.monitor = monitor                            # the filmed monitor; None means don't check
        self.reason = ""
        self._cache: dict[tuple[int, int], bool] = {}     # (window, process id) -> is it the game
        self._failure_logged = False
        self._elsewhere_logged = False
        self._unknown_logged = False

    def game_in_front(self) -> bool:
        try:
            hwnd = user32.GetForegroundWindow()
            if not self._is_game(hwnd):
                self.reason = "game not in front"
                return False
            if self.monitor is not None and window_monitor_rect(hwnd) != self.monitor:
                self.reason = "game on another monitor"
                if not self._elsewhere_logged:
                    log.warning("the game is on another monitor than the one filmed (%s) - set \"monitor\" in "
                                "config.json; not filming meanwhile", self.monitor)
                    self._elsewhere_logged = True
                return False
            self.reason, self._elsewhere_logged = "", False
            return True
        except Exception as e:      # broad on purpose: any failure has to mean "don't film"
            if not self._failure_logged:
                log.warning("cannot tell which window is in front (%s) - not filming until it can", e)
                self._failure_logged = True
            self.reason = "game not in front"
            return False

    def _is_game(self, hwnd) -> bool:
        if not hwnd:
            return False
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        key = (int(hwnd), pid.value)
        if key not in self._cache:
            exe = _exe_name(pid.value)
            if exe is None:
                if not self._unknown_logged:
                    log.warning("cannot tell which program the window in front belongs to (process %d) - not "
                                "filming while it is in front", pid.value)
                    self._unknown_logged = True
                return False
            if len(self._cache) > 200:
                self._cache.clear()
            self._cache[key] = "pathofexile" in exe.lower()
        return self._cache[key]


class AlreadyRunning(RuntimeError):
    pass


class SingleInstance:
    """A named mutex, so a second copy of the capture or analysis (they share the video folder) refuses to start. Use
    it as a context manager."""

    def __init__(self, name: str = "Local\\poe2layout-capture"):
        self.name = name
        self._handle = None

    def __enter__(self) -> SingleInstance:
        h = kernel32.CreateMutexW(None, False, self.name)
        if not h:
            raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(h)
            raise AlreadyRunning("the capture or the analysis is already running in another window")
        self._handle = h
        return self

    def __exit__(self, *exc) -> None:
        if self._handle:
            kernel32.CloseHandle(self._handle)
            self._handle = None
