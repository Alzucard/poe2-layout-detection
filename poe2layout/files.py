"""File helpers: reading and writing images on non-ASCII Windows paths, safe folder names, and JSON writes that
survive a crash."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def imread(path: Path) -> np.ndarray | None:
    """cv2.imread that also works for non-ASCII Windows paths. Returns None when the file is missing or unreadable."""
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def imwrite(path: Path, img: np.ndarray) -> bool:
    """cv2.imwrite that also works for non-ASCII Windows paths. It writes through a temporary file and a rename, like
    write_text_atomic, so a crash never leaves half a picture behind."""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if ok:
        _replace_atomic(path, lambda tmp: buf.tofile(str(tmp)))
    return bool(ok)


def safe_name(s: str) -> str:
    """s turned into a valid Windows file or folder name."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip().rstrip(".")
    if s.split(".")[0].upper() in _RESERVED:
        s = "_" + s
    return s or "_"


def _replace_atomic(path: Path, write) -> None:
    """Call write(tmp) on a temporary file, then rename it to path. A crash leaves either the old file or the new
    one, never half of one. If the write or the rename fails (another program has path locked), the temporary file is
    removed."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """Write through a temporary file and a rename (see _replace_atomic)."""
    _replace_atomic(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def write_json_atomic(path: Path, data, indent: int = 1, sort_keys: bool = False) -> None:
    write_text_atomic(path, json.dumps(data, indent=indent, sort_keys=sort_keys))


def read_json(path: Path, default=None):
    """The JSON content of path, or `default` if the file doesn't exist. Raises ValueError if it's corrupt."""
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
