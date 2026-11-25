"""Training frames for a future outline detector, kept while the zone videos are analyzed.

With "training_frames": {"enabled": true}, the analysis keeps a raw frame of the video every `every` seconds (video
time) in dataset/<zone>/<instance id>/, as JPEG plus frames.jsonl with what the stitcher made of each frame. Frames
where tracking failed are kept too, since those are the interesting ones. The folder name matches the run folder
runs/<zone>/<instance id>/, so its finished map can label the frames later.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import TrainingFramesConfig
from .files import safe_name
from .logs import get_logger
from .stitcher import StepResult, Stitcher

log = get_logger("collector")


def _folder_mb(root: Path) -> float:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6 if root.is_dir() else 0.0


class TrainingFrameCollector:
    def __init__(self, cfg: TrainingFramesConfig, root: Path):
        self.cfg = cfg
        self.root = root
        self._folder: Path | None = None
        self._next_number = 0
        self._last_t = 0.0
        self._used_mb = _folder_mb(root) if cfg.enabled else 0.0
        self._full_logged = False
        if cfg.enabled:
            log.info("keeping a training frame every %.0f s in %s (%.0f of %.0f MB used)",
                     cfg.every, root, self._used_mb, cfg.max_mb)

    def start(self, zone: str, run_id: str) -> None:
        """A new (or resumed) visit. Frames go to that run's folder, and the same instance adds to it."""
        self._last_t = 0.0
        self._folder = None
        if self.cfg.enabled:
            self._folder = self.root / safe_name(zone) / safe_name(run_id)
            # start after the highest number, so a frame you deleted can't make the next one overwrite the last
            numbers = [int(p.stem) for p in self._folder.glob("*.jpg") if p.stem.isdigit()]
            self._next_number = max(numbers, default=-1) + 1

    def add(self, frame: np.ndarray, stitcher: Stitcher, result: StepResult, t: float) -> None:
        if self._folder is None or result is StepResult.SKIPPED or t - self._last_t < self.cfg.every:
            return
        if self._used_mb >= self.cfg.max_mb:
            if not self._full_logged:
                log.warning("training frames: max_mb (%.0f MB) reached, no more frames kept", self.cfg.max_mb)
                self._full_logged = True
            return
        self._last_t = t
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            if not ok:
                return
            self._folder.mkdir(parents=True, exist_ok=True)
            name = f"{self._next_number:04d}.jpg"
            buf.tofile(str(self._folder / name))
            self._next_number += 1
            record = {"file": name, "t": round(t, 2), "result": str(result), "method": str(stitcher.method),
                      "offset": list(stitcher.offset) if stitcher.offset is not None else None,
                      "shifted": list(stitcher.shifted), "pieces": len(stitcher.pieces), "frames": stitcher.frames,
                      "scale": stitcher.scale, "screen": [frame.shape[1], frame.shape[0]]}
            with open(self._folder / "frames.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self._used_mb += len(buf) / 1e6
        except OSError:
            log.exception("saving a training frame failed")
