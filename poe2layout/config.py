"""Settings: config.json on top of the defaults defined here (this is the only place defaults live).

Unknown keys and values of the wrong type get reported instead of being silently ignored.
"""
from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
ZONES_PATH = PROJECT_ROOT / "data" / "zones.json"       # reference data that ships with the program, read only
VIDEO_CODECS = {"HFYU": ".avi", "FFV1": ".mkv"}         # HuffYUV is fast to write, FFV1 is about 20% smaller
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
MIN_FPS = 0.2                                            # the recorder also notices zone changes once per frame
# settings from the first version: analysis.<old> is now training_frames.<new>
RENAMED = {"collect_frames": "enabled", "collect_every": "every", "collect_max_mb": "max_mb",
           "collect_quality": "jpeg_quality"}


class ConfigError(ValueError):
    pass


@dataclass
class RecorderConfig:
    fps: float = 2.0
    video_dir: str = "videos"          # relative paths are inside data_dir
    video_codec: str = "HFYU"
    min_free_gb: float = 20.0          # filming pauses when there's less free disk space than this
    manual: bool = True                # only film after hotkeys.record was pressed, until it's pressed again


@dataclass
class AnalysisConfig:
    ocr_every: float = 1.0             # seconds of video between label reads
    keep_instances: int = 2            # analyzed instances kept in memory (about 64 MB each) so their map can continue
    map_area: list[list[float]] | None = None   # screen polygon of the Tab map (fractions); None means the built-in one


@dataclass
class TrainingFramesConfig:
    enabled: bool = False
    every: float = 4.0                 # seconds of video between kept frames
    max_mb: float = 3000.0
    jpeg_quality: int = 95


@dataclass
class HotkeyConfig:
    quit: str = "ctrl+alt+q"
    record: str = "ctrl+alt+r"         # starts and stops filming (unless recorder.manual is off)
    pause: str = "ctrl+alt+p"          # pauses filming and goes on with the same video


@dataclass
class Config:
    monitor: int = 1                   # 1 is the main monitor
    client_log: str = ""               # Path of Exile 2/logs/Client.txt; leave empty to find it automatically
    poll_interval: float = 0.4         # seconds between reads of the game log
    data_dir: str = ""                 # where runs/, logs/, dataset/ and videos/ go; empty means the project folder
    log_level: str = "INFO"            # DEBUG logs every frame decision of the map stitching
    debug_snapshots: bool = False      # save the frame and its outline to logs/snapshots when tracking gets lost
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    training_frames: TrainingFramesConfig = field(default_factory=TrainingFramesConfig)

    # ---- where things get saved
    @property
    def data_root(self) -> Path:
        p = Path(self.data_dir) if self.data_dir else PROJECT_ROOT
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "logs"

    @property
    def snapshots_dir(self) -> Path:
        return self.logs_dir / "snapshots"

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / "dataset"

    @property
    def learned_exits_path(self) -> Path:
        return self.data_root / "learned_exits.json"

    @property
    def video_dir(self) -> Path:
        p = Path(self.recorder.video_dir or "videos")
        return p if p.is_absolute() else self.data_root / p

    @property
    def failed_video_dir(self) -> Path:
        return self.video_dir / "failed"

    def validate(self) -> None:
        def need(ok: bool, msg: str) -> None:
            if not ok:
                raise ConfigError(msg)

        need(self.monitor >= 1, "monitor must be 1 or higher")
        need(self.poll_interval > 0, "poll_interval must be > 0")
        need(self.log_level.upper() in LOG_LEVELS, f"log_level must be one of {', '.join(LOG_LEVELS)}")
        need(self.recorder.fps >= MIN_FPS, f"recorder.fps must be at least {MIN_FPS}")
        need(self.recorder.video_codec.upper() in VIDEO_CODECS,
             f"recorder.video_codec must be one of {', '.join(VIDEO_CODECS)}")
        need(self.analysis.ocr_every > 0, "analysis.ocr_every must be > 0")
        need(self.analysis.keep_instances >= 0, "analysis.keep_instances must be >= 0")
        need(1 <= self.training_frames.jpeg_quality <= 100, "training_frames.jpeg_quality must be 1..100")
        from .hotkey import parse_hotkey  # imported here, not at the top: hotkey.py loads the Windows libraries
        keys = {}
        for name in ("quit", "record", "pause"):
            try:
                keys[name] = parse_hotkey(getattr(self.hotkeys, name))
            except ValueError as e:
                raise ConfigError(f"hotkeys.{name}: {e}") from e
        used: dict[tuple[int, int], str] = {}
        for name, key in keys.items():
            if name == "record" and not self.recorder.manual:
                continue                       # not registered then, so it may share a key
            need(key not in used, f"hotkeys.{name} must be a different key than hotkeys.{used.get(key)}")
            used[key] = name
        area = self.analysis.map_area
        if area is not None:
            need(isinstance(area, list) and len(area) >= 3 and all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in p)
                for p in area), "analysis.map_area must be a list of at least 3 [x, y] screen fractions")


def _coerce(value, hint, key: str):
    """Check one plain value against its field type (ints are fine for floats)."""
    if hint is bool:
        ok = isinstance(value, bool)
    elif hint is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif hint is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif hint is str:
        ok = isinstance(value, str)
    else:
        ok = True                       # Config.validate checks the compound types
    if not ok:
        raise ConfigError(f"{key}: expected {getattr(hint, '__name__', hint)}, got {value!r}")
    return value


def _build(cls, data: dict, prefix: str, problems: list[str]):
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix.rstrip('.') or 'config'}: expected an object")
    hints = typing.get_type_hints(cls)
    obj = cls()
    names = {f.name for f in fields(cls)}
    for key, value in data.items():
        full = prefix + key
        if key not in names:
            problems.append(f"unknown setting {full!r} in config.json - ignored")
            continue
        hint = hints[key]
        if is_dataclass(hint):
            setattr(obj, key, _build(hint, value, full + ".", problems))
        else:
            setattr(obj, key, _coerce(value, hint, full))
    return obj


def _renamed(data: dict, problems: list[str]) -> dict:
    """data with the first version's settings under their new names (each one gets reported, so it can be renamed)."""
    analysis = data.get("analysis") if isinstance(data, dict) else None
    frames = (data.get("training_frames") or {}) if isinstance(data, dict) else None
    if not isinstance(analysis, dict) or not RENAMED.keys() & analysis.keys() or not isinstance(frames, dict):
        return data                              # _build reports a training_frames that isn't an object
    analysis, frames = dict(analysis), dict(frames)
    for old, new in RENAMED.items():
        if old in analysis:
            frames.setdefault(new, analysis.pop(old))
            problems.append(f"'analysis.{old}' is now 'training_frames.{new}' - rename it in config.json")
    return {**data, "analysis": analysis, "training_frames": frames}


def config_from_dict(data: dict) -> tuple[Config, list[str]]:
    """Returns (config, problems worth a warning). Raises ConfigError for values we can't use."""
    problems: list[str] = []
    data = _renamed(data, problems)
    cfg = _build(Config, data, "", problems)
    cfg.recorder.video_codec = cfg.recorder.video_codec.upper()
    cfg.log_level = {"WARN": "WARNING"}.get(cfg.log_level.upper(), cfg.log_level.upper())
    cfg.validate()
    return cfg, problems


def load_config(path: Path = CONFIG_PATH) -> tuple[Config, list[str]]:
    """Settings from config.json, or the defaults if there isn't one."""
    if not path.exists():
        return Config(), []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))    # -sig so files saved with a BOM work too
    except ValueError as e:
        raise ConfigError(f"{path.name} is not valid JSON: {e}") from e
    return config_from_dict(data)
