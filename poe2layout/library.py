"""Saved runs: runs/<zone>/<instance id>/ with map.png, meta.json, path.png and labels.png, plus
runs/<zone>/layouts.json (instance id -> layout letter, orientation, mirrored).

Folders are named after the instance id (zone code + seed), which never changes. Which layout a run belongs to is
only kept in layouts.json, because layouts can merge as more runs come in.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .files import imread, imwrite, read_json, safe_name, write_json_atomic
from .labels import entrance_point, without_screen_fixed
from .layouts import DEFAULT_THRESHOLDS, LayoutThresholds, MapVariants, group_runs
from .logs import get_logger
from .outline import denoise_map
from .render import render_floors, render_run_images
from .stitcher import Stitcher
from .zones import Instance

log = get_logger("library")

META_VERSION = 2
MIN_MAP_PIXELS = 2000          # we don't save a map with fewer stable pixels than this
MAX_SAVED_PATH_POINTS = 300    # meta.json keeps a thinned route (path.png is drawn from the full one)
NEW, OLD = ".new", ".old"      # a run gets written into <id>.new and then swapped in (the earlier one becomes <id>.old)


def thinned(points: list) -> list:
    """Every n-th point, so that we keep at most about MAX_SAVED_PATH_POINTS."""
    return points[::max(1, len(points) // MAX_SAVED_PATH_POINTS)]


@dataclass
class SavedRun:
    folder: Path
    id: str                    # instance id
    meta: dict
    maps: MapVariants          # the map, cleaned the same way for every run so they compare alike


@dataclass
class SaveResult:
    folder: Path
    meta: dict


class RunLibrary:
    """Not thread-safe; only the analysis thread uses it."""

    def __init__(self, runs_dir: Path, diagnostics_dir: Path | None = None, keep_zones: int = 2,
                 thresholds: LayoutThresholds = DEFAULT_THRESHOLDS):
        self.runs_dir = runs_dir
        self.diagnostics_dir = diagnostics_dir   # pictures of maps too small to save go here
        self.keep_zones = keep_zones             # zones whose runs stay loaded (plus turned variants, ~20 MB a run)
        self.thresholds = thresholds
        self._cache: dict[str, list[SavedRun]] = {}

    def zone_dir(self, zone: str) -> Path:
        return self.runs_dir / safe_name(zone)

    def run_dir(self, zone: str, run_id: str) -> Path:
        return self.zone_dir(zone) / safe_name(run_id)

    def layouts_path(self, zone: str) -> Path:
        return self.zone_dir(zone) / "layouts.json"

    def count(self, zone: str) -> int:
        """How many runs of a zone are saved, without loading them (and without touching saves that were cut short)."""
        return len(self._saved_folders(zone))

    def _saved_folders(self, zone: str) -> list[Path]:
        d = self.zone_dir(zone)
        return sorted(p for p in d.iterdir() if p.is_dir() and p.suffix not in (NEW, OLD)
                      and (p / "map.png").exists()) if d.is_dir() else []

    def _run_folders(self, zone: str) -> list[Path]:
        """The zone's run folders, after finishing or dropping any saves that were cut short (see save)."""
        d = self.zone_dir(zone)
        if not d.is_dir():
            return []
        for p in sorted(d.iterdir()):
            if p.suffix not in (NEW, OLD):
                continue
            folder = p.with_suffix("")
            if p.suffix == NEW and not folder.exists() and (p / "meta.json").is_file():
                p.rename(folder)                 # cut short between the two renames, but the new run is complete
                log.info("finished the save of %s that was cut short", folder)
            else:
                shutil.rmtree(p, ignore_errors=True)
        return self._saved_folders(zone)

    # ---- saving
    def save(self, zone: str, stitcher: Stitcher, instance: Instance, came_from: str | None,
             exits: set[str] = frozenset(), min_floor_frames: int = 1) -> SaveResult | None:
        """Save the stitcher's map as the run of this instance. If the same instance is saved again, its run gets
        overwritten when the new map is bigger. exits: zone_keys of the zone's gates (used to find the entrance).

        A zone with several floors (Temple of Kopec) gets one map per floor with at least min_floor_frames frames.
        map.png is the first floor (where you came in), the others are map_floor2.png and so on in the order you
        reached them, and floors.png shows them stacked with the first at the bottom."""
        floors = stitcher.floors(min_floor_frames)
        first = floors[0] if floors else None
        m = first.image if first else np.zeros((0, 0), np.uint8)
        map_px = int((m > 0).sum())
        if m.size == 0 or map_px < MIN_MAP_PIXELS:
            log.warning("run in %r not saved: only %d stable map pixels (crop %s, bbox %s, %d frames, "
                        "hits max %d, seen max %d)", zone, map_px, m.shape, stitcher.bbox(), stitcher.frames,
                        int(stitcher.hits.max()), int(stitcher.seen.max()))
            self._save_unsaved_picture(zone, stitcher)
            return None
        run_id = instance.id
        self._run_folders(zone)
        folder = self._run_folder(zone, run_id)
        if folder.with_name(folder.name + OLD).exists():   # not removed, a file in it is open (an image viewer, say)
            raise PermissionError(f"{folder.name}{OLD} could not be removed - run not saved")
        old = imread(folder / "map.png") if (folder / "map.png").exists() else None
        old_px = int((old > 0).any(axis=2).sum()) if old is not None else 0
        if old_px > map_px:
            # the same instance was filmed again (e.g. the capture was restarted inside the zone), so keep the bigger
            # map
            log.info("run in %r not saved: %s already holds a bigger map of this instance (%d vs %d px)",
                     zone, folder.name, old_px, map_px)
            return None
        floor_labels = without_screen_fixed([f.labels for f in floors])
        labels = [lb.to_json() for lb in floor_labels[0]]
        route = first.path
        start = list(route[0]) if route else None
        other_floors = [{"map": f"map_floor{i}.png", "frames": f.frames,
                         "labels": [lb.to_json() for lb in f_labels],
                         "path": thinned(f.path), "path_t": thinned(f.path_t)}
                        for i, (f, f_labels) in enumerate(zip(floors[1:], floor_labels[1:], strict=True), 2)]
        meta = {
            "version": META_VERSION,
            "zone": zone,
            "instance": run_id,
            "zone_code": instance.code,
            "seed": instance.seed,
            "scale": stitcher.scale,
            "screen": list(stitcher.screen) if stitcher.screen else None,   # maps of different sizes can't be compared
            "frames": first.frames,
            "came_from": came_from,
            "first_point": start,                 # where the map began (a few seconds after you came in)
            "entrance": entrance_point(start, labels, came_from, exits),   # where the route and labels start from
            "labels": labels,
            "path": thinned(route),
            "path_t": thinned(first.path_t),       # tick of each path point (see Stitcher.path_t); gaps are lost time
            "other_floors": other_floors,         # floors 2, 3, ...: each with its own map, labels and route
        }
        # The whole run (maps, meta.json, pictures) gets written into <id>.new and then swapped in, so the maps and
        # meta.json always belong together. A crash leaves either the earlier run or the new one (see _run_folders).
        new = folder.with_name(folder.name + NEW)
        shutil.rmtree(new, ignore_errors=True)
        new.mkdir(parents=True)
        for name, img in [("map.png", m), *((f"map_floor{i}.png", f.image) for i, f in enumerate(floors[1:], 2))]:
            if not imwrite(new / name, img):
                raise OSError(f"{folder / name}: the map could not be encoded - run not saved")
        write_json_atomic(new / "meta.json", meta)
        try:
            render_run_images(new, m, meta, path=route, times=first.path_t)
            for i, (f, f_meta) in enumerate(zip(floors[1:], other_floors, strict=True), 2):
                render_run_images(new, f.image, f_meta, path=f.path, times=f.path_t, suffix=f"_floor{i}")
            if len(floors) > 1:
                render_floors(new, [f.image for f in floors])
        except (cv2.error, OSError, ValueError) as e:     # these pictures are just for you, so they never stop the save
            log.warning("could not draw the pictures of %s (%s)", folder, e)
        if folder.exists():
            folder.rename(folder.with_name(folder.name + OLD))
        new.rename(folder)
        shutil.rmtree(folder.with_name(folder.name + OLD), ignore_errors=True)
        if old is not None:
            self._forget_layout(zone, run_id)     # the bigger map gets sorted in from scratch, not by its old grouping
        if zone in self._cache:                   # the zone's loaded runs, with this one (re)loaded
            loaded = [r for r in self._cache[zone] if r.id != run_id]
            run = self._load_run(folder)
            self._cache[zone] = sorted([*loaded, *([run] if run else [])], key=lambda r: r.folder)
        return SaveResult(folder, meta)

    def _run_folder(self, zone: str, run_id: str) -> Path:
        """The run's folder. A folder the first version named after its layout (A_<seed>) is renamed to the instance
        id."""
        folder = self.run_dir(zone, run_id)
        d = self.zone_dir(zone)
        if folder.exists() or not d.is_dir():
            return folder
        for old in d.iterdir():
            if not (old / "meta.json").is_file():
                continue
            try:
                found = read_json(old / "meta.json").get("instance") == run_id
            except (OSError, ValueError):
                continue
            if found:
                old.rename(folder)
                log.info("run folder %s renamed to %s", old.name, folder.name)
                return folder
        return folder

    def _save_unsaved_picture(self, zone: str, stitcher: Stitcher) -> None:
        """What a map that was too small to save looked like: the raw outline hits (left) and the voted map (right)."""
        bb = stitcher.bbox()
        if bb is None or self.diagnostics_dir is None:
            return
        try:
            bx0, by0, bx1, by1 = bb
            h = stitcher.hits[by0:by1, bx0:bx1].astype(np.float32)
            raw = (255 * h / max(1.0, float(h.max()))).astype(np.uint8)
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
            name = f"unsaved_{safe_name(zone)}_{time.strftime('%Y%m%d_%H%M%S')}.png"
            imwrite(self.diagnostics_dir / name, np.hstack([raw, stitcher.map_window(bx0, by0, bx1, by1)]))
        except (cv2.error, OSError, ValueError) as e:
            log.warning("could not write the unsaved-map picture (%s)", e)

    def _forget_layout(self, zone: str, run_id: str) -> None:
        """Drop a run from layouts.json so the next layouts() compares its (new) map again. The other runs keep their
        groups, and a layout that two runs were linked into through this one stays joined."""
        f = self.layouts_path(zone)
        try:
            known = read_json(f, default={})
            if run_id in known:
                del known[run_id]
                write_json_atomic(f, known)
                log.info("run %s of %r saved again with a bigger map - it is sorted into a layout again", run_id, zone)
        except (OSError, ValueError) as e:
            log.warning("could not update %s (%s)", f, e)

    # ---- loading and grouping
    def runs(self, zone: str) -> list[SavedRun]:
        if zone in self._cache:
            self._cache[zone] = self._cache.pop(zone)          # most recently used last
            return self._cache[zone]
        out = [r for r in map(self._load_run, self._run_folders(zone)) if r is not None]
        self._cache[zone] = out
        while len(self._cache) > max(1, self.keep_zones):
            self._cache.pop(next(iter(self._cache)))
        return out

    def _load_run(self, folder: Path) -> SavedRun | None:
        if not (folder / "meta.json").exists():
            return None
        img = imread(folder / "map.png")
        if img is None:
            log.warning("saved run %s skipped: map.png unreadable", folder)
            return None
        try:
            meta = read_json(folder / "meta.json")
        except (OSError, ValueError) as e:
            log.warning("saved run %s skipped: meta.json unreadable (%s)", folder, e)
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return SavedRun(folder, meta.get("instance") or folder.name, meta,
                        MapVariants(denoise_map(gray, path=meta.get("path"))))

    def layouts(self, zone: str) -> dict[str, dict]:
        """Sort the zone's runs into layouts (see layouts.group_runs) and keep the result in layouts.json."""
        f = self.layouts_path(zone)
        try:
            known = read_json(f, default={})
        except (OSError, ValueError) as e:
            log.warning("layouts.json of %r unreadable (%s) - regrouping all runs", zone, e)
            known = {}
        runs = self.runs(zone)
        screens = {r.id: tuple(r.meta["screen"]) for r in runs if r.meta.get("screen")}
        out, changed = group_runs(zone, [(r.id, r.maps) for r in runs], known, self.thresholds, screens)
        if changed:
            try:
                write_json_atomic(f, out)
            except OSError as e:
                log.warning("could not write %s (%s)", f, e)
        return out
