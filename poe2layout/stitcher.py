"""Stitches the Tab map frames of a zone into one map.

While the Tab map is open, every frame of the zone video shows the outline of the walkable area, centred on the
player. The map's zoom never changes, so two frames in a row only differ by a shift. The Stitcher works out that
shift and paints each frame into one growing map of the zone, and it records the route you walked on the way. If it
loses track of where you are for good, it puts the map so far aside as a piece and merges it back later, once it fits
somewhere.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from .files import imwrite
from .labels import LabelVotes, VotedLabel
from .logs import get_logger
from .matching import locate_ranked
from .outline import DEFAULT_MAP_AREA, MapArea, denoise_map, is_black, line_features, outline_mask, view_shift

log = get_logger("stitcher")

MAX_SNAPSHOTS = 50
VOTE_RATIO = 0.3               # a pixel counts as wall if it showed up in this share of the frames that saw it
PEAK_RADIUS = 24               # canvas px around a match peak that we treat as the same place

# Tracking (see _locate). Scores come from TM_CCORR_NORMED on blurred outlines, and the response is what
# cv2.phaseCorrelate returns.
MOTION_MIN_RESPONSE = 0.2      # below this we can't measure the motion between two frames
MOTION_TRUST_RESPONSE = 0.3    # above this we trust the motion even where there's no map to snap to yet
SNAP_PX = 6                    # how far (canvas px) we may move the motion estimate to snap it onto the map
NEARBY_MIN_SCORE = 0.5         # searching near the last position: the best match needs this score...
NEARBY_MIN_LEAD = 0.04         # ...and has to beat the best match elsewhere by this much
REATTACH_COARSE = 4            # the whole-map search runs at a quarter of the size...
REATTACH_MIN_LEAD = 0.05       # ...has to beat the best match elsewhere by this much, because some places look alike...
REATTACH_REFINE_PX = 8         # ...and is then refined at full size within this many canvas px

# Losing the position (see _add)
LOST_FOR_GOOD = 8              # frames without a position before we put the map so far aside as a piece
STABLE_FRAMES = 15             # a map with this many frames is stable: put aside when lost, never thrown away
START_MAX_ATTEMPTS = 12        # if a new map placed less than START_MIN_SHARE of this many frames, we throw it away...
START_MIN_SHARE = 0.5
START_GIVE_UP = 3              # ...or at once after losing this many frames per placed one (bad first frame)
MERGE_EVERY = 20               # how many placed frames we wait between attempts to merge old pieces back
GAP_TICKS = 10                 # a gap in the video (you were away) counts as this many frames without a position


class StepResult(StrEnum):
    """What Stitcher.add() did with a frame."""
    SKIPPED = "skipped"        # not enough outline: the Tab map is closed or it's a loading screen
    STARTED = "started"        # the first frame of a map
    TRACKED = "tracked"        # placed on the map
    LOST = "lost"              # couldn't find the position, so the frame wasn't placed
    RESTARTED = "restarted"    # the map's start was shaky, so we threw it away
    NEW_PIECE = "new piece"    # lost for good: the map so far was put aside and a new piece started
    FADED = "faded"            # the screen went black (stairs, or a teleport), so the map was put aside

    @property
    def placed(self) -> bool:
        """True when the frame was painted onto the map, which means its labels can be placed too."""
        return self in (StepResult.STARTED, StepResult.TRACKED, StepResult.NEW_PIECE)


class TrackMethod(StrEnum):
    """How we found the position of the last frame."""
    NONE = ""
    MOTION_SNAP = "motion+snap"        # motion from the previous frame, snapped onto the map
    MOTION_ONLY = "motion only"        # we trusted the motion, there was nothing on the map to snap to yet
    COASTING = "coasting"              # weak motion: we followed it, but didn't paint the frame
    NEARBY_SEARCH = "nearby search"    # found by searching around the last position
    REATTACH = "re-attach"             # found by searching the whole map
    LOST = "lost"


@dataclass(eq=False)          # compare by identity, so two pieces are never equal
class MapPiece:
    """Part of the map that was put aside after we lost track for good. Its coordinates are its own crop's."""
    image: np.ndarray                  # the piece's cleaned map
    hits: np.ndarray
    seen: np.ndarray
    frames: int
    path: list[tuple[int, int]]
    path_t: list[int]
    labels: list[VotedLabel]
    floor: int = 0                     # pieces put aside with no black screen in between belong to the same floor


def map_pixels(piece: MapPiece) -> int:
    """How much map a piece holds. Frames where you stood still don't add any."""
    return int((piece.image > 0).sum())


class Stitcher:
    """Collects the outline masks of the player-centred Tab map into one map.

    Everything is kept on a square canvas at `scale` (0.5 by default) of the screen size. Frame k covers
    canvas[oy:oy+h, ox:ox+w], where (ox, oy) is the offset, and the player stands at the frame's centre.
    """

    def __init__(self, scale: float = 0.5, canvas_size: int = 4096, max_shift: int = 260, min_lines: int = 300,
                 start_lines: int = 4000, map_area: MapArea = DEFAULT_MAP_AREA, snapshot_dir: Path | None = None):
        self.scale = scale
        self.canvas_size = canvas_size
        self.max_shift = max_shift      # the biggest shift between two frames in a row (canvas px)
        self.min_lines = min_lines      # with less outline (screen px) the Tab map is closed or it's a loading screen
        self.start_lines = start_lines  # we don't start a map from a frame with less outline than this (screen px)
        self.map_area = map_area
        self.snapshot_dir = snapshot_dir   # frame and outline get saved here when tracking is lost (None: off)
        self.hits = np.zeros((canvas_size, canvas_size), np.uint16)   # how many frames showed a line here
        self.seen = np.zeros((canvas_size, canvas_size), np.uint16)   # how many frames showed map near here
        self.offset: tuple[int, int] | None = None   # top-left corner of the last frame on the canvas
        self.path: list[tuple[int, int]] = []        # where the player was on the canvas, in the order walked
        self.path_t: list[int] = []                  # tick of each path point: one per frame, and GAP_TICKS for a gap
        self.frames = 0
        self.pieces: list[MapPiece] = []
        self.labels = LabelVotes()                   # in this piece's canvas coordinates, so they move along with it
        self.method = TrackMethod.NONE
        self.shifted = (0, 0)                        # how far _recentre has moved the canvas content in total
        self.screen: tuple[int, int] | None = None   # (width, height) of the frames; the thresholds assume one size
        self._tick = 0
        self._attempts = 0
        self._prev_features: np.ndarray | None = None
        self._synced = True              # previous frame was placed, so we can add the motion since then
        self._unsynced = 0
        self._bbox: tuple[int, int, int, int] | None = None   # the explored rectangle, which _paint keeps up to date
        self._near_kernel = np.ones((61, 61), np.uint8)       # the "revealed" area around lines we've seen
        self._hanning: np.ndarray | None = None
        self._lost_streak = 0
        self._snapshots = 0
        self._counts: dict[str, int] = {}
        self._calls = 0
        self._last_mask: np.ndarray | None = None
        self._last_lines = 0
        self._last_score = 0.0
        self._restart_info = (0, 0)
        self._floor = 0                  # floor of the current map; a black screen (FADED) starts the next one
        self._next_floor = 1
        self._floor_unsure = False       # back after a gap and not found yet, so you might be on another floor

    # ---- public
    def add(self, frame_bgr: np.ndarray, outline: np.ndarray | None = None) -> StepResult:
        """Add one frame of the zone video. Pass outline if you already computed outline_mask(frame_bgr, map_area)."""
        result = self._add(frame_bgr, outline)
        self._log(result, frame_bgr)
        return result

    def mark_gap(self) -> None:
        """Some frames are missing before the next one (you were away), so find the position again before we trust
        motion."""
        self._synced = False
        self._floor_unsure = True
        self._tick += GAP_TICKS          # we don't know the route across the gap

    def bbox(self) -> tuple[int, int, int, int] | None:
        """The explored rectangle on the canvas as (x0, y0, x1, y1), or None before the first frame."""
        return self._bbox

    def map_window(self, x0, y0, x1, y1, min_ratio: float = VOTE_RATIO, min_seen: int = 2) -> np.ndarray:
        """The voted map (uint8, 0 or 255) inside a canvas rectangle: the lines that showed up in enough of the
        frames that saw them."""
        hits, seen = self.hits[y0:y1, x0:x1], self.seen[y0:y1, x0:x1]
        m = (hits >= np.maximum(1, min_ratio * seen.astype(np.float32))) & (seen >= min_seen)
        return (m * 255).astype(np.uint8)

    def cropped_map(self, pad: int = 20) -> tuple[np.ndarray, tuple[int, int]]:
        """Returns the cleaned map cropped to the explored area, and the (x0, y0) canvas position of that crop."""
        if self._bbox is None:
            return np.zeros((0, 0), np.uint8), (0, 0)
        x0, y0, x1, y1 = self._bbox
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        path = [(x - x0, y - y0) for x, y in self.path]
        return denoise_map(self.map_window(x0, y0, x1 + pad, y1 + pad), path=path), (x0, y0)

    def try_merge(self, min_agree: float = 0.72, margin: float = 0.1, min_common: int = 800) -> int:
        """Put earlier map pieces back into the current map wherever they overlap convincingly. This is strict on
        purpose: a wrong merge ruins the map, while a missed one just gets retried later. Returns the number of
        merges."""
        if not self.pieces or self.offset is None:
            return 0
        merged = 0
        cur, (cx0, cy0) = self.cropped_map()
        for piece in list(self.pieces):
            if cur.size == 0:
                return merged
            piece_smaller = piece.image.size <= cur.size
            partial, full = (piece.image, cur) if piece_smaller else (cur, piece.image)
            ranked = locate_ranked(partial, full)
            if not ranked:
                continue
            ag, dx, dy = ranked[0]
            second = next((r[0] for r in ranked[1:] if abs(r[1] - dx) + abs(r[2] - dy) > 30), 0.0)   # elsewhere
            px, py = (cx0 + dx, cy0 + dy) if piece_smaller else (cx0 - dx, cy0 - dy)   # top-left of the piece's crop
            ph, pw = piece.image.shape
            if not self._on_canvas(px, py, pw, ph):
                continue
            # how much outline the two maps really share when placed like this
            here = self.map_window(px, py, px + pw, py + ph)
            common = int(((piece.image > 0) & (cv2.dilate(here, np.ones((5, 5), np.uint8)) > 0)).sum())
            if ag < min_agree or ag - second < margin or common < min_common:
                log.debug("piece not merged yet: agreement %.2f (next %.2f), shared outline %d px", ag, second, common)
                continue
            self._put_piece(piece, px, py)
            self._join_floor(piece.floor)
            merged += 1
            cur, (cx0, cy0) = self.cropped_map()
            log.info("map piece merged back (%d frames, agreement %.2f vs next %.2f, shared outline %d px); "
                     "%d pieces left", piece.frames, ag, second, common, len(self.pieces))
        return merged

    def finalize(self) -> int:
        """The visit is over. Carry on with the largest map, since the saved map should be the most complete one, not
        just whichever piece came last (or nothing at all after a black screen). Then try one last time to merge the
        other pieces into it. Returns the number of merges."""
        merged = self.try_merge()
        if self.pieces:
            biggest = max(self.pieces, key=map_pixels)
            if self.offset is None or map_pixels(biggest) > map_pixels(self._as_piece()):
                if self.frames:
                    self._archive_piece()
                self.reset()
                h, w = biggest.image.shape
                x0, y0 = (self.canvas_size - w) // 2, (self.canvas_size - h) // 2
                self._put_piece(biggest, x0, y0)
                self._floor = biggest.floor
                # we don't know where you last stood on this piece; when you come back to the instance, a search finds
                # it
                self.offset = (x0, y0)
                self._synced = False
                merged += self.try_merge()
                log.info("saving the largest map piece (%d frames); %d smaller pieces left out",
                         self.frames, len(self.pieces))
        return merged

    def reset(self) -> None:
        """Forget the current map. Pieces that were put aside are kept."""
        self.hits[:] = 0
        self.seen[:] = 0
        self.offset = None
        self.path = []
        self.path_t = []
        self.frames = 0
        self._attempts = 0
        self._prev_features = None
        self._synced = True
        self._unsynced = 0
        self.labels = LabelVotes()
        self._bbox = None

    # ---- logging
    def _log(self, result: StepResult, frame_bgr) -> None:
        self._calls += 1
        key = str(self.method) if result is StepResult.TRACKED else str(result)
        self._counts[key] = self._counts.get(key, 0) + 1
        if result is StepResult.STARTED:
            self._lost_streak = 0
            log.info("map started (outline %d px)", self._last_lines)
        elif result is StepResult.NEW_PIECE:
            self._lost_streak = 0
            log.warning("position lost for good: map piece put aside (%d frames), started a new piece "
                        "(%d pieces so far)", self.pieces[-1].frames, len(self.pieces))
        elif result is StepResult.RESTARTED:
            self._lost_streak = 0
            log.warning("unstable start: map thrown away (%d of %d frames placed), %s",
                        self._restart_info[0], self._restart_info[1],
                        "started again from this frame" if self.offset is not None else "waiting for a good frame")
        elif result is StepResult.LOST:
            self._lost_streak += 1
            if self._lost_streak == 1:
                log.info("tracking lost (best score %.2f, outline %d px)", self._last_score, self._last_lines)
            if self._lost_streak == 5 and self.snapshot_dir is not None and self._snapshots < MAX_SNAPSHOTS:
                self._snapshot(frame_bgr)
        elif result is StepResult.TRACKED:
            if self._lost_streak:
                log.info("tracking recovered after %d lost frames via %s (score %.2f)",
                         self._lost_streak, self.method, self._last_score)
            elif self.method in (TrackMethod.REATTACH, TrackMethod.NEARBY_SEARCH):
                log.info("position found by %s (score %.2f)", self.method, self._last_score)
            log.debug("frame %s score %.2f offset %s", self.method, self._last_score, self.offset)
            self._lost_streak = 0
        if self._calls % 60 == 0:
            bb = self._bbox
            size = f"{(bb[2] - bb[0]) / self.scale:.0f}x{(bb[3] - bb[1]) / self.scale:.0f}px" if bb else "-"
            log.info("stitching summary: %s | map %s | path %d points", self._counts, size, len(self.path))

    def _snapshot(self, frame_bgr) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if frame_bgr is not None:
            small = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            imwrite(self.snapshot_dir / f"{stamp}_frame.jpg", small)
        if self._last_mask is not None:
            imwrite(self.snapshot_dir / f"{stamp}_outline.png", self._last_mask)
        self._snapshots += 1
        log.info("snapshot saved: %s_*", self.snapshot_dir / stamp)

    # ---- one frame
    def _enough(self, mask: np.ndarray, screen_px: int) -> bool:
        return int((mask > 0).sum()) >= screen_px * self.scale * self.scale

    def _add(self, frame_bgr: np.ndarray, outline: np.ndarray | None) -> StepResult:
        self._last_score = 0.0
        self._tick += 1                    # every frame counts, so a gap in path_t is time without a position
        if self.screen is None:
            self.screen = (frame_bgr.shape[1], frame_bgr.shape[0])
        mask = outline_mask(frame_bgr, self.map_area) if outline is None else outline
        if self.scale != 1:
            mask = cv2.resize(mask, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            mask = ((mask > 60) * 255).astype(np.uint8)
        self._last_lines = int((mask > 0).sum() / (self.scale * self.scale))
        self._last_mask = mask
        if not self._enough(mask, self.min_lines):
            if self.offset is not None and is_black(frame_bgr, self.map_area):
                # This is another floor or a teleport. Searching the map for the next frame could glue another floor
                # onto this one (Temple of Kopec), so we start a new piece. If it turns out to be the same floor,
                # try_merge puts it back.
                log.info("screen went black: map piece put aside (%d frames)", self.frames)
                if self.frames >= STABLE_FRAMES:
                    self._archive_piece()
                self.reset()
                self._new_floor()
                return StepResult.FADED
            return StepResult.SKIPPED
        h, w = mask.shape
        shift = view_shift(frame_bgr)
        if self.offset is None:
            if not self._enough(mask, self.start_lines):
                return StepResult.SKIPPED
            self._start_piece(mask, shift)
            return StepResult.STARTED
        pos, score = self._locate(mask)
        self._last_score = float(score)
        self._attempts += 1
        if pos is not None:
            pos = self._fit_on_canvas(pos, w, h)
            if pos is None:
                self.method = TrackMethod.LOST
                self._synced = False
        if pos is None:
            self._unsynced = self._unsynced + 1 if not self._synced else 0
            # We've lost the position for a while now. Rather than staying stuck for the rest of the zone, put this
            # piece aside and start a new one from this frame.
            if (self._unsynced >= LOST_FOR_GOOD and self.frames >= STABLE_FRAMES
                    and self._enough(mask, self.start_lines)):
                self._archive_piece()
                self.reset()
                if self._floor_unsure:       # lost since a gap, so this may be another floor; a merge will tell
                    self._new_floor()
                self._start_piece(mask, shift)
                return StepResult.NEW_PIECE
            # If the first frames of a map don't hold together, it started on a bad frame: the old zone's map, the
            # loading screen right after a zone change, or effects. Throw it away and start again, from this frame if it
            # has enough outline. A start that lost START_GIVE_UP frames per placed frame is dropped right away. Waiting
            # the full 12 frames cost about 6 s and ~300 px of the entrance in Hunting Grounds, and 25 of the 38
            # restarts in the log had placed only 1-2 frames.
            lost_since = self._attempts - (self.frames - 1)
            if self.frames < STABLE_FRAMES and self.frames < self._attempts * START_MIN_SHARE and (
                    self._attempts >= START_MAX_ATTEMPTS or lost_since >= START_GIVE_UP * self.frames):
                self._restart_info = (self.frames, self._attempts)
                self.reset()
                if self._enough(mask, self.start_lines):
                    self._start_piece(mask, shift)
                return StepResult.RESTARTED
            return StepResult.LOST
        self.offset = pos
        self._unsynced = 0
        self._floor_unsure = False
        self._paint(mask, shift)
        if self.pieces and self.frames % MERGE_EVERY == 0:
            self.try_merge()
        return StepResult.TRACKED

    def _start_piece(self, mask: np.ndarray, shift: int = 0) -> None:
        """Start a map (or piece) from this frame, in the middle of the canvas."""
        h, w = mask.shape
        self.offset = ((self.canvas_size - w) // 2, (self.canvas_size - h) // 2)
        self._prev_features = line_features(mask)
        self._synced = True              # the next frame can use the motion from this one, even after a gap
        self._paint(mask, shift)

    def _on_canvas(self, x: int, y: int, w: int, h: int) -> bool:
        """True when the w x h rectangle with its top-left at (x, y) lies fully on the canvas."""
        return x >= 0 and y >= 0 and x + w <= self.canvas_size and y + h <= self.canvas_size

    def _fit_on_canvas(self, pos, w: int, h: int):
        """The frame would lie (partly) off the canvas, so move the whole map until the explored part plus this frame
        sit in the middle again. Returns None when the map is bigger than the canvas."""
        x, y = pos
        n = self.canvas_size
        if self._on_canvas(x, y, w, h):
            return pos
        bx0, by0, bx1, by1 = self._bbox or (x, y, x + w, y + h)
        ux0, uy0, ux1, uy1 = min(bx0, x), min(by0, y), max(bx1, x + w), max(by1, y + h)
        if ux1 - ux0 > n or uy1 - uy0 > n:
            log.warning("map is larger than the canvas - frame not placed")
            return None
        dx, dy = (n - (ux1 - ux0)) // 2 - ux0, (n - (uy1 - uy0)) // 2 - uy0
        self._recentre(dx, dy)
        log.info("map reached the canvas edge: moved it by (%d, %d)", dx, dy)
        return x + dx, y + dy

    def _recentre(self, dx: int, dy: int) -> None:
        """Move everything that lives in canvas coordinates by (dx, dy)."""
        if self._bbox is not None:
            x0, y0, x1, y1 = self._bbox
            for arr in (self.hits, self.seen):
                block = arr[y0:y1, x0:x1].copy()
                arr[y0:y1, x0:x1] = 0
                arr[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = block
            self._bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        self.path = [(x + dx, y + dy) for x, y in self.path]
        self.labels.shift(dx, dy)
        if self.offset is not None:
            self.offset = (self.offset[0] + dx, self.offset[1] + dy)
        self.shifted = (self.shifted[0] + dx, self.shifted[1] + dy)

    def _put_piece(self, piece: MapPiece, px: int, py: int) -> None:
        """Add a map piece to the canvas, with the top-left of its crop at (px, py)."""
        ph, pw = piece.image.shape
        self.hits[py:py + ph, px:px + pw] += piece.hits
        self.seen[py:py + ph, px:px + pw] += piece.seen
        # pieces can merge back in any order, so keep the path in the order walked (path[0] is where the visit began)
        moved = zip(piece.path_t or [0] * len(piece.path), [(x + px, y + py) for x, y in piece.path], strict=True)
        merged = sorted([*moved, *zip(self.path_t, self.path, strict=True)], key=lambda tp: tp[0])
        self.path_t, self.path = [t for t, _ in merged], [p for _, p in merged]
        self.frames += piece.frames
        self.labels.absorb(piece.labels, px, py)
        self._grow_bbox(px, py, px + pw, py + ph)
        self.pieces = [p for p in self.pieces if p is not piece]

    def floors(self, min_frames: int) -> list[MapPiece]:
        """One map per floor of this instance (see FADED), in the order the floors were reached. For each floor
        that's the biggest of its maps with at least min_frames frames, out of the current map and the pieces that
        never merged into it. The floor's other pieces (where we lost the position for good) are left out."""
        biggest: dict[int, MapPiece] = {}
        for p in [*self.pieces, *([self._as_piece()] if self.frames else [])]:
            if p.frames >= min_frames and (p.floor not in biggest or map_pixels(p) > map_pixels(biggest[p.floor])):
                biggest[p.floor] = p
        return sorted(biggest.values(), key=lambda p: p.path_t[0] if p.path_t else 0)

    def _join_floor(self, floor: int) -> None:
        """A piece of this floor was merged into the current map, so both are one floor. The lower number stays."""
        keep, drop = min(self._floor, floor), max(self._floor, floor)
        for p in self.pieces:
            if p.floor == drop:
                p.floor = keep
        self._floor = keep

    def _new_floor(self) -> None:
        """The map from here on may be another floor, so it gets the next floor number."""
        self._floor, self._next_floor = self._next_floor, self._next_floor + 1

    def _as_piece(self) -> MapPiece:
        m, (x0, y0) = self.cropped_map()
        h, w = m.shape
        return MapPiece(
            image=m, frames=self.frames,
            hits=self.hits[y0:y0 + h, x0:x0 + w].copy(), seen=self.seen[y0:y0 + h, x0:x0 + w].copy(),
            path=[(x - x0, y - y0) for x, y in self.path], path_t=list(self.path_t),
            labels=self.labels.shifted_copy(-x0, -y0), floor=self._floor)

    def _archive_piece(self) -> None:
        self.pieces.append(self._as_piece())

    def _grow_bbox(self, x0, y0, x1, y1) -> None:
        n = self.canvas_size
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(n, x1), min(n, y1)
        b = self._bbox
        self._bbox = (x0, y0, x1, y1) if b is None else (min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1))

    def _paint(self, mask: np.ndarray, shift: int = 0) -> None:
        """shift: screen px from the screen centre to the player, sideways, when a side panel is open."""
        ox, oy = self.offset
        h, w = mask.shape
        self.hits[oy:oy + h, ox:ox + w] += (mask > 0)
        # Only count "seen" near map this frame actually shows. The Tab map reveals walls as you explore, so a spot
        # that's on screen but not revealed yet must not count as "seen, no wall". Otherwise long visits would vote out
        # walls that got revealed late. Noise blobs near shown lines still get voted out.
        self.seen[oy:oy + h, ox:ox + w] += cv2.dilate((mask > 0).astype(np.uint8), self._near_kernel) > 0
        self._grow_bbox(ox, oy, ox + w, oy + h)
        self.path.append((ox + w // 2 + round(shift * self.scale), oy + h // 2))
        self.path_t.append(self._tick)
        self.frames += 1

    # ---- finding a frame's position
    def _outline_probability(self, x0, y0, x1, y1) -> np.ndarray:
        """Blurred 0..1 probability that a canvas pixel in this rectangle is outline."""
        hits = self.hits[y0:y1, x0:x1].astype(np.float32)
        seen = self.seen[y0:y1, x0:x1].astype(np.float32)
        ref = np.where(seen > 0, hits / np.maximum(seen, 1), 0)
        return cv2.GaussianBlur(ref, (0, 0), 3)

    def _match_in(self, features, x0, y0, x1, y1, coarse: int = 1):
        """Best spot for the frame's features inside a canvas rectangle. Returns (position, score, best score
        elsewhere)."""
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.canvas_size, x1), min(self.canvas_size, y1)
        ref = self._outline_probability(x0, y0, x1, y1)
        h, w = features.shape
        if ref.shape[0] < h or ref.shape[1] < w:
            return None, 0.0, 0.0
        if coarse > 1:
            ref = cv2.resize(ref, None, fx=1 / coarse, fy=1 / coarse, interpolation=cv2.INTER_AREA)
            features = cv2.resize(features, None, fx=1 / coarse, fy=1 / coarse, interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(ref, features, cv2.TM_CCORR_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        # the best score somewhere else (away from the peak); places that look alike make a match ambiguous
        cv2.circle(res, loc, max(3, PEAK_RADIUS // coarse), 0, -1)
        second = float(res.max()) if res.size else 0.0
        return (x0 + loc[0] * coarse, y0 + loc[1] * coarse), float(score), second

    def _lines_agree(self, mask: np.ndarray, pos, min_agree: float = 0.65, min_px: int = 200) -> bool:
        """A position found by searching needs more than a good blurry correlation. Where the frame overlaps map we
        already know, its lines have to line up with the known ones, both ways. A frame that only shows new ground
        can't be checked, so a search must not place it."""
        ox, oy = pos
        h, w = mask.shape
        if not self._on_canvas(ox, oy, w, h):
            return False
        known = self.seen[oy:oy + h, ox:ox + w] >= 2
        ref = self.map_window(ox, oy, ox + w, oy + h) > 0
        mine = (mask > 0) & known
        if int(mine.sum()) < min_px or int(ref.sum()) < min_px:
            return False
        k = np.ones((5, 5), np.uint8)
        a = (mine & (cv2.dilate(ref.astype(np.uint8), k) > 0)).sum() / mine.sum()
        near = cv2.dilate((mask > 0).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        ref_near = ref & near
        if int(ref_near.sum()) < min_px:
            return False
        b = (ref_near & (cv2.dilate((mask > 0).astype(np.uint8), k) > 0)).sum() / ref_near.sum()
        return min(a, b) >= min_agree

    def _lost(self, score: float):
            """No position for this frame: motion has to be confirmed against the map again before we trust it."""
            self.method = TrackMethod.LOST
            self._synced = False
            return None, score

    def _locate(self, mask: np.ndarray, accept: float = 0.35, reattach: float = 0.6):
        """Find where this frame sits on the canvas.
        1) Take the motion since the last good frame (phase correlation, since the Tab map only shifts a few px per
           frame), then snap it onto the map built so far within a few px so it doesn't drift.
        2) If the motion can't be measured (the Tab map was closed, or a big jump), search around the last position,
           then the whole explored map. Only a very confident match is accepted, because placing a frame in the wrong
           spot wrecks the map."""
        ox, oy = self.offset
        h, w = mask.shape
        features = line_features(mask)
        prev = self._prev_features
        # Motion is measured against the previous frame, so we can only add it to the position if that frame was placed
        # too. After a lost frame the two are out of step (you kept moving), so first we have to confirm the position
        # against the map again.
        if prev is not None and prev.shape == features.shape and self._synced:
            if self._hanning is None or self._hanning.shape != features.shape:
                self._hanning = cv2.createHanningWindow(features.shape[::-1], cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(prev, features, self._hanning)
            if response >= MOTION_MIN_RESPONSE and abs(dx) < self.max_shift and abs(dy) < self.max_shift:
                px, py = int(round(ox - dx)), int(round(oy - dy))
                c = SNAP_PX
                pos, score, _ = self._match_in(features, px - c, py - c, px + w + c, py + h + c)
                if pos is not None and score >= accept:
                    self._prev_features = features
                    self.method = TrackMethod.MOTION_SNAP
                    self._synced = True
                    return pos, score
                if response >= MOTION_TRUST_RESPONSE:
                    # We trust the motion, but there's nothing on the map to snap to yet (you're walking into a new
                    # area), so we go with the motion estimate.
                    self._prev_features = features
                    self.method = TrackMethod.MOTION_ONLY
                    self._synced = True
                    return (px, py), response
                # We can measure the motion, but it's too weak to draw this frame into the map. Don't draw it, but keep
                # following the movement (dead reckoning). That keeps the position and the reference frame in step, so
                # the next good frame can be placed by motion again without a risky search.
                self.offset = (px, py)
                self._prev_features = features
                self.method = TrackMethod.COASTING
                return None, response
        # Always compare the next frame with this one. An outdated reference frame makes every later motion estimate
        # worse, until tracking never recovers.
        self._prev_features = features
        r = self.max_shift
        pos, score, second = self._match_in(features, ox - r, oy - r, ox + w + r, oy + h + r)
        if (pos is not None and score >= max(accept, NEARBY_MIN_SCORE) and score - second >= NEARBY_MIN_LEAD
                and self._lines_agree(mask, pos)):
            self.method = TrackMethod.NEARBY_SEARCH
            self._synced = True
            return pos, score
        if self._bbox is None:
            self._synced = False
            return None, score
        bx0, by0, bx1, by1 = self._bbox
        pos, score, second = self._match_in(features, bx0 - w // 2, by0 - h // 2, bx1 + w // 2, by1 + h // 2,
                                            coarse=REATTACH_COARSE)
        if pos is None or score - second < REATTACH_MIN_LEAD:            # two places look alike, so don't guess
            return self._lost(score)
        c = REATTACH_REFINE_PX                                      # refine the coarse hit
        pos, score, _ = self._match_in(features, pos[0] - c, pos[1] - c, pos[0] + w + c, pos[1] + h + c)
        if pos is None or score < reattach or not self._lines_agree(mask, pos):
            return self._lost(score)
        self.method = TrackMethod.REATTACH
        self._synced = True
        return pos, score
