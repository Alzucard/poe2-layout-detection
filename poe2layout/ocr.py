"""Windows OCR (Windows.Media.Ocr, built into Windows 10/11). We only use it on recorded videos, never on the live
game."""
from __future__ import annotations

import asyncio
from typing import NamedTuple

import cv2
import numpy as np


class TextBox(NamedTuple):
    text: str
    x: float
    y: float
    w: float
    h: float


class WinOcr:
    """Create it in the thread that uses it, since it owns an asyncio loop."""

    def __init__(self, language: str = "en-US"):
        # winrt is imported here so everything else works (and can be tested) without OCR installed
        from winrt.windows.globalization import Language
        from winrt.windows.media.ocr import OcrEngine

        engine = None
        try:
            lang = Language(language)
            if OcrEngine.is_language_supported(lang):
                engine = OcrEngine.try_create_from_language(lang)
        except OSError:
            engine = None
        if engine is None:
            engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            raise RuntimeError("Windows OCR is not available. Add English in Settings > Time & language > Language")
        self._engine = engine
        self._max_side = int(OcrEngine.max_image_dimension)
        self._loop = asyncio.new_event_loop()

    def close(self) -> None:
        self._loop.close()

    def read_boxes(self, img: np.ndarray, upscale: float = 2.0) -> list[TextBox]:
        """Lines of text with their boxes, in the coordinates of `img`. Small map labels need upscaling."""
        from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import DataWriter

        if img.size == 0:
            return []
        # the engine refuses images bigger than max_image_dimension (10000 px), and a 5120 px wide screen at 2x would be
        # 10240
        upscale = min(upscale, self._max_side / max(img.shape[:2]))
        big = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC) if upscale != 1 else img
        bgra = cv2.cvtColor(big, cv2.COLOR_BGR2BGRA)
        writer = DataWriter()
        writer.write_bytes(bgra.tobytes())
        bmp = SoftwareBitmap.create_copy_from_buffer(
            writer.detach_buffer(), BitmapPixelFormat.BGRA8, bgra.shape[1], bgra.shape[0],
            BitmapAlphaMode.PREMULTIPLIED)
        result = self._loop.run_until_complete(self._engine.recognize_async(bmp))
        out = []
        for line in result.lines:
            rects = [w.bounding_rect for w in line.words]
            if not rects or not line.text.strip():
                continue
            x0, y0 = min(r.x for r in rects), min(r.y for r in rects)
            x1, y1 = max(r.x + r.width for r in rects), max(r.y + r.height for r in rects)
            out.append(TextBox(line.text.strip(), x0 / upscale, y0 / upscale,
                               (x1 - x0) / upscale, (y1 - y0) / upscale))
        return out
