import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("winrt.windows.graphics.imaging")
from poe2layout.ocr import WinOcr  # noqa: E402 - after the skip, since it needs winrt


class RecordingEngine:
    """Stands in for OcrEngine and remembers the size of the image it was given."""

    def __init__(self):
        self.sizes = []

    def recognize_async(self, bmp):
        self.sizes.append((bmp.pixel_width, bmp.pixel_height))

        async def result():
            return SimpleNamespace(lines=[])
        return result()


def test_upscale_stays_within_the_engine_limit():
    # a wide screen: upscaling 2x would go past the engine's max_image_dimension, which it refuses
    ocr = WinOcr.__new__(WinOcr)
    ocr._engine, ocr._max_side, ocr._loop = RecordingEngine(), 100, asyncio.new_event_loop()
    try:
        assert ocr.read_boxes(np.zeros((30, 80, 3), np.uint8), upscale=2.0) == []
        assert ocr.read_boxes(np.zeros((30, 40, 3), np.uint8), upscale=2.0) == []
    finally:
        ocr.close()
    assert ocr._engine.sizes == [(100, 38), (80, 60)]       # capped at 100 px; a small image still gets 2x
