import cv2
import numpy as np

from poe2layout.outline import denoise_map


def test_denoise_keeps_what_you_walked_past():
    m = np.zeros((600, 900), np.uint8)
    cv2.rectangle(m, (50, 50), (400, 400), 255, 3)          # the main body of the map
    cv2.circle(m, (470, 220), 6, 255, 2)                     # a small pillar next to the route
    cv2.rectangle(m, (650, 450), (850, 580), 255, 3)        # a big scrap far from the route
    m[300:303, 300:303] = 255                                # a speck
    body, pillar, scrap, speck = (50, 200), (220, 464), (450, 700), (301, 301)

    out = denoise_map(m)                                     # no route, so size decides
    assert out[body] and out[scrap] and not out[pillar] and not out[speck]
    out = denoise_map(m, path=[(100, 100), (380, 220), (480, 230)])     # with a route, nearness decides
    assert out[body] and out[pillar] and not out[scrap] and not out[speck]
    assert set(np.unique(out)) == {0, 255}
