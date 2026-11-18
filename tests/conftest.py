import numpy as np
import pytest


@pytest.fixture
def red_disc():
    """200x300 skin-toned RGB image with a dark-red disc of radius 40 at the centre."""
    h, w, r = 200, 300, 40
    img = np.full((h, w, 3), (210, 170, 150), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    disc = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r * r
    img[disc] = (180, 40, 40)
    return img, disc.astype(np.float32)
