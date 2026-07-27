import numpy as np
from src.features.eye import eye_aspect_ratio
from src.features.mouth import mouth_aspect_ratio


def test_eye_aspect_ratio_square_eye():
    # A perfectly "open" symmetric eye shape should give a ratio > 0
    eye = np.array([
        [0, 0], [1, -2], [2, -2], [3, 0], [2, 2], [1, 2]
    ], dtype=float)
    ratio = eye_aspect_ratio(eye)
    assert ratio > 0


def test_mouth_aspect_ratio_closed_mouth():
    # Nearly flat mouth (closed) should give a low ratio
    mouth = np.array([[i, 0] for i in range(12)], dtype=float)
    ratio = mouth_aspect_ratio(mouth)
    assert ratio == 0
