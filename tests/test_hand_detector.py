from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from handrom.data_models import HandSide
from handrom.hand_detector import HandDetectionError, detect_hand


class FakeLandmarker:
    def __init__(self, result: object) -> None:
        self.result = result

    def detect(self, _image: object) -> object:
        return self.result


def landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> object:
    return SimpleNamespace(x=x, y=y, z=z)


def category(name: str = "Right", score: float = 0.95) -> object:
    return SimpleNamespace(category_name=name, score=score)


def result(hand_count: int) -> object:
    hands = [[landmark()] * 21 for _ in range(hand_count)]
    return SimpleNamespace(
        hand_landmarks=hands,
        hand_world_landmarks=[[landmark()] * 21 for _ in range(hand_count)],
        handedness=[[category()] for _ in range(hand_count)],
    )


def test_no_hand_rejected() -> None:
    with pytest.raises(HandDetectionError) as exc:
        detect_hand(FakeLandmarker(result(0)), np.zeros((480, 640, 3), dtype=np.uint8))
    assert exc.value.code == "no_hand_detected"


def test_multiple_hands_rejected() -> None:
    with pytest.raises(HandDetectionError) as exc:
        detect_hand(FakeLandmarker(result(2)), np.zeros((480, 640, 3), dtype=np.uint8))
    assert exc.value.code == "multiple_hands_detected"


def test_single_hand_returns_world_landmarks_and_side() -> None:
    detected = detect_hand(FakeLandmarker(result(1)), np.zeros((480, 640, 3), dtype=np.uint8))
    assert detected.detected_side is HandSide.RIGHT
    assert detected.confidence == pytest.approx(0.95)
    assert detected.normalized_landmarks.shape == (21, 3)
    assert detected.world_landmarks.shape == (21, 3)


def test_missing_world_landmarks_rejected() -> None:
    fake = result(1)
    fake.hand_world_landmarks = []
    with pytest.raises(HandDetectionError) as exc:
        detect_hand(FakeLandmarker(fake), np.zeros((480, 640, 3), dtype=np.uint8))
    assert exc.value.code == "missing_world_landmarks"
