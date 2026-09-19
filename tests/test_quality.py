from __future__ import annotations

import numpy as np

from handrom.data_models import HandSide, QualityLevel
from handrom.quality import assess_image_quality, repeatability_quality


def _landmarks() -> np.ndarray:
    x = np.linspace(0.30, 0.70, 21)
    y = np.linspace(0.25, 0.75, 21)
    return np.column_stack((x, y, np.zeros(21)))


def test_good_textured_image_passes_major_checks() -> None:
    checker = (np.indices((600, 800)).sum(axis=0) % 2 * 180 + 35).astype(np.uint8)
    rgb = np.repeat(checker[:, :, None], 3, axis=2)
    result = assess_image_quality(
        rgb,
        _landmarks(),
        expected_side=HandSide.RIGHT,
        detected_side=HandSide.RIGHT,
        handedness_confidence=0.95,
    )
    assert result.level is QualityLevel.HIGH
    assert result.checks["sharpness"]
    assert result.checks["framing"]


def test_blur_and_brightness_are_warnings() -> None:
    rgb = np.full((600, 800, 3), 20, dtype=np.uint8)
    result = assess_image_quality(
        rgb,
        _landmarks(),
        expected_side=HandSide.RIGHT,
        detected_side=HandSide.RIGHT,
        handedness_confidence=0.95,
    )
    assert result.level is QualityLevel.MEDIUM
    assert "blur_detected" in result.warnings
    assert "image_too_dark" in result.warnings


def test_mirrored_handedness_validation() -> None:
    rgb = np.full((600, 800, 3), 128, dtype=np.uint8)
    result = assess_image_quality(
        rgb,
        _landmarks(),
        expected_side=HandSide.LEFT,
        detected_side=HandSide.RIGHT,
        handedness_confidence=0.95,
        mirrored=True,
    )
    assert result.checks["expected_hand_side"]
    assert "hand_side_mismatch" not in result.warnings


def test_clipped_key_landmark_is_low() -> None:
    landmarks = _landmarks()
    landmarks[0, 0] = 0.001
    result = assess_image_quality(
        np.full((600, 800, 3), 128, dtype=np.uint8),
        landmarks,
        expected_side=HandSide.RIGHT,
        detected_side=HandSide.RIGHT,
        handedness_confidence=0.95,
    )
    assert result.level is QualityLevel.LOW
    assert "wrist_or_fingertips_near_edge" in result.warnings


def test_target_finger_framing_ignores_other_fingertips() -> None:
    landmarks = _landmarks()
    landmarks[[0, 5, 6, 7, 8], :2] = [
        (0.30, 0.70),
        (0.38, 0.55),
        (0.45, 0.45),
        (0.52, 0.35),
        (0.60, 0.25),
    ]
    landmarks[20, 0] = 0.001
    result = assess_image_quality(
        np.full((600, 800, 3), 128, dtype=np.uint8),
        landmarks,
        expected_side=HandSide.RIGHT,
        detected_side=HandSide.RIGHT,
        handedness_confidence=0.95,
        target_finger="index",
    )
    assert result.level is QualityLevel.MEDIUM
    assert result.checks["target_chain_in_frame"]
    assert "wrist_or_fingertips_near_edge" not in result.warnings


def test_repeatability_quality_tiers() -> None:
    assert repeatability_quality(0, None) is QualityLevel.LOW
    assert repeatability_quality(1, 0.0) is QualityLevel.MEDIUM
    assert repeatability_quality(2, 5.0) is QualityLevel.HIGH
    assert repeatability_quality(2, 7.0) is QualityLevel.MEDIUM
    assert repeatability_quality(2, 11.0) is QualityLevel.LOW
