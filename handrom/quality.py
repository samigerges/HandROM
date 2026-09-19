"""Photograph and repeatability quality checks."""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import ArrayLike

from handrom.config import (
    BLUR_LAPLACIAN_THRESHOLD,
    BRIGHTNESS_BRIGHT_THRESHOLD,
    BRIGHTNESS_DARK_THRESHOLD,
    CONSISTENCY_HIGH_MAX_STD_DEG,
    CONSISTENCY_MEDIUM_MAX_STD_DEG,
    HANDEDNESS_CONFIDENCE_THRESHOLD,
    LANDMARK_EDGE_MARGIN,
    MIN_HAND_SPAN_RATIO,
)
from handrom.data_models import HandSide, QualityAssessment, QualityLevel
from handrom.landmark_mapping import FINGER_LANDMARKS


def assess_image_quality(
    rgb: np.ndarray,
    normalized_landmarks: ArrayLike | None,
    *,
    expected_side: HandSide,
    detected_side: HandSide,
    handedness_confidence: float,
    mirrored: bool = False,
    target_finger: str | None = None,
) -> QualityAssessment:
    """Evaluate objective image/detection checks without claiming occlusion detection."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    checks: dict[str, bool] = {
        "resolution": rgb.shape[1] >= 640 and rgb.shape[0] >= 480,
        "sharpness": blur_variance >= BLUR_LAPLACIAN_THRESHOLD,
        "brightness": BRIGHTNESS_DARK_THRESHOLD <= brightness <= BRIGHTNESS_BRIGHT_THRESHOLD,
        "handedness_confidence": handedness_confidence >= HANDEDNESS_CONFIDENCE_THRESHOLD,
    }
    warnings: list[str] = []

    effective_side = detected_side
    if mirrored:
        effective_side = HandSide.LEFT if detected_side is HandSide.RIGHT else HandSide.RIGHT
    checks["expected_hand_side"] = effective_side is expected_side

    if not checks["sharpness"]:
        warnings.append("blur_detected")
    if brightness < BRIGHTNESS_DARK_THRESHOLD:
        warnings.append("image_too_dark")
    elif brightness > BRIGHTNESS_BRIGHT_THRESHOLD:
        warnings.append("image_too_bright")
    if not checks["handedness_confidence"]:
        warnings.append("low_handedness_confidence")
    if not checks["expected_hand_side"]:
        warnings.append("hand_side_mismatch")

    if normalized_landmarks is None:
        checks.update({"landmarks_complete": False, "framing": False, "hand_size": False})
        warnings.append("missing_landmarks")
    else:
        points = np.asarray(normalized_landmarks, dtype=np.float64)
        finite = points.shape == (21, 3) and np.all(np.isfinite(points))
        checks["landmarks_complete"] = bool(finite)
        if finite:
            xy = points[:, :2]
            if target_finger is not None:
                if target_finger not in FINGER_LANDMARKS:
                    raise ValueError(f"Unsupported target finger: {target_finger}.")
                mapping = FINGER_LANDMARKS[target_finger]
                target_indices = [
                    0,
                    mapping["mcp"],
                    mapping["pip"],
                    mapping["dip"],
                    mapping["tip"],
                ]
                key_points = points[target_indices, :2]
                span_points = key_points
                in_bounds_points = key_points
            else:
                key_points = points[[0, 4, 8, 12, 16, 20], :2]
                span_points = xy
                in_bounds_points = xy
            in_bounds = bool(np.all((in_bounds_points >= 0.0) & (in_bounds_points <= 1.0)))
            away_from_edges = bool(
                np.all(
                    (key_points >= LANDMARK_EDGE_MARGIN) & (key_points <= 1 - LANDMARK_EDGE_MARGIN)
                )
            )
            span = float(max(np.ptp(span_points[:, 0]), np.ptp(span_points[:, 1])))
            checks["framing"] = in_bounds and away_from_edges
            if target_finger is not None:
                checks["target_chain_in_frame"] = checks["framing"]
            checks["hand_size"] = span >= MIN_HAND_SPAN_RATIO
            if not checks["framing"]:
                warnings.append("wrist_or_fingertips_near_edge")
            if not checks["hand_size"]:
                warnings.append("hand_too_small")
        else:
            checks.update({"framing": False, "hand_size": False})
            warnings.append("invalid_landmarks")

    major_checks = (
        checks.get("resolution", False),
        checks.get("landmarks_complete", False),
        checks.get("framing", False),
        checks.get("hand_size", False),
    )
    level = (
        QualityLevel.LOW
        if not all(major_checks)
        else (QualityLevel.MEDIUM if warnings else QualityLevel.HIGH)
    )
    return QualityAssessment(
        level=level,
        warnings=warnings,
        checks=checks,
        metrics={"blur_variance": blur_variance, "mean_brightness": brightness},
    )


def repeatability_quality(valid_image_count: int, max_std: float | None) -> QualityLevel:
    if valid_image_count <= 0 or max_std is None or not math.isfinite(max_std):
        return QualityLevel.LOW
    if valid_image_count == 1:
        return QualityLevel.MEDIUM
    if max_std <= CONSISTENCY_HIGH_MAX_STD_DEG:
        return QualityLevel.HIGH
    if max_std <= CONSISTENCY_MEDIUM_MAX_STD_DEG:
        return QualityLevel.MEDIUM
    return QualityLevel.LOW
