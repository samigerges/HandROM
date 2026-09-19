"""Transparent median aggregation across included photographs."""

from __future__ import annotations

import math

import numpy as np

from handrom.config import ANGLE_NOISE_FLOOR_DEG
from handrom.data_models import (
    AggregateStats,
    AggregationResult,
    ImageAnalysis,
    PoseType,
    QualityLevel,
)
from handrom.landmark_mapping import FINGERS, JOINTS
from handrom.quality import repeatability_quality


def aggregate_measurements(
    analyses: list[ImageAnalysis],
    pose: PoseType,
) -> AggregationResult:
    included = [item for item in analyses if item.pose is pose and item.valid and item.included]
    explicit_targets = {
        item.target_finger
        for item in analyses
        if item.pose is pose and item.target_finger is not None
    }
    target_fingers = tuple(finger for finger in FINGERS if finger in explicit_targets)
    if not target_fingers:
        target_fingers = FINGERS
    stats: dict[str, dict[str, AggregateStats | None]] = {
        finger: {joint: None for joint in JOINTS} for finger in FINGERS
    }
    standard_deviations: list[float] = []
    finger_image_counts: dict[str, int] = {}
    finger_quality: dict[str, QualityLevel] = {}
    finger_max_std: dict[str, float | None] = {}
    warnings: list[str] = []
    missing = False

    for finger in target_fingers:
        finger_items = [
            item for item in included if item.target_finger is None or item.target_finger == finger
        ]
        finger_image_counts[finger] = len(finger_items)
        finger_deviations: list[float] = []
        finger_missing = False
        for joint in JOINTS:
            values = [item.angles.get(finger, {}).get(joint) for item in finger_items]
            clean = [float(value) for value in values if value is not None and math.isfinite(value)]
            if not clean:
                missing = True
                finger_missing = True
                continue
            array = np.asarray(clean, dtype=np.float64)
            median = float(np.median(array))
            if pose is PoseType.EXTENSION and median < ANGLE_NOISE_FLOOR_DEG:
                median = 0.0
            deviation = float(np.std(array, ddof=0))
            standard_deviations.append(deviation)
            finger_deviations.append(deviation)
            stats[finger][joint] = AggregateStats(
                median=median,
                mean=float(np.mean(array)),
                std=deviation,
                minimum=float(np.min(array)),
                maximum=float(np.max(array)),
                count=len(clean),
            )

        current_max_std = max(finger_deviations, default=None)
        finger_max_std[finger] = current_max_std
        current_quality = repeatability_quality(len(finger_items), current_max_std)
        if finger_missing or any(item.quality.level is QualityLevel.LOW for item in finger_items):
            current_quality = QualityLevel.LOW
        finger_quality[finger] = current_quality
        if len(finger_items) == 1:
            warnings.append(f"{finger}:single_image_no_repeatability")
        if current_max_std is not None and current_max_std > 5.0:
            warnings.append(f"{finger}:inconsistent_measurements")

    max_std = max(standard_deviations, default=None)
    if any(level is QualityLevel.LOW for level in finger_quality.values()):
        quality = QualityLevel.LOW
    elif any(level is QualityLevel.MEDIUM for level in finger_quality.values()):
        quality = QualityLevel.MEDIUM
    else:
        quality = QualityLevel.HIGH
    if len(included) == 1:
        warnings.append("single_image_no_repeatability")
    if max_std is not None and max_std > 5.0:
        warnings.append("inconsistent_measurements")
    if missing:
        quality = QualityLevel.LOW
        warnings.append("missing_required_measurements")
    if any(item.quality.level is QualityLevel.LOW for item in included):
        quality = QualityLevel.LOW
        warnings.append("included_image_has_low_quality")

    return AggregationResult(
        pose=pose,
        stats=stats,
        quality=quality,
        max_std=max_std,
        valid_image_count=len(included),
        warnings=warnings,
        finger_image_counts=finger_image_counts,
        finger_quality=finger_quality,
        finger_max_std=finger_max_std,
        target_fingers=target_fingers,
    )
