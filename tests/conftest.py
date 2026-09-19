from __future__ import annotations

import numpy as np
import pytest

from handrom.data_models import (
    HandSide,
    ImageAnalysis,
    PoseType,
    QualityAssessment,
    QualityLevel,
)
from handrom.landmark_mapping import FINGER_LANDMARKS, FINGERS


@pytest.fixture
def straight_world_landmarks() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = (0.0, 0.0, 0.0)
    points[1:5] = [(-0.7, 0.15, 0), (-0.8, 0.25, 0), (-0.9, 0.35, 0), (-1.0, 0.45, 0)]
    bases = {
        "index": np.array((-0.55, 0.65, 0.0)),
        "middle": np.array((-0.18, 0.78, 0.0)),
        "ring": np.array((0.22, 0.72, 0.0)),
        "little": np.array((0.58, 0.58, 0.0)),
    }
    for finger, base in bases.items():
        mapping = FINGER_LANDMARKS[finger]
        direction = base / np.linalg.norm(base)
        points[mapping["mcp"]] = base
        points[mapping["pip"]] = base + direction * 0.30
        points[mapping["dip"]] = base + direction * 0.52
        points[mapping["tip"]] = base + direction * 0.70
    return points


def make_analysis(
    pose: PoseType,
    base: dict[str, float],
    *,
    included: bool = True,
    valid: bool = True,
    quality: QualityLevel = QualityLevel.HIGH,
    target_finger: str | None = None,
) -> ImageAnalysis:
    angles = (
        {target_finger: dict(base)}
        if target_finger is not None
        else {finger: dict(base) for finger in FINGERS}
    )
    return ImageAnalysis(
        image_id=f"{pose.value}-{base['mcp']}-{id(base)}",
        original_name="fixture.png",
        pose=pose,
        valid=valid,
        included=included,
        angles=angles,
        quality=QualityAssessment(level=quality),
        target_finger=target_finger,
        capture_confirmed=target_finger is not None,
        detected_side=HandSide.RIGHT,
        detection_confidence=0.99,
        resolution=(800, 600),
        side_confirmed=True,
    )
