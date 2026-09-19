from __future__ import annotations

import numpy as np
import pytest

from handrom.angle_estimator import (
    estimate_joint_angles,
    estimate_mcp_flexion,
    estimate_side_view_angles,
    validate_world_landmarks,
)
from handrom.geometry import GeometryError
from handrom.landmark_mapping import FINGERS
from handrom.palm_coordinate_system import build_palm_coordinate_system


def test_straight_hand_angles_are_zero(straight_world_landmarks: np.ndarray) -> None:
    result = estimate_joint_angles(straight_world_landmarks)
    assert tuple(result) == FINGERS
    for finger in FINGERS:
        assert result[finger]["mcp"] == pytest.approx(0.0, abs=1e-6)
        assert result[finger]["pip"] == pytest.approx(0.0, abs=1e-6)
        assert result[finger]["dip"] == pytest.approx(0.0, abs=1e-6)


def test_mcp_out_of_plane_flexion(straight_world_landmarks: np.ndarray) -> None:
    points = straight_world_landmarks.copy()
    palm = build_palm_coordinate_system(points)
    mapping = {"mcp": 5, "pip": 6}
    metacarpal = points[mapping["mcp"]] / np.linalg.norm(points[mapping["mcp"]])
    points[mapping["pip"]] = points[mapping["mcp"]] + 0.3 * (
        np.cos(np.deg2rad(45)) * metacarpal + np.sin(np.deg2rad(45)) * palm.normal
    )
    assert estimate_mcp_flexion(points, "index", palm) == pytest.approx(45.0)


def test_invalid_world_landmark_shape() -> None:
    with pytest.raises(GeometryError):
        validate_world_landmarks(np.zeros((20, 3)))


def test_side_view_angles_use_only_selected_finger_chain() -> None:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0, :2] = (0.10, 0.20)
    points[5, :2] = (0.20, 0.20)
    points[6, :2] = (0.30, 0.20)
    points[7, :2] = (0.30, 0.40)
    points[8, :2] = (0.40, 0.40)
    result = estimate_side_view_angles(points, "index", image_size=(1000, 500))
    assert result == pytest.approx({"mcp": 0.0, "pip": 90.0, "dip": 90.0})


def test_side_view_angles_restore_pixel_aspect_ratio() -> None:
    width, height = 400, 300
    pixel_points = ((100, 100), (200, 150), (150, 250))
    points = np.zeros((21, 3), dtype=np.float64)
    for index, (x, y) in zip((0, 5, 6), pixel_points, strict=True):
        points[index, :2] = (x / width, y / height)
    points[7, :2] = (100 / width, 250 / height)
    points[8, :2] = (50 / width, 250 / height)
    result = estimate_side_view_angles(points, "index", image_size=(width, height))
    assert result["mcp"] == pytest.approx(90.0)


def test_non_finite_world_landmark() -> None:
    points = np.zeros((21, 3))
    points[8, 1] = np.nan
    with pytest.raises(GeometryError):
        validate_world_landmarks(points)
