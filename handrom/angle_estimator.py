"""Joint-angle estimates derived exclusively from 3D world landmarks."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from handrom.data_models import PoseType
from handrom.geometry import GeometryError, angle_between_segments, dot, normalize
from handrom.landmark_mapping import FINGER_LANDMARKS, FINGERS, joint_triplets
from handrom.palm_coordinate_system import (
    PalmCoordinateSystem,
    build_palm_coordinate_system,
    projected_metacarpal_direction,
)


def validate_world_landmarks(landmarks: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise GeometryError("World landmarks must contain exactly 21 (x, y, z) points.")
    if not np.all(np.isfinite(points)):
        raise GeometryError("World landmarks contain NaN or infinite coordinates.")
    return points


def estimate_side_view_angles(
    normalized_landmarks: ArrayLike,
    finger: str,
    *,
    image_size: tuple[int, int],
) -> dict[str, float]:
    """Estimate one finger's flexion angles in a calibrated side-view image plane.

    Normalized x/y coordinates must be scaled back to pixels before measuring so
    a non-square photograph does not distort the angles. The wrist-to-MCP ray is
    used as the visible metacarpal reference for MCP flexion.
    """
    if finger not in FINGER_LANDMARKS:
        raise GeometryError(f"Unsupported target finger: {finger}.")
    points = np.asarray(normalized_landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise GeometryError("Normalized landmarks must contain exactly 21 (x, y, z) points.")
    if not np.all(np.isfinite(points)):
        raise GeometryError("Normalized landmarks contain NaN or infinite coordinates.")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise GeometryError("Image dimensions must be positive.")

    pixel_xy = points[:, :2] * np.asarray((width, height), dtype=np.float64)
    image_points = np.column_stack((pixel_xy, np.zeros(len(pixel_xy), dtype=np.float64)))
    mapping = FINGER_LANDMARKS[finger]
    triplets = joint_triplets(finger)
    return {
        "mcp": angle_between_segments(
            image_points[0], image_points[mapping["mcp"]], image_points[mapping["pip"]]
        ),
        "pip": angle_between_segments(
            image_points[triplets["pip"][0]],
            image_points[triplets["pip"][1]],
            image_points[triplets["pip"][2]],
        ),
        "dip": angle_between_segments(
            image_points[triplets["dip"][0]],
            image_points[triplets["dip"][1]],
            image_points[triplets["dip"][2]],
        ),
    }


def estimate_mcp_flexion(
    points: NDArray[np.float64],
    finger: str,
    palm: PalmCoordinateSystem,
) -> float:
    """Estimate MCP flexion after removing the transverse ab/adduction component."""
    mapping = FINGER_LANDMARKS[finger]
    metacarpal = projected_metacarpal_direction(points[0], points[mapping["mcp"]], palm)
    proximal_phalanx = normalize(points[mapping["pip"]] - points[mapping["mcp"]])
    longitudinal_component = dot(proximal_phalanx, metacarpal)
    normal_component = abs(dot(proximal_phalanx, palm.normal))
    angle = math.degrees(math.atan2(normal_component, longitudinal_component))
    if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
        raise GeometryError("Calculated MCP angle is invalid.")
    return angle


def estimate_joint_angles(
    world_landmarks: ArrayLike,
    pose: PoseType | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate MCP, PIP, and DIP flexion for the four non-thumb digits."""
    del pose  # Same geometry; pose interpretation is applied during aggregation/TAM.
    points = validate_world_landmarks(world_landmarks)
    palm = build_palm_coordinate_system(points)
    results: dict[str, dict[str, float]] = {}
    for finger in FINGERS:
        triplets = joint_triplets(finger)
        results[finger] = {
            "mcp": estimate_mcp_flexion(points, finger, palm),
            "pip": angle_between_segments(
                points[triplets["pip"][0]], points[triplets["pip"][1]], points[triplets["pip"][2]]
            ),
            "dip": angle_between_segments(
                points[triplets["dip"][0]], points[triplets["dip"][1]], points[triplets["dip"][2]]
            ),
        }
    return results
