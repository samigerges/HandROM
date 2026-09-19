"""Hand-local palm plane construction used for MCP flexion estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from handrom.geometry import GeometryError, cross, normalize, project_onto_plane


@dataclass(frozen=True, slots=True)
class PalmCoordinateSystem:
    longitudinal: NDArray[np.float64]
    transverse: NDArray[np.float64]
    normal: NDArray[np.float64]


def build_palm_coordinate_system(
    landmarks: NDArray[np.float64],
) -> PalmCoordinateSystem:
    """Build an orthonormal palm basis from wrist and MCP landmarks."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise GeometryError("Expected 21 finite three-dimensional hand landmarks.")

    raw_longitudinal = points[9] - points[0]
    raw_transverse = points[17] - points[5]
    longitudinal = normalize(raw_longitudinal)
    transverse = normalize(project_onto_plane(raw_transverse, longitudinal))
    normal = normalize(cross(transverse, longitudinal))
    transverse = normalize(cross(longitudinal, normal))
    return PalmCoordinateSystem(longitudinal, transverse, normal)


def projected_metacarpal_direction(
    wrist: NDArray[np.float64],
    mcp: NDArray[np.float64],
    palm: PalmCoordinateSystem,
) -> NDArray[np.float64]:
    return normalize(project_onto_plane(mcp - wrist, palm.normal))
