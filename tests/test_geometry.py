from __future__ import annotations

import math

import numpy as np
import pytest

from handrom.geometry import (
    GeometryError,
    angle_between_segments,
    clamp_cosine,
    normalize,
    project_onto_plane,
)
from handrom.palm_coordinate_system import build_palm_coordinate_system


def test_straight_segments_are_zero() -> None:
    assert angle_between_segments((0, 0, 0), (1, 0, 0), (2, 0, 0)) == pytest.approx(0.0)


def test_perpendicular_segments_are_ninety() -> None:
    assert angle_between_segments((0, 0, 0), (1, 0, 0), (1, 1, 0)) == pytest.approx(90.0)


def test_forty_five_degree_bend() -> None:
    assert angle_between_segments((0, 0, 0), (1, 0, 0), (2, 1, 0)) == pytest.approx(45.0)


def test_cosine_clamping() -> None:
    assert clamp_cosine(1.0000001) == 1.0
    assert clamp_cosine(-1.0000001) == -1.0


def test_zero_length_rejected() -> None:
    with pytest.raises(GeometryError):
        angle_between_segments((0, 0, 0), (0, 0, 0), (1, 0, 0))
    with pytest.raises(GeometryError):
        normalize((0, 0, 0))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinates_rejected(bad: float) -> None:
    with pytest.raises(GeometryError):
        angle_between_segments((bad, 0, 0), (1, 0, 0), (2, 0, 0))


def test_plane_projection() -> None:
    projected = project_onto_plane((1, 2, 3), (0, 0, 1))
    np.testing.assert_allclose(projected, (1, 2, 0))


def test_palm_plane_degeneracy_rejected() -> None:
    points = np.zeros((21, 3), dtype=np.float64)
    points[5] = (1, 0, 0)
    points[9] = (2, 0, 0)
    points[17] = (3, 0, 0)
    with pytest.raises(GeometryError):
        build_palm_coordinate_system(points)


def test_valid_palm_basis_is_orthogonal(straight_world_landmarks: np.ndarray) -> None:
    palm = build_palm_coordinate_system(straight_world_landmarks)
    assert np.dot(palm.normal, palm.longitudinal) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(palm.transverse, palm.longitudinal) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(palm.normal) == pytest.approx(1.0)
