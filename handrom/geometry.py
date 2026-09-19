"""Numerically safe three-dimensional vector geometry."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from handrom.config import EPSILON


class GeometryError(ValueError):
    """Raised when a geometric measurement cannot be calculated safely."""


Vector = NDArray[np.float64]


def as_vector(value: ArrayLike) -> Vector:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise GeometryError("Expected a three-dimensional vector.")
    if not np.all(np.isfinite(vector)):
        raise GeometryError("Vector contains NaN or infinite coordinates.")
    return vector


def subtract(a: ArrayLike, b: ArrayLike) -> Vector:
    return as_vector(a) - as_vector(b)


def dot(a: ArrayLike, b: ArrayLike) -> float:
    return float(np.dot(as_vector(a), as_vector(b)))


def cross(a: ArrayLike, b: ArrayLike) -> Vector:
    return np.cross(as_vector(a), as_vector(b))


def magnitude(vector: ArrayLike) -> float:
    length = float(np.linalg.norm(as_vector(vector)))
    if not math.isfinite(length):
        raise GeometryError("Vector magnitude is not finite.")
    return length


def normalize(vector: ArrayLike, *, epsilon: float = EPSILON) -> Vector:
    candidate = as_vector(vector)
    length = magnitude(candidate)
    if length <= epsilon:
        raise GeometryError("Vector has zero or nearly zero length.")
    return candidate / length


def clamp_cosine(value: float) -> float:
    if not math.isfinite(value):
        raise GeometryError("Cosine is not finite.")
    if abs(value - 1.0) <= 1e-12:
        return 1.0
    if abs(value + 1.0) <= 1e-12:
        return -1.0
    return min(1.0, max(-1.0, value))


def radians_to_degrees(value: float) -> float:
    if not math.isfinite(value):
        raise GeometryError("Angle is not finite.")
    return math.degrees(value)


def project_onto_axis(vector: ArrayLike, axis: ArrayLike) -> Vector:
    unit_axis = normalize(axis)
    return dot(vector, unit_axis) * unit_axis


def project_onto_plane(vector: ArrayLike, plane_normal: ArrayLike) -> Vector:
    candidate = as_vector(vector)
    return candidate - project_onto_axis(candidate, plane_normal)


def angle_between_segments(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> float:
    """Return flexion angle for A→B→C, with straight segments equal to 0°."""
    proximal = subtract(b, a)
    distal = subtract(c, b)
    denominator = magnitude(proximal) * magnitude(distal)
    if denominator <= EPSILON:
        raise GeometryError("A bone segment has zero or nearly zero length.")
    cosine = clamp_cosine(dot(proximal, distal) / denominator)
    angle = radians_to_degrees(math.acos(cosine))
    if not 0.0 <= angle <= 180.0 or not math.isfinite(angle):
        raise GeometryError("Calculated angle is outside 0°–180°.")
    return angle
