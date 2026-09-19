"""Typed data exchanged between HandROM modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

import numpy as np

from handrom.config import APP_VERSION


class HandSide(StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


class PoseType(StrEnum):
    EXTENSION = "maximum_extension"
    FLEXION = "maximum_flexion"


class QualityLevel(StrEnum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass(slots=True)
class SessionMetadata:
    participant_id: str
    hand_side: HandSide
    mirrored: bool = False
    label: str = ""
    notes: str = ""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    application_version: str = APP_VERSION


@dataclass(slots=True)
class ValidatedImage:
    image_id: str
    original_name: str
    mime_type: str
    rgb: np.ndarray
    inference_rgb: np.ndarray
    width: int
    height: int
    format: str


@dataclass(slots=True)
class QualityAssessment:
    level: QualityLevel
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionResult:
    normalized_landmarks: np.ndarray
    world_landmarks: np.ndarray
    detected_side: HandSide
    confidence: float


@dataclass(slots=True)
class ImageAnalysis:
    image_id: str
    original_name: str
    pose: PoseType
    valid: bool
    included: bool
    angles: dict[str, dict[str, float | None]]
    quality: QualityAssessment
    target_finger: str | None = None
    capture_confirmed: bool = False
    detected_side: HandSide | None = None
    detection_confidence: float | None = None
    resolution: tuple[int, int] = (0, 0)
    normalized_landmarks: np.ndarray | None = None
    world_landmarks: np.ndarray | None = None
    original_rgb: np.ndarray | None = None
    annotated_png: bytes | None = None
    error: str | None = None
    side_confirmed: bool = False


@dataclass(slots=True)
class AggregateStats:
    median: float
    mean: float
    std: float
    minimum: float
    maximum: float
    count: int


@dataclass(slots=True)
class AggregationResult:
    pose: PoseType
    stats: dict[str, dict[str, AggregateStats | None]]
    quality: QualityLevel
    max_std: float | None
    valid_image_count: int
    warnings: list[str] = field(default_factory=list)
    finger_image_counts: dict[str, int] = field(default_factory=dict)
    finger_quality: dict[str, QualityLevel] = field(default_factory=dict)
    finger_max_std: dict[str, float | None] = field(default_factory=dict)
    target_fingers: tuple[str, ...] = ()


@dataclass(slots=True)
class TamResult:
    finger: str
    extension_deficits: dict[str, float]
    maximum_flexion: dict[str, float]
    total_flexion: float
    total_extension_deficit: float
    tam: float
    quality: QualityLevel


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, enums, and arrays to JSON-safe objects."""
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    return value
