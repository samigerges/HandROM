"""MediaPipe Tasks Hand Landmarker integration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from handrom.data_models import DetectionResult, HandSide


class HandDetectionError(RuntimeError):
    """An expected, user-correctable hand-detection failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def create_hand_landmarker(model_path: str | Path):
    """Create a MediaPipe Hand Landmarker in blocking IMAGE mode."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"MediaPipe model not found at {path}. See README model setup instructions."
        )
    _configure_mediapipe_cache()
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is not installed. Install requirements.txt.") from exc

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hand(landmarker: object, rgb: np.ndarray) -> DetectionResult:
    """Detect exactly one hand and require both normalized and world landmarks."""
    _configure_mediapipe_cache()
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is not installed. Install requirements.txt.") from exc
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    result = landmarker.detect(  # type: ignore[attr-defined]
        mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    )
    hands = list(result.hand_landmarks or [])
    if not hands:
        raise HandDetectionError(
            "no_hand_detected",
            "No hand was detected. Retake the photograph with one complete hand, even lighting, and a contrasting background.",
        )
    if len(hands) != 1:
        raise HandDetectionError(
            "multiple_hands_detected",
            "More than one hand was detected. Upload a photograph containing only the selected hand.",
        )
    if not result.hand_world_landmarks or len(result.hand_world_landmarks) != 1:
        raise HandDetectionError(
            "missing_world_landmarks",
            "MediaPipe did not return 3D world landmarks, so angles cannot be calculated safely.",
        )
    if not result.handedness or not result.handedness[0]:
        raise HandDetectionError(
            "missing_handedness",
            "MediaPipe did not return a handedness estimate.",
        )

    category = result.handedness[0][0]
    name = str(category.category_name).title()
    try:
        side = HandSide(name)
    except ValueError as exc:
        raise HandDetectionError("invalid_handedness", f"Unexpected handedness: {name}") from exc
    normalized = np.asarray([[point.x, point.y, point.z] for point in hands[0]], dtype=np.float64)
    world = np.asarray(
        [[point.x, point.y, point.z] for point in result.hand_world_landmarks[0]],
        dtype=np.float64,
    )
    if normalized.shape != (21, 3) or world.shape != (21, 3):
        raise HandDetectionError(
            "incomplete_landmarks",
            "MediaPipe did not return all 21 normalized and world landmarks.",
        )
    return DetectionResult(
        normalized_landmarks=normalized,
        world_landmarks=world,
        detected_side=side,
        confidence=float(category.score),
    )


def _configure_mediapipe_cache() -> None:
    """Keep Matplotlib's MediaPipe import cache in a writable temporary directory."""
    cache_dir = Path(tempfile.gettempdir()) / "handrom-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
