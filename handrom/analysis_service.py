"""End-to-end processing of one validated image."""

from __future__ import annotations

from handrom.angle_estimator import estimate_joint_angles, estimate_side_view_angles
from handrom.data_models import (
    HandSide,
    ImageAnalysis,
    PoseType,
    QualityAssessment,
    QualityLevel,
    ValidatedImage,
)
from handrom.geometry import GeometryError
from handrom.hand_detector import HandDetectionError, detect_hand
from handrom.landmark_mapping import FINGER_LANDMARKS
from handrom.quality import assess_image_quality
from handrom.visualization import annotate_image


def analyze_validated_image(
    image: ValidatedImage,
    *,
    pose: PoseType,
    expected_side: HandSide,
    mirrored: bool,
    landmarker: object,
    target_finger: str | None = None,
    capture_confirmed: bool = False,
) -> ImageAnalysis:
    """Detect, measure, quality-check, and annotate one in-memory photograph."""
    empty_angles: dict[str, dict[str, float | None]] = {}
    try:
        if target_finger is not None and target_finger not in FINGER_LANDMARKS:
            raise ValueError(f"Unsupported target finger: {target_finger}.")
        detection = detect_hand(landmarker, image.inference_rgb)
        if target_finger is None:
            angles = estimate_joint_angles(detection.world_landmarks, pose)
        else:
            inference_height, inference_width = image.inference_rgb.shape[:2]
            angles = {
                target_finger: estimate_side_view_angles(
                    detection.normalized_landmarks,
                    target_finger,
                    image_size=(inference_width, inference_height),
                )
            }
        quality = assess_image_quality(
            image.rgb,
            detection.normalized_landmarks,
            expected_side=expected_side,
            detected_side=detection.detected_side,
            handedness_confidence=detection.confidence,
            mirrored=mirrored,
            target_finger=target_finger,
        )
        effective_side = detection.detected_side
        if mirrored:
            effective_side = HandSide.LEFT if effective_side is HandSide.RIGHT else HandSide.RIGHT
        side_matches = effective_side is expected_side
        valid_geometry = all(
            quality.checks.get(check, False)
            for check in ("resolution", "landmarks_complete", "framing", "hand_size")
        )
        analysis = ImageAnalysis(
            image_id=image.image_id,
            original_name=image.original_name,
            pose=pose,
            valid=bool(valid_geometry),
            included=bool(
                valid_geometry
                and side_matches
            ),
            angles=angles,
            quality=quality,
            target_finger=target_finger,
            capture_confirmed=capture_confirmed,
            detected_side=detection.detected_side,
            detection_confidence=detection.confidence,
            resolution=(image.width, image.height),
            normalized_landmarks=detection.normalized_landmarks,
            world_landmarks=detection.world_landmarks,
            original_rgb=image.rgb,
            side_confirmed=side_matches,
        )
        analysis.annotated_png = annotate_image(
            image.rgb,
            detection.normalized_landmarks,
            angles,
            side=detection.detected_side,
            pose=pose,
            quality=quality.level,
            target_finger=target_finger,
        )
        return analysis
    except (HandDetectionError, GeometryError) as exc:
        code = exc.code if isinstance(exc, HandDetectionError) else "invalid_geometry"
        return ImageAnalysis(
            image_id=image.image_id,
            original_name=image.original_name,
            pose=pose,
            valid=False,
            included=False,
            angles=empty_angles,
            quality=QualityAssessment(
                level=QualityLevel.LOW,
                warnings=[code],
                checks={"analysis_succeeded": False},
            ),
            target_finger=target_finger,
            capture_confirmed=capture_confirmed,
            resolution=(image.width, image.height),
            original_rgb=image.rgb,
            error=str(exc),
        )
