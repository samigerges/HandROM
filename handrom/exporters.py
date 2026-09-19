"""CSV and JSON exports for later manual-goniometer validation."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

from handrom.data_models import (
    AggregationResult,
    ImageAnalysis,
    SessionMetadata,
    TamResult,
)
from handrom.landmark_mapping import FINGERS, JOINTS


def automatic_measurements_dataframe(
    session: SessionMetadata,
    analyses: list[ImageAnalysis],
    extension: AggregationResult | None = None,
    flexion: AggregationResult | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    aggregations = {item.pose: item for item in (extension, flexion) if item is not None}
    for analysis in analyses:
        measurement_quality = aggregations.get(analysis.pose)
        analysis_fingers = (analysis.target_finger,) if analysis.target_finger else FINGERS
        for finger in analysis_fingers:
            for joint in JOINTS:
                rows.append(
                    {
                        "session_id": session.session_id,
                        "participant_id": session.participant_id,
                        "timestamp_utc": session.timestamp_utc,
                        "application_version": session.application_version,
                        "hand_side": session.hand_side.value,
                        "pose": analysis.pose.value,
                        "image_id": analysis.image_id,
                        "target_finger": analysis.target_finger or "",
                        "finger": finger,
                        "joint": joint,
                        "mediapipe_angle_deg": analysis.angles.get(finger, {}).get(joint),
                        "included_in_aggregation": analysis.included,
                        "image_quality_level": analysis.quality.level.value,
                        "measurement_quality_level": (
                            measurement_quality.finger_quality.get(
                                finger, measurement_quality.quality
                            ).value
                            if measurement_quality
                            else ""
                        ),
                        "detection_confidence": analysis.detection_confidence,
                        "capture_confirmed": analysis.capture_confirmed,
                        "warning_codes": "|".join(analysis.quality.warnings),
                    }
                )
    return pd.DataFrame(rows)


def tam_summary_dataframe(
    session: SessionMetadata,
    tam_results: dict[str, TamResult],
    extension_image_count: int | Mapping[str, int],
    flexion_image_count: int | Mapping[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for finger in FINGERS:
        if finger not in tam_results:
            continue
        result = tam_results[finger]
        rows.append(
            {
                "session_id": session.session_id,
                "participant_id": session.participant_id,
                "timestamp_utc": session.timestamp_utc,
                "hand_side": session.hand_side.value,
                "finger": finger,
                "mcp_extension_deficit_deg": result.extension_deficits["mcp"],
                "pip_extension_deficit_deg": result.extension_deficits["pip"],
                "dip_extension_deficit_deg": result.extension_deficits["dip"],
                "mcp_max_flexion_deg": result.maximum_flexion["mcp"],
                "pip_max_flexion_deg": result.maximum_flexion["pip"],
                "dip_max_flexion_deg": result.maximum_flexion["dip"],
                "total_flexion_deg": result.total_flexion,
                "total_extension_deficit_deg": result.total_extension_deficit,
                "mediapipe_tam_deg": result.tam,
                "extension_image_count": _image_count(extension_image_count, finger),
                "flexion_image_count": _image_count(flexion_image_count, finger),
                "measurement_quality_level": result.quality.value,
            }
        )
    return pd.DataFrame(rows)


def _image_count(value: int | Mapping[str, int], finger: str) -> int:
    return int(value.get(finger, 0)) if isinstance(value, Mapping) else int(value)


def manual_validation_dataframe(
    session: SessionMetadata,
    analyses: list[ImageAnalysis],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for analysis in analyses:
        if not analysis.valid or not analysis.included:
            continue
        analysis_fingers = (analysis.target_finger,) if analysis.target_finger else FINGERS
        for finger in analysis_fingers:
            for joint in JOINTS:
                rows.append(
                    {
                        "session_id": session.session_id,
                        "participant_id": session.participant_id,
                        "hand_side": session.hand_side.value,
                        "pose": analysis.pose.value,
                        "finger": finger,
                        "joint": joint,
                        "mediapipe_angle_deg": analysis.angles[finger][joint],
                        "manual_goniometer_angle_deg": "",
                        "signed_error_deg": "",
                        "absolute_error_deg": "",
                    }
                )
    return pd.DataFrame(rows)


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def session_json_bytes(
    session: SessionMetadata,
    analyses: list[ImageAnalysis],
    extension: AggregationResult,
    flexion: AggregationResult,
    tam_results: dict[str, TamResult],
) -> bytes:
    per_image = []
    for analysis in analyses:
        per_image.append(
            {
                "image_id": analysis.image_id,
                "original_name": analysis.original_name,
                "pose": analysis.pose.value,
                "target_finger": analysis.target_finger,
                "valid": analysis.valid,
                "included": analysis.included,
                "detected_side": analysis.detected_side.value if analysis.detected_side else None,
                "detection_confidence": analysis.detection_confidence,
                "resolution": list(analysis.resolution),
                "angles": analysis.angles,
                "quality": {
                    "level": analysis.quality.level.value,
                    "warnings": analysis.quality.warnings,
                    "checks": analysis.quality.checks,
                    "metrics": analysis.quality.metrics,
                },
                "side_confirmed": analysis.side_confirmed,
                "capture_confirmed": analysis.capture_confirmed,
                "error": analysis.error,
            }
        )
    payload = {
        "session": {
            "session_id": session.session_id,
            "participant_id": session.participant_id,
            "timestamp_utc": session.timestamp_utc,
            "application_version": session.application_version,
            "hand_side": session.hand_side.value,
            "mirrored": session.mirrored,
            "label": session.label,
            "notes": session.notes,
        },
        "images": per_image,
        "aggregated_angles": {
            "maximum_extension": _aggregation_payload(extension),
            "maximum_flexion": _aggregation_payload(flexion),
        },
        "tam_results": {
            finger: {
                "extension_deficits": result.extension_deficits,
                "maximum_flexion": result.maximum_flexion,
                "total_flexion": result.total_flexion,
                "total_extension_deficit": result.total_extension_deficit,
                "estimated_tam": result.tam,
                "measurement_quality": result.quality.value,
            }
            for finger, result in tam_results.items()
        },
    }
    return json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")


def _aggregation_payload(result: AggregationResult) -> dict[str, object]:
    return {
        "quality": result.quality.value,
        "valid_image_count": result.valid_image_count,
        "max_std_deg": result.max_std,
        "warning_codes": result.warnings,
        "target_fingers": list(result.target_fingers),
        "finger_image_counts": result.finger_image_counts,
        "finger_quality": {
            finger: quality.value for finger, quality in result.finger_quality.items()
        },
        "finger_max_std_deg": result.finger_max_std,
        "joints": {
            finger: {
                joint: (
                    {
                        "median": stat.median,
                        "mean": stat.mean,
                        "std": stat.std,
                        "minimum": stat.minimum,
                        "maximum": stat.maximum,
                        "count": stat.count,
                    }
                    if stat
                    else None
                )
                for joint, stat in joint_stats.items()
            }
            for finger, joint_stats in result.stats.items()
        },
    }
