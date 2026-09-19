"""Correct Total Active Motion calculation."""

from __future__ import annotations

import math

from handrom.data_models import AggregationResult, QualityLevel, TamResult
from handrom.landmark_mapping import FINGERS, JOINTS


class TamCalculationError(ValueError):
    """Raised when a required TAM input is missing or invalid."""


def _extract(aggregation: AggregationResult, finger: str, joint: str) -> float:
    stats = aggregation.stats.get(finger, {}).get(joint)
    if stats is None:
        raise TamCalculationError(f"Missing {finger} {joint} measurement.")
    value = float(stats.median)
    if not math.isfinite(value) or not 0.0 <= value <= 180.0:
        raise TamCalculationError(f"Invalid {finger} {joint} angle.")
    return value


def calculate_tam(
    extension: AggregationResult,
    flexion: AggregationResult,
) -> dict[str, TamResult]:
    """Calculate TAM = total maximum flexion − total extension deficit."""
    if extension.quality is QualityLevel.LOW or flexion.quality is QualityLevel.LOW:
        raise TamCalculationError("TAM requires non-Low extension and flexion quality.")
    extension_targets = extension.target_fingers or FINGERS
    flexion_targets = flexion.target_fingers or FINGERS
    target_fingers = tuple(
        finger for finger in FINGERS if finger in extension_targets and finger in flexion_targets
    )
    if not target_fingers:
        raise TamCalculationError("Extension and flexion do not contain the same target finger.")
    results: dict[str, TamResult] = {}
    for finger in target_fingers:
        extension_quality = extension.finger_quality.get(finger, extension.quality)
        flexion_quality = flexion.finger_quality.get(finger, flexion.quality)
        if extension_quality is QualityLevel.LOW or flexion_quality is QualityLevel.LOW:
            raise TamCalculationError(f"TAM requires non-Low {finger} image quality.")
        quality = (
            QualityLevel.HIGH
            if extension_quality is flexion_quality is QualityLevel.HIGH
            else QualityLevel.MEDIUM
        )
        deficits = {joint: _extract(extension, finger, joint) for joint in JOINTS}
        maximum = {joint: _extract(flexion, finger, joint) for joint in JOINTS}
        total_flexion = math.fsum(maximum.values())
        total_extension_deficit = math.fsum(deficits.values())
        tam = total_flexion - total_extension_deficit
        if not math.isfinite(tam) or tam < 0.0:
            raise TamCalculationError(f"Mathematically invalid negative TAM for {finger}.")
        results[finger] = TamResult(
            finger=finger,
            extension_deficits=deficits,
            maximum_flexion=maximum,
            total_flexion=total_flexion,
            total_extension_deficit=total_extension_deficit,
            tam=tam,
            quality=quality,
        )
    return results


def display_degrees(value: float) -> str:
    """Round only at the display boundary."""
    return f"{value:.1f}°"
