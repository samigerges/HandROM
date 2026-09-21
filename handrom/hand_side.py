"""Resolve one session hand from pose-dependent MediaPipe predictions."""

from __future__ import annotations

import math

from handrom.data_models import HandSide, ImageAnalysis, PoseType


def resolve_session_hand_side(
    analyses: list[ImageAnalysis],
) -> tuple[HandSide | None, bool]:
    """Return the session side and whether included photos disagreed.

    Side-view flexion can obscure the thumb and other digits, making MediaPipe's
    handedness classification unstable even while target landmarks remain usable.
    Extension images are therefore the canonical evidence for the session label.
    When more than one canonical image exists, confidence-weighted voting is used.
    """
    included = [
        item
        for item in analyses
        if item.valid and item.included and item.detected_side is not None
    ]
    if not included:
        return None, False

    all_sides = {item.detected_side for item in included}
    extension = [item for item in included if item.pose is PoseType.EXTENSION]
    canonical = extension or included
    scores = dict.fromkeys(HandSide, 0.0)
    for item in canonical:
        confidence = item.detection_confidence
        if confidence is None or not math.isfinite(confidence):
            confidence = 0.5
        confidence = min(1.0, max(0.0, float(confidence)))
        scores[item.detected_side] += confidence

    # Preserve the first canonical prediction as the deterministic tie-breaker.
    selected = canonical[0].detected_side
    for side in HandSide:
        if scores[side] > scores[selected]:
            selected = side
    return selected, len(all_sides) > 1
