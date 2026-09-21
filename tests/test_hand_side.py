from __future__ import annotations

from handrom.data_models import HandSide, PoseType
from handrom.hand_side import resolve_session_hand_side
from tests.conftest import make_analysis


def _analysis(pose: PoseType, side: HandSide, confidence: float):
    analysis = make_analysis(
        pose,
        {"mcp": 80.0, "pip": 70.0, "dip": 40.0},
        target_finger="index",
    )
    analysis.detected_side = side
    analysis.detection_confidence = confidence
    return analysis


def test_flexion_disagreement_does_not_override_extension_side() -> None:
    side, disagreed = resolve_session_hand_side(
        [
            _analysis(PoseType.EXTENSION, HandSide.LEFT, 0.75),
            _analysis(PoseType.FLEXION, HandSide.RIGHT, 0.99),
        ]
    )
    assert side is HandSide.LEFT
    assert disagreed


def test_multiple_extension_images_use_confidence_weighted_vote() -> None:
    side, disagreed = resolve_session_hand_side(
        [
            _analysis(PoseType.EXTENSION, HandSide.LEFT, 0.80),
            _analysis(PoseType.EXTENSION, HandSide.LEFT, 0.70),
            _analysis(PoseType.EXTENSION, HandSide.RIGHT, 0.95),
        ]
    )
    assert side is HandSide.LEFT
    assert disagreed


def test_no_included_detection_returns_no_session_side() -> None:
    analysis = _analysis(PoseType.EXTENSION, HandSide.LEFT, 0.90)
    analysis.included = False
    assert resolve_session_hand_side([analysis]) == (None, False)
