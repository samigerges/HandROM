from __future__ import annotations

import pytest

from handrom.aggregation import aggregate_measurements
from handrom.data_models import PoseType, QualityLevel
from handrom.landmark_mapping import FINGERS
from tests.conftest import make_analysis


def test_one_valid_image_is_medium() -> None:
    result = aggregate_measurements(
        [make_analysis(PoseType.FLEXION, {"mcp": 90, "pip": 80, "dip": 50})],
        PoseType.FLEXION,
    )
    assert result.valid_image_count == 1
    assert result.quality is QualityLevel.MEDIUM
    assert result.stats["index"]["mcp"].median == 90


def test_two_consistent_images_are_high() -> None:
    analyses = [
        make_analysis(PoseType.FLEXION, {"mcp": 88, "pip": 80, "dip": 50}),
        make_analysis(PoseType.FLEXION, {"mcp": 92, "pip": 82, "dip": 52}),
    ]
    result = aggregate_measurements(analyses, PoseType.FLEXION)
    assert result.quality is QualityLevel.HIGH
    stats = result.stats["middle"]["mcp"]
    assert stats.median == 90
    assert stats.mean == 90
    assert stats.std == pytest.approx(2.0)
    assert stats.minimum == 88
    assert stats.maximum == 92
    assert stats.count == 2


def test_three_images_use_median() -> None:
    analyses = [
        make_analysis(PoseType.FLEXION, {"mcp": value, "pip": 80, "dip": 50})
        for value in (80, 90, 100)
    ]
    result = aggregate_measurements(analyses, PoseType.FLEXION)
    assert result.stats["index"]["mcp"].median == 90
    assert result.stats["index"]["mcp"].std == pytest.approx(8.1649658)
    assert result.quality is QualityLevel.MEDIUM


def test_inconsistent_images_are_low() -> None:
    analyses = [
        make_analysis(PoseType.FLEXION, {"mcp": 60, "pip": 60, "dip": 40}),
        make_analysis(PoseType.FLEXION, {"mcp": 100, "pip": 100, "dip": 80}),
    ]
    result = aggregate_measurements(analyses, PoseType.FLEXION)
    assert result.max_std == pytest.approx(20.0)
    assert result.quality is QualityLevel.LOW
    assert "inconsistent_measurements" in result.warnings


def test_excluded_image_is_not_used() -> None:
    analyses = [
        make_analysis(PoseType.FLEXION, {"mcp": 90, "pip": 80, "dip": 50}),
        make_analysis(PoseType.FLEXION, {"mcp": 20, "pip": 20, "dip": 20}, included=False),
    ]
    result = aggregate_measurements(analyses, PoseType.FLEXION)
    assert result.valid_image_count == 1
    assert result.stats["ring"]["mcp"].median == 90


def test_all_images_invalid_is_low() -> None:
    analyses = [make_analysis(PoseType.FLEXION, {"mcp": 90, "pip": 80, "dip": 50}, valid=False)]
    result = aggregate_measurements(analyses, PoseType.FLEXION)
    assert result.valid_image_count == 0
    assert result.quality is QualityLevel.LOW
    assert result.stats["little"]["dip"] is None


def test_extension_noise_floor_applied_only_to_median() -> None:
    result = aggregate_measurements(
        [make_analysis(PoseType.EXTENSION, {"mcp": 2.5, "pip": 10, "dip": 5})],
        PoseType.EXTENSION,
    )
    stats = result.stats["index"]["mcp"]
    assert stats.median == 0.0
    assert stats.mean == 2.5


def test_target_finger_images_are_aggregated_only_for_their_target() -> None:
    analyses = [
        make_analysis(
            PoseType.FLEXION,
            {"mcp": 80 + index, "pip": 70 + index, "dip": 40 + index},
            target_finger=finger,
        )
        for index, finger in enumerate(FINGERS)
    ]
    result = aggregate_measurements(analyses, PoseType.FLEXION)
    assert result.valid_image_count == 4
    assert result.finger_image_counts == dict.fromkeys(FINGERS, 1)
    assert result.quality is QualityLevel.MEDIUM
    assert result.stats["index"]["mcp"].median == 80
    assert result.stats["little"]["mcp"].median == 83


def test_single_target_finger_does_not_require_other_fingers() -> None:
    result = aggregate_measurements(
        [
            make_analysis(
                PoseType.FLEXION,
                {"mcp": 85, "pip": 80, "dip": 50},
                target_finger="index",
            )
        ],
        PoseType.FLEXION,
    )
    assert result.target_fingers == ("index",)
    assert result.quality is QualityLevel.MEDIUM
    assert result.stats["index"]["mcp"].median == 85
    assert result.stats["middle"]["mcp"] is None
