from __future__ import annotations

import math

import pytest

import handrom.tam_calculator as tam_module
from handrom.aggregation import aggregate_measurements
from handrom.data_models import PoseType, QualityLevel
from handrom.landmark_mapping import FINGERS
from handrom.tam_calculator import TamCalculationError, calculate_tam, display_degrees
from tests.conftest import make_analysis


def test_correct_tam_formula_all_fingers() -> None:
    extension = aggregate_measurements(
        [make_analysis(PoseType.EXTENSION, {"mcp": 0.0, "pip": 10.0, "dip": 5.0})],
        PoseType.EXTENSION,
    )
    flexion = aggregate_measurements(
        [make_analysis(PoseType.FLEXION, {"mcp": 90.0, "pip": 85.0, "dip": 55.0})],
        PoseType.FLEXION,
    )
    results = calculate_tam(extension, flexion)
    assert tuple(results) == FINGERS
    for result in results.values():
        assert result.total_flexion == 230.0
        assert result.total_extension_deficit == 15.0
        assert result.tam == 215.0
        assert result.quality is QualityLevel.MEDIUM


def test_missing_joint_rejected() -> None:
    extension = aggregate_measurements([], PoseType.EXTENSION)
    flexion = aggregate_measurements(
        [make_analysis(PoseType.FLEXION, {"mcp": 90.0, "pip": 85.0, "dip": 55.0})],
        PoseType.FLEXION,
    )
    with pytest.raises(TamCalculationError):
        calculate_tam(extension, flexion)


def test_negative_tam_rejected() -> None:
    extension = aggregate_measurements(
        [make_analysis(PoseType.EXTENSION, {"mcp": 100.0, "pip": 100.0, "dip": 100.0})],
        PoseType.EXTENSION,
    )
    flexion = aggregate_measurements(
        [make_analysis(PoseType.FLEXION, {"mcp": 10.0, "pip": 10.0, "dip": 10.0})],
        PoseType.FLEXION,
    )
    with pytest.raises(TamCalculationError, match="negative TAM"):
        calculate_tam(extension, flexion)


def test_rounding_is_display_only() -> None:
    value = 215.049
    assert display_degrees(value) == "215.0°"
    assert math.isclose(value, 215.049)


def test_no_tam_percentage_function_exists() -> None:
    assert not hasattr(tam_module, "calculate_tam_percentage")


def test_tam_can_be_calculated_for_one_target_finger() -> None:
    extension = aggregate_measurements(
        [
            make_analysis(
                PoseType.EXTENSION,
                {"mcp": 0.0, "pip": 5.0, "dip": 3.0},
                target_finger="ring",
            )
        ],
        PoseType.EXTENSION,
    )
    flexion = aggregate_measurements(
        [
            make_analysis(
                PoseType.FLEXION,
                {"mcp": 85.0, "pip": 90.0, "dip": 55.0},
                target_finger="ring",
            )
        ],
        PoseType.FLEXION,
    )
    results = calculate_tam(extension, flexion)
    assert tuple(results) == ("ring",)
    assert results["ring"].tam == 222.0
