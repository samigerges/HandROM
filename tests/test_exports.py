from __future__ import annotations

import json

from handrom.aggregation import aggregate_measurements
from handrom.data_models import HandSide, PoseType, SessionMetadata
from handrom.exporters import (
    automatic_measurements_dataframe,
    manual_validation_dataframe,
    session_json_bytes,
    tam_summary_dataframe,
)
from handrom.report_generator import generate_pdf_report
from handrom.tam_calculator import calculate_tam
from tests.conftest import make_analysis


def _results():
    analyses = [
        make_analysis(PoseType.EXTENSION, {"mcp": 0, "pip": 10, "dip": 5}),
        make_analysis(PoseType.FLEXION, {"mcp": 90, "pip": 85, "dip": 55}),
    ]
    extension = aggregate_measurements(analyses, PoseType.EXTENSION)
    flexion = aggregate_measurements(analyses, PoseType.FLEXION)
    return analyses, extension, flexion, calculate_tam(extension, flexion)


def test_csv_schemas_and_blank_manual_columns() -> None:
    session = SessionMetadata("DEMO-01", HandSide.RIGHT)
    analyses, extension, flexion, tam = _results()
    automatic = automatic_measurements_dataframe(session, analyses, extension, flexion)
    assert len(automatic) == 24
    assert {"image_id", "mediapipe_angle_deg", "warning_codes"} <= set(automatic.columns)
    summary = tam_summary_dataframe(session, tam, 1, 1)
    assert len(summary) == 4
    assert summary.loc[0, "mediapipe_tam_deg"] == 215
    manual = manual_validation_dataframe(session, analyses)
    assert len(manual) == 24
    assert (manual["manual_goniometer_angle_deg"] == "").all()
    assert (manual["signed_error_deg"] == "").all()
    assert (manual["absolute_error_deg"] == "").all()


def test_json_contains_no_raw_images() -> None:
    session = SessionMetadata("DEMO-01", HandSide.RIGHT)
    analyses, extension, flexion, tam = _results()
    payload = json.loads(session_json_bytes(session, analyses, extension, flexion, tam))
    assert payload["session"]["participant_id"] == "DEMO-01"
    assert payload["tam_results"]["index"]["estimated_tam"] == 215
    serialized = json.dumps(payload)
    assert "original_rgb" not in serialized
    assert "annotated_png" not in serialized


def test_pdf_report_generation_with_images_optional() -> None:
    session = SessionMetadata("DEMO-01", HandSide.RIGHT)
    analyses, _extension, _flexion, tam = _results()
    without_images = generate_pdf_report(session, tam, analyses)
    with_images = generate_pdf_report(session, tam, analyses, include_annotated_images=True)
    assert without_images.startswith(b"%PDF")
    assert with_images.startswith(b"%PDF")


def test_target_finger_exports_do_not_duplicate_other_fingers() -> None:
    session = SessionMetadata("DEMO-01", HandSide.RIGHT)
    analysis = make_analysis(
        PoseType.EXTENSION,
        {"mcp": 4, "pip": 5, "dip": 6},
        target_finger="ring",
    )
    automatic = automatic_measurements_dataframe(session, [analysis])
    manual = manual_validation_dataframe(session, [analysis])
    assert len(automatic) == 3
    assert len(manual) == 3
    assert set(automatic["finger"]) == {"ring"}
    assert set(automatic["target_finger"]) == {"ring"}
    assert automatic["capture_confirmed"].all()


def test_one_finger_tam_summary_and_pdf() -> None:
    session = SessionMetadata("DEMO-01", HandSide.RIGHT)
    analyses = [
        make_analysis(
            PoseType.EXTENSION,
            {"mcp": 0, "pip": 5, "dip": 3},
            target_finger="index",
        ),
        make_analysis(
            PoseType.FLEXION,
            {"mcp": 90, "pip": 85, "dip": 55},
            target_finger="index",
        ),
    ]
    extension = aggregate_measurements(analyses, PoseType.EXTENSION)
    flexion = aggregate_measurements(analyses, PoseType.FLEXION)
    tam = calculate_tam(extension, flexion)
    summary = tam_summary_dataframe(
        session, tam, extension.finger_image_counts, flexion.finger_image_counts
    )
    assert len(summary) == 1
    assert summary.loc[0, "finger"] == "index"
    assert generate_pdf_report(session, tam, analyses).startswith(b"%PDF")
