"""HandROM Streamlit application entry point."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from handrom.aggregation import aggregate_measurements
from handrom.analysis_service import analyze_validated_image
from handrom.config import (
    APP_VERSION,
    MAX_IMAGES_PER_POSE,
    MODEL_PATH,
    POSE_ASSET_DIR,
)
from handrom.data_models import (
    HandSide,
    ImageAnalysis,
    PoseType,
    QualityLevel,
    SessionMetadata,
)
from handrom.demo_data import create_demo_analyses
from handrom.hand_detector import create_hand_landmarker
from handrom.image_processing import ImageValidationError, validate_uploaded_file
from handrom.landmark_mapping import FINGERS
from handrom.report_generator import generate_pdf_report
from handrom.session_manager import (
    clear_session_state,
    initialize_session_state,
)
from handrom.tam_calculator import TamCalculationError, calculate_tam

st.set_page_config(page_title="HandROM", page_icon="🖐️", layout="wide")

DEMO_MODE = os.getenv("HANDROM_DEMO", "0") == "1"

SEPARATE_FINGER_POSE_ASSETS = {
    "index": (
        ("Maximum active extension", "index_extension_avatar_hand.png"),
        ("Maximum active flexion", "index_flexion_avatar_hand.png"),
    ),
    "middle": (
        ("Maximum active extension", "middle_extension_avatar_hand.png"),
        ("Maximum active flexion", "middle_flexion_avatar_hand.png"),
    ),
    "ring": (
        ("Maximum active extension", "ring_extension_avatar_hand.png"),
        ("Maximum active flexion", "ring_flexion_avatar_hand.png"),
    ),
    "little": (
        ("Maximum active extension", "little_extension_avatar_hand.png"),
        ("Maximum active flexion", "little_flexion_avatar_hand.png"),
    ),
}


@st.cache_resource(show_spinner="Loading MediaPipe Hand Landmarker…")
def load_landmarker(model_path: str):
    """Load the task model once without caching any patient image or result."""
    return create_hand_landmarker(model_path)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --navy:#071E33; --blue:#0B5FA5; --green:#18794E; --amber:#A15C00; --red:#B42318; }
        .stApp { background:#F7F9FC; color:var(--navy); }
        h1,h2,h3 { color:var(--navy); letter-spacing:-0.02em; }
        div[data-testid="stVerticalBlockBorderWrapper"] { background:white; border-color:#D8E1EA; border-radius:16px; }
        .status-card { background:white; border:1px solid #D8E1EA; border-radius:16px; padding:1rem 1.1rem; min-height:108px; }
        .eyebrow { color:#0B5FA5; font-weight:700; text-transform:uppercase; letter-spacing:.08em; font-size:.78rem; }
        .clinical-note { border-left:4px solid #0B5FA5; padding:.75rem 1rem; background:#EAF3FA; border-radius:0 10px 10px 0; }
        .demo-banner { border:1px solid #A15C00; background:#FFF7E6; color:#684000; padding:.8rem 1rem; border-radius:12px; font-weight:700; }
        .placeholder { border:2px dashed #9BAFBE; background:#F4F7FA; border-radius:14px; min-height:230px; display:flex; align-items:center; justify-content:center; text-align:center; color:#435565; padding:1.5rem; }
        .placeholder b { font-size:.9rem; line-height:1.25; }
        .placeholder small { display:block; margin-top:.45rem; font-size:.72rem; line-height:1.25; overflow-wrap:anywhere; word-break:break-word; }
        .stButton button, .stDownloadButton button { border-radius:10px; min-height:2.65rem; font-weight:650; }
        .stButton button:focus-visible, .stDownloadButton button:focus-visible { outline:3px solid #85BDEE; outline-offset:2px; }
        @media (max-width: 700px) { .status-card { min-height:auto; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def session_metadata() -> SessionMetadata:
    return SessionMetadata(
        participant_id=st.session_state.participant_id,
        hand_side=HandSide(st.session_state.hand_side),
        mirrored=bool(st.session_state.mirrored),
        label=st.session_state.session_label,
        notes=st.session_state.notes,
        session_id=st.session_state.session_id,
        timestamp_utc=st.session_state.session_timestamp_utc,
    )


def render_progress() -> None:
    labels = ["Welcome", "Upload", "Results & Export"]
    step = int(st.session_state.wizard_step)
    if step not in range(1, len(labels) + 1):
        step = 1
        st.session_state.wizard_step = step
    st.progress(step / len(labels), text=f"Step {step} of {len(labels)} — {labels[step - 1]}")


def render_demo_banner() -> None:
    if DEMO_MODE:
        st.markdown(
            '<div class="demo-banner">Demo data — not calculated from patient images.</div>',
            unsafe_allow_html=True,
        )


def welcome_screen() -> None:
    st.markdown('<p class="eyebrow">Clinician measurement support</p>', unsafe_allow_html=True)
    st.title("🖐️ HandROM")
    st.subheader("Estimate motion one finger at a time")
    st.caption(
        "Choose the hand and one target finger, capture side-view extension and flexion, then review and export the estimate."
    )
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            '<div class="status-card"><b>1. Choose</b><br>Pick the hand and case ID.</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            '<div class="status-card"><b>2. Capture</b><br>Add side-view photos for each finger.</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            '<div class="status-card"><b>3. Export</b><br>Review TAM and download the report.</div>',
            unsafe_allow_html=True,
        )
    if st.button("Start measurement", type="primary", use_container_width=True):
        st.session_state.wizard_step = 2
        st.rerun()


def _render_avatar_image(filename: str, caption: str) -> None:
    path = POSE_ASSET_DIR / filename
    if path.is_file():
        with Image.open(path) as source:
            guide = source.convert("RGBA")
        st.image(guide, caption=caption, width="stretch")
    else:
        st.markdown(
            '<div class="placeholder"><div><b>Side-view avatar guide unavailable.</b>'
            '<small>Missing<br>assets/pose_examples/<wbr>'
            f'{filename}</small></div></div>',
            unsafe_allow_html=True,
        )


def render_avatar_hand_guide(finger: str) -> None:
    st.markdown(f"### {finger.title()} finger avatar pose guide")
    with st.container(border=True):
        columns = st.columns(2)
        for column, (pose_label, filename) in zip(
            columns, SEPARATE_FINGER_POSE_ASSETS[finger], strict=True
        ):
            with column:
                st.markdown(f"**{pose_label}**")
                _render_avatar_image(
                    filename,
                    f"{finger.title()} finger — {pose_label.lower()}",
                )
    st.caption(
        f"Match the avatar's extension and flexion positions with the {finger} finger. "
        "Use the side-view capture checklist for the photo angle. "
        "Keep its MCP, PIP, DIP, and fingertip visible. "
        "صوّر إصبعًا واحدًا من الجانب مع إظهار جميع المفاصل وطرف الإصبع."
    )


def render_capture_instructions() -> None:
    with st.expander("Side-view capture and acceptance checklist", expanded=False):
        st.markdown(
            """
- Support the forearm, keep the wrist neutral, and leave the hand free to move.
- Place the camera perpendicular to the selected finger's bending plane, approximately 40–70 cm away.
- Let the hand occupy about 70% of the image. Keep the wrist and selected fingertip inside the frame.
- Use bright, even lighting and a plain contrasting background.
- Keep the same camera position for extension and flexion. Do not photograph from directly above the hand or point the finger toward the camera.
- For extension, straighten the selected finger actively without pressing it against a surface. Move the other fingers away or gently flex them.
- For flexion, bend the selected finger actively and keep the thumb and other fingers away. If the fingertip is hidden, rotate the hand or camera only 10–15°.

Reject or replace a photograph if another finger overlaps the selected finger, a required joint is hidden, the wrist or fingertip is cropped, the wrist is visibly flexed or extended, the target finger points toward the camera, a landmark overlay falls outside the visible target finger, or extension and flexion use substantially different camera angles.
"""
        )


def _view_instruction(finger: str) -> str:
    side = "thumb side" if finger in {"index", "middle"} else "little-finger side"
    return f"Photograph mainly from the {side}; reverse the physical hand orientation for the opposite hand."


def upload_screen() -> None:
    st.title("New measurement")
    st.caption("Capture one target finger at a time from the side.")
    if "hand_side_input_v2" not in st.session_state:
        st.session_state.hand_side_input_v2 = st.session_state.hand_side
    if "selected_finger_input_v4" not in st.session_state:
        st.session_state.selected_finger_input_v4 = st.session_state.selected_finger
    settings = st.columns(2)
    settings[0].radio(
        "Hand",
        [HandSide.LEFT.value, HandSide.RIGHT.value],
        horizontal=True,
        key="hand_side_input_v2",
    )
    settings[1].selectbox(
        "Finger to measure",
        FINGERS,
        format_func=str.title,
        key="selected_finger_input_v4",
    )
    st.session_state.hand_side = st.session_state.hand_side_input_v2
    st.session_state.mirrored = False
    st.session_state.selected_finger = st.session_state.selected_finger_input_v4
    st.caption(f"Session {st.session_state.session_id[:8]} · HandROM {APP_VERSION}")
    finger = st.session_state.selected_finger
    render_avatar_hand_guide(finger)
    render_capture_instructions()

    st.markdown(f"### {finger.title()} finger")
    st.caption(_view_instruction(finger))
    extension_col, flexion_col = st.columns(2)
    with extension_col, st.container(border=True):
        st.markdown("**Maximum active extension**")
        st.caption("Straighten actively; do not press against a table.")
        extension_files = st.file_uploader(
            f"Add 1–3 {finger} extension photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"{finger}_extension_uploader_v4",
        )
        st.caption(f"{len(extension_files)} of {MAX_IMAGES_PER_POSE} selected")
        if extension_files:
            st.image(extension_files, width=145)
    with flexion_col, st.container(border=True):
        st.markdown("**Maximum active flexion**")
        st.caption("Bend actively; keep the target chain unobstructed.")
        flexion_files = st.file_uploader(
            f"Add 1–3 {finger} flexion photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"{finger}_flexion_uploader_v4",
        )
        st.caption(f"{len(flexion_files)} of {MAX_IMAGES_PER_POSE} selected")
        if flexion_files:
            st.image(flexion_files, width=145)
    uploaded_by_finger: dict[str, dict[PoseType, list[object]]] = {
        finger: {
            PoseType.EXTENSION: list(extension_files),
            PoseType.FLEXION: list(flexion_files),
        }
    }

    st.caption("Bright, even light · Plain background · Same camera position · Two images per pose recommended")
    back, clear, analyze = st.columns([1, 1, 2])
    if back.button("Back", use_container_width=True):
        st.session_state.wizard_step = 1
        st.rerun()
    if clear.button("Clear Session", use_container_width=True):
        clear_session_state(st.session_state)
        st.rerun()
    button_label = "Load synthetic demo data" if DEMO_MODE else "Analyze Images"
    if analyze.button(button_label, type="primary", use_container_width=True):
        if DEMO_MODE:
            _load_demo()
            return
        for pose, label in (
            (PoseType.EXTENSION, "maximum-extension"),
            (PoseType.FLEXION, "maximum-flexion"),
        ):
            files = uploaded_by_finger[finger][pose]
            if not files:
                st.error(f"Add at least one {finger.title()} {label} image.")
                return
            if len(files) > MAX_IMAGES_PER_POSE:
                st.error(f"Upload no more than three {finger} {label} images.")
                return
        _analyze_uploads(uploaded_by_finger)


def _load_demo() -> None:
    st.session_state.analyses = create_demo_analyses((st.session_state.selected_finger,))
    st.session_state.demo_loaded = True
    recompute_results()
    st.session_state.wizard_step = 3
    st.rerun()


def _analyze_uploads(
    uploaded_by_finger: dict[str, dict[PoseType, list[object]]],
) -> None:
    if not Path(MODEL_PATH).is_file():
        st.error(
            f"MediaPipe model missing at {MODEL_PATH}. Download hand_landmarker.task as described in README.md, then try again."
        )
        return
    try:
        landmarker = load_landmarker(str(MODEL_PATH))
    except Exception as exc:  # The UI must survive model initialization failures.
        st.error(f"The MediaPipe model could not be loaded: {exc}")
        return
    analyses: list[ImageAnalysis] = []
    progress = st.progress(0, text="Validating and analyzing images…")
    jobs = [
        (item, pose, finger)
        for finger in uploaded_by_finger
        for pose in (PoseType.EXTENSION, PoseType.FLEXION)
        for item in uploaded_by_finger[finger][pose]
    ]
    for index, (uploaded, pose, finger) in enumerate(jobs, start=1):
        try:
            image = validate_uploaded_file(uploaded)
            analysis = analyze_validated_image(
                image,
                pose=pose,
                expected_side=HandSide(st.session_state.hand_side),
                mirrored=bool(st.session_state.mirrored),
                landmarker=landmarker,
                target_finger=finger,
            )
        except ImageValidationError as exc:
            analysis = ImageAnalysis(
                image_id=f"invalid-{index}",
                original_name=getattr(uploaded, "name", f"image-{index}"),
                pose=pose,
                valid=False,
                included=False,
                angles={},
                quality=_invalid_quality("image_validation_failed"),
                target_finger=finger,
                error=str(exc),
            )
        except Exception as exc:
            analysis = ImageAnalysis(
                image_id=f"error-{index}",
                original_name=getattr(uploaded, "name", f"image-{index}"),
                pose=pose,
                valid=False,
                included=False,
                angles={},
                quality=_invalid_quality("unexpected_analysis_error"),
                target_finger=finger,
                error=f"Unexpected analysis error: {exc}",
            )
        analyses.append(analysis)
        progress.progress(index / len(jobs), text=f"Analyzed {index} of {len(jobs)} images")
    st.session_state.analyses = analyses
    recompute_results()
    st.session_state.wizard_step = 3
    st.rerun()


def _invalid_quality(code: str):
    from handrom.data_models import QualityAssessment

    return QualityAssessment(
        level=QualityLevel.LOW, warnings=[code], checks={"analysis_succeeded": False}
    )


def recompute_results() -> None:
    analyses = st.session_state.analyses
    extension = aggregate_measurements(analyses, PoseType.EXTENSION)
    flexion = aggregate_measurements(analyses, PoseType.FLEXION)
    st.session_state.extension_aggregation = extension
    st.session_state.flexion_aggregation = flexion
    try:
        st.session_state.tam_results = calculate_tam(extension, flexion)
    except TamCalculationError:
        st.session_state.tam_results = None


def render_measurement_images() -> None:
    """Show every analyzed image with its three joint measurements."""
    st.subheader("Photo measurements")
    st.caption("MCP, PIP, and DIP values are shown on each annotated image and beneath it.")
    for pose in (PoseType.EXTENSION, PoseType.FLEXION):
        pose_items = [item for item in st.session_state.analyses if item.pose is pose]
        if not pose_items:
            continue
        st.markdown(f"### {pose.value.replace('_', ' ').title()}")
        image_columns = st.columns(min(2, len(pose_items)))
        for index, analysis in enumerate(pose_items):
            with image_columns[index % len(image_columns)], st.container(border=True):
                target = (analysis.target_finger or "whole hand").title()
                st.markdown(f"**{target} · {analysis.original_name}**")
                if analysis.annotated_png:
                    st.image(analysis.annotated_png, caption="Measured joints", width="stretch")
                elif analysis.original_rgb is not None:
                    st.image(analysis.original_rgb, caption="Original image", width="stretch")
                else:
                    st.info("Preview unavailable.")

                finger = analysis.target_finger
                angles = analysis.angles.get(finger, {}) if finger else {}
                measurement_columns = st.columns(3)
                for column, joint in zip(measurement_columns, ("mcp", "pip", "dip"), strict=True):
                    value = angles.get(joint)
                    display_value = f"{value:.1f}°" if value is not None else "—"
                    column.metric(joint.upper(), display_value)

                status = "Included" if analysis.included else "Not included"
                st.caption(f"{status} · {analysis.quality.level.value} quality")
                if analysis.error:
                    st.error(analysis.error)
                elif analysis.quality.warnings:
                    warning_text = ", ".join(
                        warning.replace("_", " ") for warning in analysis.quality.warnings
                    )
                    st.warning(warning_text.capitalize())


def _results_frame(tam_results: dict[str, object]) -> pd.DataFrame:
    rows = []
    for finger in FINGERS:
        if finger not in tam_results:
            continue
        result = tam_results[finger]
        rows.append(
            {
                "Total flexion": f"{result.total_flexion:.1f}°",
                "Total extension deficit": f"{result.total_extension_deficit:.1f}°",
                "Estimated TAM": f"{result.tam:.1f}°",
            }
        )
    return pd.DataFrame(rows)


def render_results_actions(tam_results: dict[str, object] | None) -> None:
    """Render the three actions that finish the results page."""
    pdf_bytes = b""
    pdf_error: str | None = "A complete result is required before creating a PDF report."
    if tam_results is not None:
        try:
            pdf_bytes = generate_pdf_report(
                session_metadata(),
                tam_results,
                st.session_state.analyses,
                include_annotated_images=True,
            )
            pdf_error = None
        except Exception as exc:
            pdf_error = str(exc)
            st.error(f"PDF-generation failure: {exc}")

    pdf, edit, clear = st.columns(3)
    pdf.download_button(
        "PDF report",
        pdf_bytes,
        "handrom_report.pdf",
        "application/pdf",
        disabled=pdf_error is not None,
        width="stretch",
    )
    if edit.button("Edit photos", width="stretch"):
        st.session_state.wizard_step = 2
        st.rerun()
    if clear.button("Clear session", type="primary", width="stretch"):
        clear_session_state(st.session_state)
        st.rerun()


def results_screen() -> None:
    st.title("Results")
    st.caption("Estimated total active motion (TAM) with per-photo MCP, PIP, and DIP measurements.")
    tam_results = st.session_state.tam_results
    if tam_results is None:
        st.error("A result could not be calculated. Review the measurements, then edit the photos.")
        render_measurement_images()
        render_results_actions(None)
        return
    extension = st.session_state.extension_aggregation
    flexion = st.session_state.flexion_aggregation
    overall = (
        QualityLevel.HIGH
        if extension.quality is flexion.quality is QualityLevel.HIGH
        else QualityLevel.MEDIUM
    )
    target_finger = next(iter(tam_results))
    st.subheader("Measurement KPIs")
    cols = st.columns(5)
    labels_values = [
        ("Selected hand", st.session_state.hand_side),
        ("Target finger", target_finger.title()),
        ("Included extension images", str(extension.valid_image_count)),
        ("Included flexion images", str(flexion.valid_image_count)),
        ("Overall measurement quality", overall.value),
    ]
    for column, (label, value) in zip(cols, labels_values, strict=True):
        with column:
            st.markdown(
                f'<div class="status-card"><small>{label}</small><br><b style="font-size:1.45rem">{value}</b></div>',
                unsafe_allow_html=True,
            )

    render_measurement_images()

    st.subheader("TAM summary")
    st.dataframe(_results_frame(tam_results), hide_index=True, width="stretch", height=84)
    st.caption("TAM = total flexion − total extension deficit.")

    render_results_actions(tam_results)


def main() -> None:
    initialize_session_state(st.session_state)
    inject_styles()
    render_demo_banner()
    render_progress()
    step = int(st.session_state.wizard_step)
    if step == 1:
        welcome_screen()
    elif step == 2:
        upload_screen()
    elif step == 3:
        results_screen()
    else:
        st.session_state.wizard_step = 1
        st.rerun()


if __name__ == "__main__":
    main()
