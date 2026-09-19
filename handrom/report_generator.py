"""ReportLab PDF generation for HandROM sessions."""

from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from handrom.config import MEDICAL_DISCLAIMER
from handrom.data_models import ImageAnalysis, SessionMetadata, TamResult
from handrom.landmark_mapping import FINGERS


def generate_pdf_report(
    session: SessionMetadata,
    tam_results: dict[str, TamResult],
    analyses: list[ImageAnalysis],
    *,
    include_annotated_images: bool = False,
) -> bytes:
    """Generate a report without patient images unless explicitly requested."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"HandROM report {session.session_id}",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "SmallBody", parent=styles["BodyText"], fontSize=8.5, leading=11, alignment=TA_LEFT
        )
    )
    story: list[object] = [
        Paragraph("HandROM", styles["Title"]),
        Paragraph("Experimental finger range-of-motion estimate", styles["Heading2"]),
        Spacer(1, 3 * mm),
        _metadata_table(session),
        Spacer(1, 5 * mm),
        Paragraph("Estimated Total Active Motion", styles["Heading2"]),
        _tam_table(tam_results),
        Spacer(1, 5 * mm),
        Paragraph("Quality and warnings", styles["Heading2"]),
        _quality_table(analyses),
        Spacer(1, 5 * mm),
        Paragraph("Method summary", styles["Heading2"]),
        Paragraph(
            "Each photograph targets one unobstructed finger from the side. MCP, PIP, and DIP estimates use the target finger's 2D image-plane MediaPipe landmarks; wrist-to-MCP is the visible metacarpal reference for MCP flexion. "
            "Multiple included images are aggregated with the median. TAM is total maximum flexion minus total extension deficit.",
            styles["SmallBody"],
        ),
        Spacer(1, 3 * mm),
        Paragraph("Known limitations", styles["Heading2"]),
        Paragraph(
            "MediaPipe estimates surface landmarks rather than directly measuring bones. "
            "Perspective, camera misalignment, overlap, hidden-landmark inference, severe deformity, and lighting can affect estimates. "
            "Hyperextension and thumb TAM are unsupported. One image cannot establish repeatability. HandROM is not clinically validated and does not provide a diagnosis.",
            styles["SmallBody"],
        ),
        Spacer(1, 4 * mm),
        Paragraph(MEDICAL_DISCLAIMER, styles["SmallBody"]),
    ]
    if include_annotated_images:
        image_flowables: list[object] = []
        for analysis in analyses:
            if not analysis.annotated_png:
                continue
            preview = Image(
                BytesIO(analysis.annotated_png), width=78 * mm, height=58 * mm, kind="proportional"
            )
            image_flowables.append(
                KeepTogether(
                    [
                        Paragraph(
                            f"{(analysis.target_finger or 'Whole hand').title()} — {analysis.pose.value.replace('_', ' ').title()} — {analysis.original_name}",
                            styles["SmallBody"],
                        ),
                        preview,
                        Spacer(1, 2 * mm),
                    ]
                )
            )
        if image_flowables:
            story.extend(
                [
                    Spacer(1, 5 * mm),
                    Paragraph("Annotated images", styles["Heading2"]),
                    *image_flowables,
                ]
            )
    document.build(story)
    return output.getvalue()


def _metadata_table(session: SessionMetadata) -> Table:
    data = [
        ["Session ID", session.session_id],
        ["Anonymous participant ID", session.participant_id],
        ["Timestamp (UTC)", session.timestamp_utc],
        ["Selected hand", session.hand_side.value],
        ["Application version", session.application_version],
    ]
    table = Table(data, colWidths=[48 * mm, 120 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF3FA")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#071E33")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C8D8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _tam_table(results: dict[str, TamResult]) -> Table:
    data = [["Finger", "Total flexion", "Extension deficit", "Estimated TAM", "Quality"]]
    for finger in FINGERS:
        if finger not in results:
            continue
        result = results[finger]
        data.append(
            [
                finger.title(),
                f"{result.total_flexion:.1f}°",
                f"{result.total_extension_deficit:.1f}°",
                f"{result.tam:.1f}°",
                result.quality.value,
            ]
        )
    table = Table(data, repeatRows=1, colWidths=[34 * mm] * 5)
    table.setStyle(_table_style())
    return table


def _quality_table(analyses: list[ImageAnalysis]) -> Table:
    data = [["Image", "Finger", "Pose", "Status", "Image quality", "Warnings"]]
    for analysis in analyses:
        data.append(
            [
                analysis.original_name,
                (analysis.target_finger or "Whole hand").title(),
                analysis.pose.value.replace("_", " ").title(),
                "Included" if analysis.included else "Excluded",
                analysis.quality.level.value,
                ", ".join(analysis.quality.warnings) or "None",
            ]
        )
    table = Table(
        data,
        repeatRows=1,
        colWidths=[31 * mm, 22 * mm, 31 * mm, 22 * mm, 25 * mm, 39 * mm],
    )
    table.setStyle(_table_style())
    return table


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B5FA5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C8D8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FB")]),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
    )
