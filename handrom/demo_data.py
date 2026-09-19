"""Synthetic, clearly labeled demo measurements; never patient photographs."""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from handrom.data_models import (
    HandSide,
    ImageAnalysis,
    PoseType,
    QualityAssessment,
    QualityLevel,
)
from handrom.landmark_mapping import FINGERS


def _demo_png(
    pose: PoseType,
    sequence: int,
    finger: str,
    angles: dict[str, float],
) -> bytes:
    image = Image.new("RGB", (960, 720), "#EAF3FA")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=58)
    subtitle_font = ImageFont.load_default(size=32)
    body_font = ImageFont.load_default(size=22)
    draw.rounded_rectangle(
        (90, 130, 870, 590), radius=38, fill="#FFFFFF", outline="#7BA7CC", width=4
    )
    draw.text((170, 260), "DEMO DATA", fill="#0B5FA5", font=title_font)
    draw.text((170, 350), pose.value.replace("_", " ").title(), fill="#071E33", font=subtitle_font)
    draw.text(
        (170, 410),
        f"{finger.title()} finger · fixture {sequence} — no patient image",
        fill="#556575",
        font=body_font,
    )
    measurement_text = "   ".join(
        f"{joint.upper()} {angles[joint]:.1f}°" for joint in ("mcp", "pip", "dip")
    )
    draw.text((170, 465), measurement_text, fill="#0B5FA5", font=body_font)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def create_demo_analyses(target_fingers: tuple[str, ...] = FINGERS) -> list[ImageAnalysis]:
    extension_sets = [
        {"mcp": 1.5, "pip": 10.0, "dip": 5.0},
        {"mcp": 2.0, "pip": 9.0, "dip": 5.5},
    ]
    flexion_sets = [
        {"mcp": 90.0, "pip": 85.0, "dip": 55.0},
        {"mcp": 88.0, "pip": 87.0, "dip": 54.0},
    ]
    analyses: list[ImageAnalysis] = []
    for pose, sets in ((PoseType.EXTENSION, extension_sets), (PoseType.FLEXION, flexion_sets)):
        for sequence, base in enumerate(sets, start=1):
            for finger in target_fingers:
                finger_index = FINGERS.index(finger)
                angles = {
                    finger: {
                        joint: value + finger_index * 0.5 for joint, value in base.items()
                    }
                }
                png = _demo_png(pose, sequence, finger, angles[finger])
                analyses.append(
                    ImageAnalysis(
                        image_id=f"demo-{finger}-{pose.value}-{uuid4().hex[:8]}",
                        original_name=f"demo_{finger}_{pose.value}_{sequence}.png",
                        pose=pose,
                        valid=True,
                        included=True,
                        angles=angles,
                        quality=QualityAssessment(
                            level=QualityLevel.HIGH,
                            warnings=["demo_synthetic_measurements"],
                            checks={"synthetic_fixture": True},
                            metrics={"blur_variance": 999.0, "mean_brightness": 160.0},
                        ),
                        target_finger=finger,
                        capture_confirmed=True,
                        detected_side=HandSide.RIGHT,
                        detection_confidence=0.99,
                        resolution=(960, 720),
                        annotated_png=png,
                        side_confirmed=True,
                    )
                )
    return analyses
