"""Non-destructive annotation of uploaded photographs."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from handrom.config import HAND_CONNECTIONS
from handrom.data_models import HandSide, PoseType, QualityLevel
from handrom.landmark_mapping import FINGER_LANDMARKS

QUALITY_COLORS = {
    QualityLevel.HIGH: "#18794E",
    QualityLevel.MEDIUM: "#A15C00",
    QualityLevel.LOW: "#B42318",
}


def annotate_image(
    rgb: np.ndarray,
    normalized_landmarks: np.ndarray,
    angles: dict[str, dict[str, float | None]],
    *,
    side: HandSide,
    pose: PoseType,
    quality: QualityLevel,
    target_finger: str | None = None,
) -> bytes:
    """Return an annotated PNG while leaving the original RGB array untouched."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy(), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, round(min(image.size) / 55)))
    small = ImageFont.load_default(size=max(10, round(min(image.size) / 70)))
    width, height = image.size
    points = [
        (round(float(point[0]) * width), round(float(point[1]) * height))
        for point in normalized_landmarks
    ]

    if target_finger is not None:
        mapping = FINGER_LANDMARKS[target_finger]
        connections = (
            (0, mapping["mcp"]),
            (mapping["mcp"], mapping["pip"]),
            (mapping["pip"], mapping["dip"]),
            (mapping["dip"], mapping["tip"]),
        )
        visible_point_indices = {index for connection in connections for index in connection}
        finger_mappings = {target_finger: mapping}
    else:
        connections = HAND_CONNECTIONS
        visible_point_indices = set(range(len(points)))
        finger_mappings = FINGER_LANDMARKS

    for start, end in connections:
        draw.line((points[start], points[end]), fill="#1D6FC0", width=max(2, width // 350))
    radius = max(3, width // 180)
    for point_index in sorted(visible_point_indices):
        x, y = points[point_index]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#FFFFFF",
            outline="#0B3558",
            width=2,
        )

    joint_offsets = {"mcp": (-10, -26), "pip": (8, -20), "dip": (8, 4)}
    for finger, mapping in finger_mappings.items():
        for joint in ("mcp", "pip", "dip"):
            x, y = points[mapping[joint]]
            value = angles.get(finger, {}).get(joint)
            label = f"{finger[0].upper()} {joint.upper()}"
            if value is not None:
                label += f" {value:.1f}°"
            ox, oy = joint_offsets[joint]
            draw.text(
                (x + ox, y + oy),
                label,
                font=small,
                fill="#071E33",
                stroke_width=2,
                stroke_fill="#FFFFFF",
            )

    banner_height = max(40, height // 14)
    draw.rectangle((0, 0, width, banner_height), fill=QUALITY_COLORS[quality])
    target_label = f" · {target_finger.title()} finger" if target_finger else ""
    header = f"{side.value} hand{target_label} · {pose.value.replace('_', ' ').title()} · {quality.value} quality"
    draw.text((14, 10), header, font=font, fill="white")
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
