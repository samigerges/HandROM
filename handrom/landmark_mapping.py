"""MediaPipe hand landmark indices used by HandROM."""

from __future__ import annotations

FINGER_LANDMARKS: dict[str, dict[str, int]] = {
    "index": {"mcp": 5, "pip": 6, "dip": 7, "tip": 8},
    "middle": {"mcp": 9, "pip": 10, "dip": 11, "tip": 12},
    "ring": {"mcp": 13, "pip": 14, "dip": 15, "tip": 16},
    "little": {"mcp": 17, "pip": 18, "dip": 19, "tip": 20},
}

FINGERS: tuple[str, ...] = tuple(FINGER_LANDMARKS)
JOINTS: tuple[str, ...] = ("mcp", "pip", "dip")


def joint_triplets(finger: str) -> dict[str, tuple[int, int, int]]:
    """Return proximal, joint, distal indices for PIP and DIP."""
    mapping = FINGER_LANDMARKS[finger]
    return {
        "pip": (mapping["mcp"], mapping["pip"], mapping["dip"]),
        "dip": (mapping["pip"], mapping["dip"], mapping["tip"]),
    }
