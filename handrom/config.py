"""Application-wide configuration values."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "HandROM"
APP_VERSION = "0.2.2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
POSE_ASSET_DIR = PROJECT_ROOT / "assets" / "pose_examples"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_PIXEL_COUNT = 25_000_000
MIN_IMAGE_WIDTH = 640
MIN_IMAGE_HEIGHT = 480
INFERENCE_MAX_DIMENSION = 1600
MAX_IMAGES_PER_POSE = 3

ANGLE_NOISE_FLOOR_DEG = 3.0
EPSILON = 1e-8
BLUR_LAPLACIAN_THRESHOLD = 70.0
BRIGHTNESS_DARK_THRESHOLD = 45.0
BRIGHTNESS_BRIGHT_THRESHOLD = 220.0
LANDMARK_EDGE_MARGIN = 0.025
MIN_HAND_SPAN_RATIO = 0.22
HANDEDNESS_CONFIDENCE_THRESHOLD = 0.70
CONSISTENCY_HIGH_MAX_STD_DEG = 5.0
CONSISTENCY_MEDIUM_MAX_STD_DEG = 10.0

MEDICAL_DISCLAIMER = (
    "Experimental range-of-motion estimate based on uploaded photographs. "
    "This application is not a medical diagnosis and is not a replacement for "
    "assessment by a qualified hand therapist, surgeon, or clinician using a "
    "manual goniometer."
)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
