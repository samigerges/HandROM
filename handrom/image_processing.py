"""Safe, in-memory validation and preparation of uploaded photographs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from handrom.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    INFERENCE_MAX_DIMENSION,
    MAX_FILE_BYTES,
    MAX_PIXEL_COUNT,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)
from handrom.data_models import ValidatedImage


class ImageValidationError(ValueError):
    """Raised when an upload cannot be processed safely."""


def _resize_for_inference(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    longest = max(width, height)
    if longest <= INFERENCE_MAX_DIMENSION:
        return np.ascontiguousarray(rgb)
    scale = INFERENCE_MAX_DIMENSION / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray(rgb, mode="RGB")
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)


def validate_image_bytes(
    data: bytes,
    *,
    filename: str,
    mime_type: str,
) -> ValidatedImage:
    """Validate an upload, remove metadata, orient it, and return RGB arrays."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ImageValidationError("The file exceeds the 15 MB upload limit.")
    if mime_type.lower() not in ALLOWED_MIME_TYPES:
        raise ImageValidationError("Only JPG, JPEG, and PNG images are supported.")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("The filename must end in .jpg, .jpeg, or .png.")

    previous_truncated_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(BytesIO(data)) as image:
            if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
                raise ImageValidationError("Animated images are not supported.")
            detected_format = (image.format or "").upper()
            if detected_format not in {"JPEG", "PNG"}:
                raise ImageValidationError("The file contents are not a JPG or PNG image.")
            width, height = image.size
            if width * height > MAX_PIXEL_COUNT:
                raise ImageValidationError("The image exceeds the 25-megapixel safety limit.")
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                raise ImageValidationError(
                    f"Image resolution must be at least {MIN_IMAGE_WIDTH}×{MIN_IMAGE_HEIGHT}."
                )
            oriented = ImageOps.exif_transpose(image)
            rgb_image = oriented.convert("RGB")
            # Copy pixel data into a new image/array so EXIF and other metadata are discarded.
            rgb = np.asarray(Image.fromarray(np.asarray(rgb_image)), dtype=np.uint8).copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The image is corrupted, truncated, or malformed.") from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated_setting

    final_height, final_width = rgb.shape[:2]
    return ValidatedImage(
        image_id=str(uuid4()),
        original_name=Path(filename).name,
        mime_type=mime_type.lower(),
        rgb=rgb,
        inference_rgb=_resize_for_inference(rgb),
        width=final_width,
        height=final_height,
        format=detected_format,
    )


def validate_uploaded_file(uploaded_file: object) -> ValidatedImage:
    """Validate a Streamlit UploadedFile-like object without trusting its filename."""
    try:
        data = uploaded_file.getvalue()  # type: ignore[attr-defined]
        filename = str(uploaded_file.name)  # type: ignore[attr-defined]
        mime_type = str(uploaded_file.type)  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ImageValidationError("The uploaded object is not a readable image file.") from exc
    return validate_image_bytes(data, filename=filename, mime_type=mime_type)
