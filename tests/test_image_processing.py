from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import handrom.image_processing as image_processing
from handrom.image_processing import ImageValidationError, validate_image_bytes


def image_bytes(
    mode: str = "RGB", size: tuple[int, int] = (800, 640), fmt: str = "PNG", *, exif=None
) -> bytes:
    color = 128 if mode == "L" else (80, 130, 180)
    image = Image.new(mode, size, color)
    output = BytesIO()
    image.save(output, format=fmt, exif=exif)
    return output.getvalue()


def test_rgb_image() -> None:
    result = validate_image_bytes(image_bytes(), filename="photo.png", mime_type="image/png")
    assert result.rgb.shape == (640, 800, 3)
    assert result.rgb.dtype == np.uint8


def test_grayscale_converted_to_rgb() -> None:
    result = validate_image_bytes(image_bytes("L"), filename="scan.png", mime_type="image/png")
    assert result.rgb.shape == (640, 800, 3)
    np.testing.assert_array_equal(result.rgb[:, :, 0], result.rgb[:, :, 1])


def test_exif_rotation_corrected() -> None:
    exif = Image.Exif()
    exif[274] = 6
    data = image_bytes(size=(640, 800), fmt="JPEG", exif=exif)
    result = validate_image_bytes(data, filename="rotated.jpg", mime_type="image/jpeg")
    assert (result.width, result.height) == (800, 640)


@pytest.mark.parametrize(
    ("filename", "mime"),
    [("photo.gif", "image/gif"), ("photo.heic", "image/heic"), ("photo.pdf", "application/pdf")],
)
def test_unsupported_file(filename: str, mime: str) -> None:
    with pytest.raises(ImageValidationError, match="Only JPG"):
        validate_image_bytes(b"not-an-image", filename=filename, mime_type=mime)


def test_oversized_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_processing, "MAX_FILE_BYTES", 10)
    with pytest.raises(ImageValidationError, match="15 MB"):
        validate_image_bytes(b"x" * 11, filename="large.png", mime_type="image/png")


def test_excessive_pixel_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_processing, "MAX_PIXEL_COUNT", 100_000)
    with pytest.raises(ImageValidationError, match="25-megapixel"):
        validate_image_bytes(image_bytes(), filename="large.png", mime_type="image/png")


def test_corrupted_image() -> None:
    with pytest.raises(ImageValidationError, match="corrupted"):
        validate_image_bytes(b"\x89PNGbroken", filename="broken.png", mime_type="image/png")


def test_small_image() -> None:
    with pytest.raises(ImageValidationError, match="at least"):
        validate_image_bytes(
            image_bytes(size=(639, 479)), filename="small.png", mime_type="image/png"
        )


def test_aspect_ratio_preserved_on_resize() -> None:
    result = validate_image_bytes(
        image_bytes(size=(2400, 1200)), filename="wide.png", mime_type="image/png"
    )
    assert result.inference_rgb.shape[:2] == (800, 1600)
