import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .models import MiniatureSettings


def kernel_size_from_blur_level(level: int) -> int:
    """Map slider blur level (0-100) to a kernel size (odd integer)."""
    if level <= 0:
        return 1
    mapped = max(1, int(level / 2))
    size = mapped * 2 + 1
    return min(size, 101)


def apply_blur(image: np.ndarray, level: int, shape: str) -> np.ndarray:
    kernel_size = kernel_size_from_blur_level(level)
    if kernel_size <= 1:
        return image.copy()

    if shape == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    if shape == "circular":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(np.float32)
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel)

    return cv2.blur(image, (kernel_size, kernel_size))


def render_image(
    image: np.ndarray,
    blur_level1: int,
    blur_level2: int,
    blur_shape: str,
    line1_height: float,
    line2_height: float,
    margin_1_height: float,
    margin_2_height: float,
) -> np.ndarray:
    """Apply miniature blur to specified horizontal bands."""
    height = image.shape[0]
    bounds = np.array(
        [
            0,
            int(height * line1_height),
            int(height * margin_1_height),
            int(height * margin_2_height),
            int(height * line2_height),
            height,
        ]
    )
    bounds = np.clip(bounds, 0, height)
    if not np.all(np.diff(bounds) >= 0):
        bounds = np.sort(bounds)

    blur1 = apply_blur(image, blur_level1, blur_shape)
    blur2 = apply_blur(image, blur_level2, blur_shape)

    final_image = np.zeros_like(image)
    b0, b1, b2, b3, b4, b5 = bounds
    final_image[b0:b1] = blur1[b0:b1]
    final_image[b1:b2] = blur2[b1:b2]
    final_image[b2:b3] = image[b2:b3]
    final_image[b3:b4] = blur2[b3:b4]
    final_image[b4:b5] = blur1[b4:b5]
    return final_image


def draw_guides(image: Image.Image, settings: MiniatureSettings) -> Image.Image:
    """Overlay dashed horizontal guide lines that mark miniature bands."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    guides: Tuple[Tuple[float, Tuple[int, int, int], int], ...] = (
        (settings.line1_height, (255, 153, 0), 4),
        (settings.line2_height, (255, 153, 0), 4),
        (settings.margin1_height, (255, 222, 89), 2),
        (settings.margin2_height, (255, 222, 89), 2),
    )

    dash = 16
    gap = 10
    for relative_height, color, thickness in guides:
        y = int(relative_height * height)
        x = 0
        while x < width:
            draw.line([(x, y), (min(x + dash, width), y)], fill=color, width=thickness)
            x += dash + gap
    return image


def image_to_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()
