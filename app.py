import io
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st


MARGIN_OFFSET = 0.1
MIN_FOCUS_GAP = 0.05


@dataclass
class MiniatureSettings:
    line1_height: float = 0.35
    margin1_height: float = 0.25
    margin2_height: float = 0.75
    line2_height: float = 0.65


def build_settings(top_orange: float, bottom_orange: float, offset: float = MARGIN_OFFSET) -> MiniatureSettings:
    """Ensure orange lines define the focus band and yellow lines become fixed margins."""
    top_orange = np.clip(top_orange, 0.0, 1.0)
    bottom_orange = np.clip(bottom_orange, 0.0, 1.0)
    if bottom_orange < top_orange:
        top_orange, bottom_orange = bottom_orange, top_orange
    if bottom_orange - top_orange < MIN_FOCUS_GAP:
        bottom_orange = min(1.0, top_orange + MIN_FOCUS_GAP)

    top_yellow = max(0.0, top_orange - offset)
    bottom_yellow = min(1.0, bottom_orange + offset)
    return MiniatureSettings(
        line1_height=top_orange,
        margin1_height=top_yellow,
        margin2_height=bottom_yellow,
        line2_height=bottom_orange,
    )


def kernel_size_from_blur_level(level: int) -> int:
    """Map slider blur level (0-100) to a kernel size (odd integer)."""
    if level <= 0:
        return 1
    mapped = max(1, int(level / 2))
    size = mapped * 2 + 1
    return min(size, 101)  # avoid overly large kernels


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
    blur_shape: str = "box",
    line1_height: float = 0.35,
    line2_height: float = 0.65,
    margin_1_height: float = 0.25,
    margin_2_height: float = 0.75,
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
    """Overlay horizontal guide lines with solid white for outer and dashed white for inner."""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def _draw_line(relative_height: float, dashed: bool, thickness: int) -> None:
        y = int(relative_height * height)
        if dashed:
            dash = 18
            gap = 12
            x = 0
            while x < width:
                draw.line([(x, y), (min(x + dash, width), y)], fill=(255, 255, 255), width=thickness)
                x += dash + gap
        else:
            draw.line([(0, y), (width, y)], fill=(255, 255, 255), width=thickness)

    _draw_line(settings.margin1_height, dashed=True, thickness=1)
    _draw_line(settings.line1_height, dashed=False, thickness=3)
    _draw_line(settings.line2_height, dashed=False, thickness=3)
    _draw_line(settings.margin2_height, dashed=True, thickness=1)
    return image


def sidebar_controls() -> Tuple[int, int, str, MiniatureSettings]:
    st.sidebar.header("Miniature controls")
    blur1 = st.sidebar.slider("Outer blur", min_value=0, max_value=80, value=40, step=1)
    blur2 = st.sidebar.slider("Inner blur", min_value=0, max_value=80, value=25, step=1)
    blur_shape = st.sidebar.selectbox("Blur shape", ["box", "gaussian", "circular"], index=2)

    st.sidebar.subheader("Guide positions (%)")
    outer_top, outer_bottom = st.sidebar.slider(
        "Focus band", min_value=5, max_value=95, value=(35, 65), step=1
    )
    settings = build_settings(outer_top / 100.0, outer_bottom / 100.0)
    return blur1, blur2, blur_shape, settings


def main() -> None:
    st.set_page_config(page_title="Miniature Effect Tool", layout="wide")
    st.title("Miniature Effect Playground")
    st.caption("Upload an image and adjust the guides directly on a single preview.")

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is None:
        st.info("Select an image to get started.")
        return

    pil_image = Image.open(uploaded_image).convert("RGB")
    base_array = np.array(pil_image)

    blur1, blur2, blur_shape, settings = sidebar_controls()

    if "processed_image" not in st.session_state:
        st.session_state["processed_image"] = base_array

    effect_enabled = st.toggle("Miniature effect", value=True)
    show_guides = False
    if effect_enabled:
        show_guides = st.checkbox("Display guides", value=True)

    if effect_enabled:
        st.subheader("Preview" if show_guides else "Preview (guides hidden)")
        if st.button("Save changes"):
            processed = render_image(
                base_array,
                blur_level1=blur1,
                blur_level2=blur2,
                blur_shape=blur_shape,
                line1_height=settings.line1_height,
                line2_height=settings.line2_height,
                margin_1_height=settings.margin1_height,
                margin_2_height=settings.margin2_height,
            )
            st.session_state["processed_image"] = processed

        preview_image = Image.fromarray(st.session_state["processed_image"]).copy()
        if show_guides:
            preview_image = draw_guides(preview_image, settings)
        st.image(preview_image, use_column_width=True)
    else:
        st.subheader("Original image")
        st.image(pil_image, use_column_width=True)

    st.download_button(
        label="Download processed image",
        data=image_to_bytes(st.session_state["processed_image"]),
        file_name="miniature_effect.png",
        mime="image/png",
    )


def image_to_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


if __name__ == "__main__":
    main()
