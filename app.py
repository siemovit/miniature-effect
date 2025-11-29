import io
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st


@dataclass
class MiniatureSettings:
    line1_height: float = 0.25
    margin1_height: float = 0.35
    margin2_height: float = 0.65
    line2_height: float = 0.75


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
    line1_height: float = 0.25,
    line2_height: float = 0.75,
    margin_1_height: float = 0.35,
    margin_2_height: float = 0.65,
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
    guides = [
        (settings.line1_height, (255, 153, 0), 12),
        (settings.line2_height, (255, 153, 0), 12),
        (settings.margin1_height, (255, 222, 89), 8),
        (settings.margin2_height, (255, 222, 89), 8),
    ]

    dash = 16
    gap = 10
    for relative_height, color, thickness in guides:
        y = int(relative_height * height)
        x = 0
        while x < width:
            draw.line([(x, y), (min(x + dash, width), y)], fill=color, width=thickness)
            x += dash + gap
    return image


def clamp_settings(settings: MiniatureSettings) -> MiniatureSettings:
    """Ensure guide lines remain ordered."""
    ordered = sorted(
        [
            ("line1_height", settings.line1_height),
            ("margin1_height", settings.margin1_height),
            ("margin2_height", settings.margin2_height),
            ("line2_height", settings.line2_height),
        ],
        key=lambda item: item[1],
    )
    values = {name: value for name, value in ordered}
    return MiniatureSettings(
        line1_height=values["line1_height"],
        margin1_height=values["margin1_height"],
        margin2_height=values["margin2_height"],
        line2_height=values["line2_height"],
    )


def sidebar_controls() -> Tuple[int, int, str, MiniatureSettings, bool]:
    st.sidebar.header("Miniature controls")
    blur1 = st.sidebar.slider("Orange region blur", min_value=0, max_value=80, value=40, step=1)
    blur2 = st.sidebar.slider("Yellow region blur", min_value=0, max_value=80, value=25, step=1)
    blur_shape = st.sidebar.selectbox("Blur shape", ["box", "gaussian", "circular"], index=2)

    st.sidebar.subheader("Guide positions")
    line1 = st.sidebar.slider("Top orange line (%)", min_value=5, max_value=45, value=25, step=1) / 100
    margin1 = st.sidebar.slider("Top yellow line (%)", min_value=15, max_value=55, value=35, step=1) / 100
    margin2 = st.sidebar.slider("Bottom yellow line (%)", min_value=45, max_value=85, value=65, step=1) / 100
    line2 = st.sidebar.slider("Bottom orange line (%)", min_value=55, max_value=95, value=75, step=1) / 100

    show_guides = st.sidebar.checkbox("Show guide overlay", value=True)
    settings = clamp_settings(
        MiniatureSettings(
            line1_height=line1, margin1_height=margin1, margin2_height=margin2, line2_height=line2
        )
    )
    return blur1, blur2, blur_shape, settings, show_guides


def main() -> None:
    st.set_page_config(page_title="Miniature Effect Tool", layout="wide")
    st.title("Miniature Effect Playground")
    st.caption("Upload an image and experiment with DxO-like miniature blur bands.")

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is None:
        st.info("Select an image to get started.")
        return

    pil_image = Image.open(uploaded_image).convert("RGB")
    base_array = np.array(pil_image)

    blur1, blur2, blur_shape, settings, show_guides = sidebar_controls()

    if "processed_image" not in st.session_state:
        st.session_state["processed_image"] = base_array

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preview")
        preview_image = pil_image.copy()
        if show_guides:
            preview_image = draw_guides(preview_image, settings)
        st.image(preview_image, use_column_width=True)

    with col2:
        st.subheader("Result")
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

        st.image(st.session_state["processed_image"], use_column_width=True)

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
