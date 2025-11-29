# miniature-effect

Miniature / tilt-shift effect playground built with Streamlit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Upload an image, tune the blur for the outer (orange) and margin (yellow) regions, and position the orange guides (the yellow margins follow automatically). Toggle the miniature effect on to reveal the single preview, use the **Display guides** checkbox to show or hide the white overlays, click **Save changes** to re-render, or toggle the effect off anytime to view the original image. Use the download button to export the processed image.
