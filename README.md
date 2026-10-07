# GI Image Segmentation with U-Net

An educational Streamlit dashboard that generates a binary segmentation mask and overlay from an uploaded image using a U-Net architecture.

## What is included

- Resizes RGB inputs to 256 × 256 and normalizes pixel values.
- Adjustable mask threshold, overlay opacity, and display size.
- Shows segmented pixels, coverage, and model output scores.
- Downloads masks and overlays as PNG files.
- Includes a Kvasir-SEG training notebook.

## Getting started

Use Python compatible with the pinned TensorFlow dependencies.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

Place compatible trained weights at `segmentation.weights.h5` in the root. Upload a JPG, JPEG, or PNG image. Training requires the notebook dependencies, including pandas and scikit-learn, in addition to the dashboard requirements.

## Repository guide

- `Gastrointestinal_img_segmentation.ipynb`
- `Kvasir-SEG/`
- `README.md`
- `main.py`
- `requirements.txt`
- `segmentation.weights.h5`

## Limitations and reproducibility

If weights are absent, the current app runs an untrained model and warns in the sidebar; its output is not meaningful segmentation. The training notebook contains machine-specific paths and a later `model.keras` loading cell that needs adaptation. Model scores and coverage are not validated diagnostic confidence. This project is for academic demonstration, not clinical use.
