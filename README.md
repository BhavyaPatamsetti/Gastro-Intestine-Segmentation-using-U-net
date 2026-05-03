# 🧠 GI Segmentation AI Dashboard  
### *Medical Image Segmentation using U-Net + Streamlit*

> Upload → Segment → Visualize → Analyze → Download

---

## 🚀 Overview

**GI Segmentation AI Dashboard** is an AI-powered medical imaging tool that uses a **U-Net deep learning model** to segment gastrointestinal structures from input images.

It transforms a simple image into:
- 📊 Segmentation masks  
- 🎯 Overlay visualizations  
- 📈 Quantitative metrics  
- 📥 Downloadable outputs  

All through a **clean, premium dashboard UI** built with Streamlit.

---

## ✨ Features

### 🧠 1. Deep Learning Model (U-Net)
- Encoder–decoder architecture
- Pixel-level segmentation
- Supports medical image structures

---

### 🖼️ 2. Smart Image Processing
- Automatic resizing (256×256)
- Normalization for model input
- Real-time inference

---

### 🎨 3. Visualization Dashboard
- Original Image
- Predicted Mask
- Overlay (highlighted segmentation)

---

### 🎛️ 4. Interactive Controls
- Mask threshold tuning
- Overlay opacity adjustment
- Dynamic image sizing

---

### 📊 5. Analytics & Metrics
- Mask coverage (%)
- Segmented pixel count
- Average prediction confidence
- Max confidence score

---

### 🧾 6. AI-style Interpretation
- Human-readable insights
- Explains segmentation quality

---

### 📥 7. Export Options
- Download mask image
- Download overlay image

---

## 📂 Project Structure

code/
├── main.py  
├── requirements.txt  
├── segmentation.weights.h5  
└── .venv/  

---

## ⚙️ Installation & Setup

```bash
git clone "repo name"
cd folder_name

conda deactivate
rm -rf .venv

/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m streamlit run main.py
```

---

## 🛠️ Tech Stack
- Streamlit  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- PIL  

---

## ⚠️ Disclaimer
Educational use only. Not for medical diagnosis.

---

## 👩‍💻 Author
Radhi Sri Bhavya Patamsetti  
GitHub: https://github.com/BhavyaPatamsetti  
