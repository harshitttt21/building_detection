# Building Detection using Satellite Images 🛰️🏢

A deep learning-based building extraction system that detects and segments buildings from satellite imagery using the YOLOv8 segmentation model.

---

# Project Overview

This project focuses on automatic building detection from aerial and satellite images using computer vision and deep learning techniques.

The system uses the **YOLOv8-SEG** architecture to:
- Detect buildings
- Generate segmentation masks
- Process satellite imagery efficiently
- Visualize prediction results

The project combines:
- A Flask backend
- Deep learning inference
- Frontend image upload interface
- Satellite image analysis

---

# Features

- Building detection from satellite images
- Semantic segmentation using YOLOv8-SEG
- Flask backend integration
- Frontend for image upload and visualization
- High detection accuracy
- Real-time inference support
- Works on complex aerial imagery

---

# Technologies Used

## Backend
- Python
- Flask
- YOLOv8
- OpenCV
- PyTorch

## Frontend
- HTML
- CSS
- JavaScript

## Deep Learning
- Ultralytics YOLOv8
- Computer Vision
- Image Segmentation

---

# Dataset

The project uses the **Inria Aerial Image Labeling Dataset**, which contains high-resolution satellite images with building annotations.

Dataset characteristics:
- Urban and rural scenes
- High-resolution aerial imagery
- Building segmentation labels
- Diverse environmental conditions

---

# Methodology

## 1. Data Preprocessing

Satellite images were preprocessed using:
- Image normalization
- Resizing
- Data augmentation
  - Rotation
  - Flipping
  - Scaling

These steps improved model robustness and generalization.

---

## 2. Model Training

The YOLOv8 segmentation architecture was trained on the satellite image dataset.

The model learns:
- Building boundaries
- Spatial features
- Complex image patterns
- Segmentation masks

---

## 3. Building Detection

The trained model predicts:
- Building locations
- Segmentation masks
- Object boundaries

The system performs well even on dense urban satellite imagery.

---

# Performance Metrics

| Metric | Score |
|---|---|
| mAP@0.5 | 0.948 |
| Precision | 0.936 |
| Recall | 0.921 |
| F1-Score | 0.928 |

The high mAP and F1-score demonstrate strong detection accuracy and reliable segmentation performance.

---

# Project Workflow

1. Collect satellite image dataset
2. Preprocess images
3. Train YOLOv8 segmentation model
4. Detect buildings in unseen images
5. Generate segmentation outputs
6. Visualize prediction results

---

# Folder Structure

```text
project/
│
├── backend/
├── frontend/
├── demo/
├── dataset/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/harshitttt21/building_detection.git
```

Move into the project folder:

```bash
cd building_detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
python app.py
```

---

# Future Improvements

- Real-time satellite monitoring
- Improved segmentation accuracy
- Streamlit-based interactive frontend
- Cloud deployment
- Multi-class object detection
- Integration with GIS systems

---

# Results

The model successfully detects and segments buildings from satellite imagery with high precision and robustness.

It performs effectively under:
- Different lighting conditions
- Dense urban areas
- Complex backgrounds
- Varying image qualities

---

# Author

Harshit Sharma  
Trish Sindher

GitHub:
https://github.com/harshitttt21
