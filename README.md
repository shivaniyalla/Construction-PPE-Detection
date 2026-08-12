# 🦺 Construction PPE Detection System

A YOLO-based computer vision system for detecting Personal Protective Equipment (PPE) and identifying safety violations at construction sites.

## 📌 Project Overview

This project uses a trained YOLO object detection model to detect construction-site objects and PPE-related safety violations from images.

The system can identify:

- Excavator
- Gloves
- Hardhat
- Ladder
- Mask
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- SUV

## 🚀 Technologies Used

- Python
- YOLO
- Ultralytics
- PyTorch
- Streamlit
- OpenCV
- Pillow

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Precision | 85.7% |
| Recall | 67.0% |
| mAP@50 | 72.53% |
| mAP@50-95 | 42.7% |

## 🖥️ Application

The project includes a Streamlit web application where users can upload a construction-site image and receive:

- Detected objects
- Bounding boxes
- Confidence scores
- Detection summary
- Safety violation status

## ▶️ How to Run

Install the required packages:

```bash
pip install -r requirements.txt