# CLIP Image Encoder: PyTorch → ONNX

This project exports the image encoder from OpenAI's CLIP model (`ViT-B/32`) to ONNX, runs inference using ONNX Runtime, and compares outputs against the original PyTorch version.

## Setup

Install dependencies
```
pip install -r requirements.txt
```

## Scripts

- `export.py`: Downloads and exports the CLIP image encoder to ONNX.
- `inference.py`: Runs inference using OpenCV preprocessing + ONNX Runtime.
- `test.py`: Compares PyTorch vs ONNX outputs (cosine similarity, MAE).
