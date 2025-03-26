import onnxruntime as ort
import numpy as np
import cv2

"""
Script to run inference on an exported CLIP image encoder ONNX model using OpenCV and ONNX Runtime.

1. Implements a custom image preprocessing pipeline that mimics CLIPProcessor using OpenCV.
2. Loads an ONNX model and prepares an input tensor from an image.
3. Runs the model to generate image embeddings.
"""

def clip_preprocess_opencv(image_path):
    """
    Simulates CLIPProcessor image pipeline using OpenCV.
    Output: float32 tensor, shape (1, 3, 224, 224)
    """
    # Load and convert to RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize shortest side to 224
    h, w = img.shape[:2]
    scale = 224 / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Center crop
    start_x = (new_w - 224) // 2
    start_y = (new_h - 224) // 2
    img = img[start_y:start_y+224, start_x:start_x+224]

    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    # Normalize with CLIP mean/std
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = (img - mean) / std

    # Convert to CHW and add batch dimension
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)    # Add batch dim
    return img.astype(np.float32)


# Load ONNX model
session = ort.InferenceSession("models/clip_image_encoder.onnx")

# Preprocess image
input_tensor = clip_preprocess_opencv("cat.jpg")

# Get input/output layers
input_layer_name = session.get_inputs()[0].name
output_layer_name = session.get_outputs()[0].name

# Run inference
outputs = session.run([output_layer_name], {input_layer_name: input_tensor})
embedding = outputs[0]

print("Input shape:", input_tensor.shape)
print("Output shape", embedding.shape)