import onnxruntime as ort
import numpy as np
import cv2
import json

"""
Script to run inference on an exported CLIP image encoder ONNX model using OpenCV and ONNX Runtime.

1. Implements a custom image preprocessing pipeline that mimics CLIPProcessor using OpenCV.
2. Loads an ONNX model and prepares an input tensor from an image.
3. Runs the model to generate image embeddings.
"""

    
def load_image_opencv(image_path, input_size=224):
    """
    Loads an image and preprocesses it for CLIP using OpenCV-style blob processing.

    Args:
        image_path (str): Path to the image file.
        input_size (int): Size to which the image will be resized (usually 224 or 336).

    Returns:
        np.ndarray: Preprocessed image blob (1x3xHxW) ready for CLIP.
    """
    # CLIP mean and std (RGB)
    mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
    std = [0.26862954, 0.26130258, 0.27577711]

    # Load and resize image
    frame = cv2.imread(image_path)  # BGR
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    frame = cv2.resize(frame, (input_size, input_size))

    # Convert image to blob with mean subtraction (BGR)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0 / 255.0,
                                 size=(input_size, input_size),
                                 mean=mean, swapRB=True, crop=False)

    # # Normalize by std
    # for c in range(3):
    #     blob[0, c, :, :] /= std[c]

    return blob


# Load ONNX model
session = ort.InferenceSession("models/clip_image_encoder.onnx")

# Preprocess image
input_tensor = load_image_opencv("cat.png")

# Get input/output layers
input_layer_name = session.get_inputs()[0].name
output_layer_name = session.get_outputs()[0].name

# Run inference
outputs = session.run([output_layer_name], {input_layer_name: input_tensor})
embedding = outputs[0]

print("Input shape:", input_tensor.shape)
print("Output shape", embedding.shape)

with open('inference_emb_onnx.json', 'w') as f:
    json.dump(embedding.tolist(), f)