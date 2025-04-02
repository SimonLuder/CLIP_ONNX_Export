import torch
import numpy as np
import onnxruntime as ort
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from scipy.spatial.distance import cosine
import json
import cv2

"""
Script to compare image embeddings from a PyTorch CLIP model and its exported ONNX version.

1. Loads the CLIP model from Hugging Face and wraps the image encoder.
2. Preprocesses an image using both CLIPProcessor and a custom ONNX-compatible pipeline.
3. Runs inference with both PyTorch and ONNXRuntime.
4. Compares the outputs using cosine similarity and mean absolute difference.
"""

# --- Config ---
clip_model_id = "openai/clip-vit-base-patch32"
onnx_path = "models/clip_image_encoder.onnx"
image_path = "cat.png"
# ---------------

# === Load PyTorch CLIP Model & Processor ===
model = CLIPModel.from_pretrained(clip_model_id)
processor = CLIPProcessor.from_pretrained(clip_model_id)
model.eval()

# === Wrapper for PyTorch image encoder ===
class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model
        self.proj = model.visual_projection

    def forward(self, pixel_values):
        outputs = self.vision(pixel_values)
        pooled = outputs.pooler_output
        return self.proj(pooled)

pt_encoder = CLIPImageEncoder(model)


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
    frame = cv2.imread(image_path)  # BGR by default
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    frame = cv2.resize(frame, (input_size, input_size))

    # Convert image to blob with mean subtraction
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0 / 255.0,
                                 size=(input_size, input_size),
                                 mean=mean, swapRB=True, crop=False)

    # # Normalize by std
    # for c in range(3):
    #     blob[0, c, :, :] /= std[c]

    return blob


image_blob = load_image_opencv(image_path)
pt_pixel_values = torch.from_numpy(image_blob).float()

# === PyTorch Inference ===
with torch.no_grad():
    pt_output = pt_encoder(pt_pixel_values).cpu().numpy()

# === ONNX Inference ===
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
onnx_output = session.run([output_name], {input_name: image_blob})[0]


with open('cat_emb_pt.json', 'w') as f:
    json.dump(pt_output.tolist(), f)

with open('cat_emb_onnx.json', 'w') as f:
    json.dump(onnx_output.tolist(), f)

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

pt_norm = l2_normalize(pt_output)
onnx_norm = l2_normalize(onnx_output)

cos_sim = 1 - cosine(pt_norm[0], onnx_norm[0])
diff = np.abs(pt_norm - onnx_norm).mean()

print("Mean absolute difference:", diff)
print("Cosine similarity:", cos_sim)