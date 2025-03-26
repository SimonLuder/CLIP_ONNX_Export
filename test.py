import torch
import numpy as np
import onnxruntime as ort
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from scipy.spatial.distance import cosine

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
image_path = "cat.jpg"
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

# === Preprocess image for PyTorch (CLIPProcessor) ===
image = Image.open(image_path).convert("RGB")
pt_inputs = processor(images=image, return_tensors="pt")
pt_pixel_values = pt_inputs["pixel_values"]

# === PyTorch Inference ===
with torch.no_grad():
    pt_output = pt_encoder(pt_pixel_values).cpu().numpy()

# === ONNX Preprocessing ===
def preprocess_for_onnx(image_path):
    img = Image.open(image_path).convert("RGB")
    img = resize_with_aspect_ratio(img, 224)
    img = center_crop(img, (224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def resize_with_aspect_ratio(image, target_size):
    width, height = image.size
    short_side = min(width, height)
    scale = target_size / short_side
    new_width = round(width * scale)
    new_height = round(height * scale)
    return image.resize((new_width, new_height), Image.BICUBIC)

def center_crop(image, crop_size):
    width, height = image.size
    crop_width, crop_height = crop_size
    left = round((width - crop_width) / 2)
    top = round((height - crop_height) / 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


onnx_input = preprocess_for_onnx(image_path)

# === ONNX Inference ===
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
onnx_output = session.run([output_name], {input_name: onnx_input})[0]

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

pt_norm = l2_normalize(pt_output)
onnx_norm = l2_normalize(onnx_output)

cos_sim = 1 - cosine(pt_norm[0], onnx_norm[0])
diff = np.abs(pt_norm - onnx_norm).mean()

print("Mean absolute difference:", diff)
print("Cosine similarity:", cos_sim)