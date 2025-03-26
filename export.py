import torch
from transformers import CLIPModel, CLIPProcessor
import onnx

"""
Script to export the image encoder portion of the CLIP model, e.g. (ViT-B/32) to ONNX format.

1. Loads the pretrained CLIP model from Hugging Face.
2. Extracts and wraps the vision encoder and projection layers.
3. Generates a dummy input to define input shape and exports the model to ONNX.
4. Verifies the validity of the exported ONNX model.
"""

# PARAMETERS-------------------------------------------
input_shape = (1, 3, 224, 224)                              # Input image size
clip_model = "openai/clip-vit-base-patch32"                 # CLIP Model (default is ViT-B/32)
onnx_export_path = "models/clip_image_encoder.onnx"        # Export filename
input_layer_name = "pixel_values"                           # Default input layer name is "pixel_values"
output_layer_name = "pooled_output"                         # Select "pooled_output" or "last_hidden_state" as output layer
opset_version = 14                                          # Opset version
# -----------------------------------------------------


class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model
        self.proj = model.visual_projection

    def forward(self, pixel_values):
        vision_out = self.vision(pixel_values)
        pooled = vision_out.pooler_output
        return self.proj(pooled)
    

model = CLIPModel.from_pretrained(clip_model)
processor = CLIPProcessor.from_pretrained(clip_model)

image_encoder = CLIPImageEncoder(model)

# Dummy input
dummy_input = torch.randn(*input_shape)

# Export to ONNX
torch.onnx.export(
    image_encoder,
    dummy_input,
    onnx_export_path,
    export_params=True,
    opset_version=opset_version,
    do_constant_folding=True,
    input_names=[input_layer_name],
    output_names=[output_layer_name],
    dynamic_axes={
        input_layer_name: {0: "batch_size"},
        output_layer_name: {0: "batch_size"}
    }
)

print(f"CLIP image encoder exported to {onnx_export_path}")

# Optional: Check if the ONNX model is valid
onnx_model = onnx.load(onnx_export_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")