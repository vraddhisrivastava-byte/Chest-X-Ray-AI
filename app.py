import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from fastapi.middleware.cors import CORSMiddleware
import base64, io, os

CONDITIONS    = ["Pneumonia", "Infiltration", "Effusion", "Cardiomegaly", "Atelectasis"]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Load model
model = models.densenet121(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, 5)
)

# On Hugging Face, model file sits next to app.py
model_path = "best_model_phase2.pth"
print(f"Loading model from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
model = model.to(DEVICE)
print("Model loaded ✅")

# Predict function
def gradio_predict(image):
    img_pil      = Image.fromarray(image).convert("RGB").resize((224, 224))
    raw_np       = np.array(img_pil) / 255.0
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]

    top_idx      = int(np.argmax(probs))
    target_layer = [model.features.denseblock4]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(top_idx)]
        )[0]
        cam_image = show_cam_on_image(
            raw_np.astype(np.float32), grayscale_cam, use_rgb=True
        )

    cam_pil  = Image.fromarray(cam_image)
    buffer   = io.BytesIO()
    cam_pil.save(buffer, format="PNG")
    cam_b64  = base64.b64encode(buffer.getvalue()).decode()

    return probs.tolist(), cam_b64

# Launch with CORS
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.JSON(label="Probabilities"),
        gr.Text(label="GradCAM Base64")
    ],
    title="ChestAI Backend"
)

app = demo.app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

demo.launch()
