import os
import cv2
import torch
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from torchvision import transforms
from huggingface_hub import hf_hub_download

from model import EmotionEfficientNet
from utils.face_detector import detect_faces


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="NeoVision Emotion API",
    version="1.0.0",
    description="Emotion Recognition API using Deep Learning"
)

# Allow CORS (Flutter / Web / Mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Labels
# -------------------------------
EMOTION_LABELS = [
    "surprise",
    "fear",
    "disgust",
    "happy",
    "sad",
    "angry",
    "neutral"
]


# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# Download Model from Hugging Face
# -------------------------------
MODEL_REPO = "MohammedMajid/neovision-emotion-model"
MODEL_FILENAME = "best_rafdb.pth"

print("⬇️ Downloading model from Hugging Face...")
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME
)


# -------------------------------
# Load Model
# -------------------------------
model = EmotionEfficientNet()
model.load_state_dict(
    torch.load(model_path, map_location=device),
    strict=False
)
model.to(device)
model.eval()


# -------------------------------
# Prediction Function
# -------------------------------
def predict_emotion(face_np):
    img = Image.fromarray(face_np).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return EMOTION_LABELS[idx], confidence


# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_img = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if np_img is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid image file",
                    "emotion": "error",
                    "confidence": 0.0
                }
            )

        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        faces = detect_faces(rgb_img)

        if len(faces) == 0:
            return {
                "emotion": "no_face",
                "confidence": 0.0,
                "faces_detected": 0,
                "results": []
            }

        results = []
        person_id = 1

        for (x, y, w, h) in faces:
            face_crop = rgb_img[y:y+h, x:x+w]
            emotion, confidence = predict_emotion(face_crop)

            results.append({
                "person": person_id,
                "emotion": emotion,
                "confidence": confidence,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            })

            person_id += 1

        primary = results[0]

        return {
            "emotion": primary["emotion"],
            "confidence": primary["confidence"],
            "faces_detected": len(results),
            "results": results
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "emotion": "error",
                "confidence": 0.0
            }
        )


# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device)
    }
