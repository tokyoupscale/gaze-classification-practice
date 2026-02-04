from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io

import torch
from torchvision import transforms
from model.model import model

from typing import Dict


app = FastAPI(
    title="gaze classification api"
)

# for streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = ['down', 'left', 'right', 'straight', 'up']

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_instance = model.to(device)
model_instance.load_state_dict(torch.load("gaze-classification.pth", map_location=device))
model_instance.eval()

print(f"model loaded on {device}, ready to predict!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/health")
def health():
    return {
        "status": "online",
        "message": "hello world",
        "model_device": str(device),
        "classes": class_names
    }

@app.post("/predict")
async def prediction(file: UploadFile = File(...)) -> Dict:
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="need an img")
    
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_instance(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())

            result = {
                "success": True,
                "predicted_class": class_names[pred_idx],
                "confidence": float(probs[pred_idx]),
            }

            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="prediction error")
    
@app.get("/available-classes")
def get_available_classes():
    return {
        "classes": [
            {"name": class_names[i]}
            for i in range(len(class_names))
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5252)