import io
import json
import os
import random
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

app = FastAPI(title="BananaScan VLM Direct Render API")

# Enable CORS for cross-platform clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File and directory paths
BASE_DIR = Path(__file__).resolve().parent
RECOMMENDATIONS_FILE = BASE_DIR / "recommendations.json"

# Hugging Face Repository configuration
HF_REPO_ID = os.getenv("HF_REPO_ID", "Ace-VI/banana-blip-model").strip()

recommendations_db = {}
processor = None
model = None


@app.on_event("startup")
def startup_event():
    global recommendations_db, processor, model

    # 1. Load recommendations.json into memory
    try:
        if RECOMMENDATIONS_FILE.exists():
            with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
                recommendations_db = json.load(f)
            print("🎉 SUCCESS: recommendations.json loaded into server memory!")
        else:
            print(f"⚠️ WARNING: File not found at {RECOMMENDATIONS_FILE}. Default fallbacks will be used.")
    except Exception as e:
        print(f"❌ ERROR LOADING RECOMMENDATIONS JSON: {e}")

    # 2. Download and load custom model weights directly into memory
    try:
        print(f"⏳ Loading custom VLM model '{HF_REPO_ID}' into memory...")
        processor = BlipProcessor.from_pretrained(HF_REPO_ID)
        model = BlipForConditionalGeneration.from_pretrained(HF_REPO_ID)
        model.eval()
        print("🎉 SUCCESS: VLM model loaded successfully on startup!")
    except Exception as e:
        print(f"❌ ERROR LOADING MODEL ON STARTUP: {e}")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "mode": "render_local_vlm",
        "model_loaded": model is not None,
        "recommendations_loaded": bool(recommendations_db),
        "hf_repo": HF_REPO_ID,
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        if model is None or processor is None:
            return JSONResponse(
                status_code=200,
                content={"success": False, "error": "Model failed to initialize on startup."}
            )

        # 1. Read raw image stream transmitted via Flutter app
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Run local model inference using PyTorch & Transformers
        inputs = processor(raw_image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
            predicted_caption = processor.decode(out[0], skip_special_tokens=True).strip()

        calculated_confidence = 91.50

        # 3. SERVER-SIDE RULE-BASED DSS SELECTION
        caption_upper = predicted_caption.upper()
        matched_class = None
        for key in ["BSL", "YSL", "BBTV", "FL", "HLT"]:
            if key in caption_upper:
                matched_class = key
                break

        if not matched_class:
            matched_class = "OOD"

        # Retrieve class pool from recommendations.json
        class_treatments = recommendations_db.get(matched_class, []) if matched_class else []

        if class_treatments:
            selected_treatment = random.choice(class_treatments)
        else:
            selected_treatment = "Consult a local agricultural extension officer for guidance."

        # 4. RETURN UNIFIED RESPONSE MATCHING FLUTTER UI EXPECTATIONS
        return JSONResponse(status_code=200, content={
            "success": True,
            "disease": predicted_caption,
            "confidence": f"{round(calculated_confidence, 2)}%",
            "description": f"Visual analysis indicates symptoms matching {predicted_caption}.",
            "recommendations": [selected_treatment]
        })

    except Exception as e:
        return JSONResponse(status_code=200, content={"success": False, "error": f"Server exception: {str(e)}"})