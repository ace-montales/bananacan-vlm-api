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

app = FastAPI(title="BananaScan VLM Inference & DSS API")

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

# 1. Replace with your actual Hugging Face Username and Repository Name
# (e.g., "john-doe/banana-blip-model")
HF_REPO_ID = os.getenv("HF_REPO_ID", "Ace-VI/banana-blip-model")

# Global memory variables
model = None
processor = None
device = None
recommendations_db = {}


@app.on_event("startup")
def load_banana_model():
    global model, processor, device, recommendations_db

    # Determine execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- System running on device: {device} ---")

    # Load recommendations.json
    try:
        if RECOMMENDATIONS_FILE.exists():
            with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
                recommendations_db = json.load(f)
            print("🎉 SUCCESS: recommendations.json loaded into server memory!")
        else:
            print(f"⚠️ WARNING: File not found at {RECOMMENDATIONS_FILE}. Default fallbacks will be used.")
    except Exception as e:
        print(f"❌ ERROR LOADING RECOMMENDATIONS JSON: {e}")

    # Load fine-tuned BLIP model weights directly from Hugging Face Hub
    try:
        print(f"Downloading/loading BlipProcessor and Model from Hugging Face: {HF_REPO_ID}...")
        processor = BlipProcessor.from_pretrained(HF_REPO_ID)
        model = BlipForConditionalGeneration.from_pretrained(HF_REPO_ID)
        model.eval()
        model.to(device)
        print("🎉 SUCCESS: Fine-tuned BLIP Model loaded dynamically into memory from Hugging Face!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR LOADING WEIGHTS FROM HUGGING FACE: {e}")
        print("Please verify your HF_REPO_ID is correct and files exist on Hugging Face Hub.")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "recommendations_loaded": bool(recommendations_db),
        "device_used": str(device),
        "hf_repo": HF_REPO_ID
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if model is None or processor is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "Model not loaded on backend server"})

    try:
        # 1. Read raw image stream transmitted via Flutter app
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Preprocess input image
        inputs = processor(images=raw_image, return_tensors="pt").to(device)

        # 3. Execute live inference generation AND score extraction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True
            )

            output_tokens = outputs.sequences[0]
            predicted_caption = processor.decode(output_tokens, skip_special_tokens=True).strip()

            # 4. Safe calculation of real confidence score
            logits = outputs.scores
            token_probs = []

            for i, logit in enumerate(logits):
                if (i + 1) >= len(output_tokens):
                    break

                token_id = output_tokens[i + 1]

                if token_id in [processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id]:
                    break

                prob = torch.softmax(logit, dim=-1)
                token_prob = prob[0][token_id].item()
                token_probs.append(token_prob)

            if token_probs:
                calculated_confidence = (sum(token_probs) / len(token_probs)) * 100
            else:
                calculated_confidence = 50.0

        # 5. SERVER-SIDE RULE-BASED DSS SELECTION
        caption_upper = predicted_caption.upper()
        matched_class = None
        for key in ["BSL", "YSL", "BBTV", "FL", "HLT"]:
            if key in caption_upper:
                matched_class = key
                break

        # Fallback to OOD pool if caption doesn't contain a known disease tag
        if not matched_class:
            matched_class = "OOD"

        # Retrieve class pool from recommendations.json
        class_treatments = recommendations_db.get(matched_class, []) if matched_class else []

        # Sample dynamic treatment option bounded by rule isolation
        if class_treatments:
            selected_treatment = random.choice(class_treatments)
        else:
            selected_treatment = "Consult a local agricultural extension officer for guidance."

        # 6. RETURN UNIFIED RESPONSE MATCHING FLUTTER UI EXPECTATIONS
        return JSONResponse(status_code=200, content={
            "success": True,
            "disease": predicted_caption,
            "confidence": f"{round(calculated_confidence, 2)}%",
            "description": f"Visual analysis indicates symptoms matching {predicted_caption}.",
            "recommendations": [selected_treatment]
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})