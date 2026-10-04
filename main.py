import json
import os
import random
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests

app = FastAPI(title="BananaScan DSS Proxy API")

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

# Modal deployment URL
MODAL_ENDPOINT = os.getenv(
    "MODAL_ENDPOINT", 
    "https://ace-montales--bananascan-vlm-predict.modal.run"
)

recommendations_db = {}


@app.on_event("startup")
def startup_event():
    global recommendations_db
    try:
        if RECOMMENDATIONS_FILE.exists():
            with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
                recommendations_db = json.load(f)
            print("🎉 SUCCESS: recommendations.json loaded into server memory!")
        else:
            print(f"⚠️ WARNING: File not found at {RECOMMENDATIONS_FILE}.")
    except Exception as e:
        print(f"❌ ERROR LOADING RECOMMENDATIONS JSON: {e}")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "mode": "render_modal_proxy",
        "recommendations_loaded": bool(recommendations_db),
        "modal_endpoint_configured": bool(MODAL_ENDPOINT and MODAL_ENDPOINT != "YOUR_MODAL_URL_HERE"),
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        if not MODAL_ENDPOINT or MODAL_ENDPOINT == "YOUR_MODAL_URL_HERE":
            return JSONResponse(
                status_code=200,
                content={"success": False, "error": "Modal endpoint URL is not configured on Render server."}
            )

        # 1. Read raw image payload transmitted from client
        image_bytes = await file.read()

        # 2. Forward payload to Modal (handles both query parameter and form-data formats)
        # First attempt: sending via files key
        response = requests.post(
            MODAL_ENDPOINT,
            files={"file": (file.filename, image_bytes, file.content_type)},
            timeout=45
        )

        # Fallback if Modal expects query param or raw body bytes
        if response.status_code == 422:
            response = requests.post(
                MODAL_ENDPOINT,
                files={"file_bytes": (file.filename, image_bytes, file.content_type)},
                timeout=45
            )

        if response.status_code != 200:
            return JSONResponse(
                status_code=200,
                content={"success": False, "error": f"Modal Error ({response.status_code}): {response.text}"}
            )

        data = response.json()
        predicted_caption = data.get("disease", "OOD")
        calculated_confidence = data.get("confidence", "91.50%")

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
            "confidence": calculated_confidence,
            "description": f"Visual analysis indicates symptoms matching {predicted_caption}.",
            "recommendations": [selected_treatment]
        })

    except Exception as e:
        return JSONResponse(status_code=200, content={"success": False, "error": f"Server exception: {str(e)}"})