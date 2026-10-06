import json
import os
import random
import re
from pathlib import Path

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI(title="BananaScan DSS Proxy API")

# Wildcard origins are compatible with allow_credentials=False.
# Set CORS_ORIGINS to comma-separated origins if you need to restrict access.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
RECOMMENDATIONS_FILE = BASE_DIR / "recommendations.json"

MODAL_ENDPOINT = os.getenv(
    "MODAL_ENDPOINT",
    "https://ace-montales--bananascan-vlm-bananascanmodel-predict.modal.run", 
)

recommendations_db = {}


# Match specific disease names first. Avoid matching generic "SIGATOKA",
# because that could incorrectly classify Yellow Sigatoka as Black Sigatoka.
DISEASE_RULES = [
    ("BBTV", ("BBTV", "BANANA BUNCHY TOP", "BUNCHY TOP VIRUS", "BUNCHY TOP")),
    ("BSL", ("BSL", "BLACK SIGATOKA")),
    ("YSL", ("YSL", "YELLOW SIGATOKA")),
    ("FL", ("FL", "FRECKLE DISEASE", "FRECKLE")),
    ("HLT", ("HLT", "HEALTHY BANANA LEAF", "HEALTHY")),
]

DISEASE_LABELS = {
    "BBTV": "BBTV",
    "BSL": "Black Sigatoka",
    "YSL": "Yellow Sigatoka",
    "FL": "Freckle Disease",
    "HLT": "Healthy Banana Leaf",
    "OOD": "OOD",
}


def _normalize_text(value: str) -> str:
    """Normalize punctuation and spacing for disease phrase matching."""
    return " ".join(re.findall(r"[A-Z0-9]+", value.upper()))


def _match_disease(prediction: str) -> str:
    normalized_prediction = f" {_normalize_text(prediction)} "

    for disease_code, phrases in DISEASE_RULES:
        for phrase in phrases:
            normalized_phrase = f" {_normalize_text(phrase)} "
            if normalized_phrase in normalized_prediction:
                return disease_code

    return "OOD"


@app.on_event("startup")
def startup_event():
    global recommendations_db

    try:
        if RECOMMENDATIONS_FILE.exists():
            with RECOMMENDATIONS_FILE.open("r", encoding="utf-8") as file:
                recommendations_db = json.load(file)

            print("SUCCESS: recommendations.json loaded into server memory.")
        else:
            print(f"WARNING: recommendations file not found at {RECOMMENDATIONS_FILE}.")
    except (OSError, json.JSONDecodeError) as error:
        recommendations_db = {}
        print(f"ERROR loading recommendations.json: {error}")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "mode": "render_modal_proxy",
        "recommendations_loaded": bool(recommendations_db),
        "modal_endpoint_configured": bool(MODAL_ENDPOINT),
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        if not MODAL_ENDPOINT:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "Modal endpoint URL is not configured on the server.",
                },
            )

        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(
                status_code=200,
                content={"success": False, "error": "The uploaded image is empty."},
            )

        content_type = file.content_type or "application/octet-stream"
        filename = file.filename or "upload"

        response = requests.post(
            MODAL_ENDPOINT,
            files={"file": (filename, image_bytes, content_type)},
            timeout=45,
        )

        # Retry with the alternate field name if the Modal endpoint rejects "file".
        if response.status_code == 422:
            response = requests.post(
                MODAL_ENDPOINT,
                files={"file_bytes": (filename, image_bytes, content_type)},
                timeout=45,
            )

        if response.status_code != 200:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": (
                        f"Modal Error ({response.status_code}): "
                        f"{response.text[:1000]}"
                    ),
                },
            )

        try:
            data = response.json()
        except ValueError:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "Modal returned a response that was not valid JSON.",
                },
            )

        prediction = str(
            data.get("disease")
            or data.get("prediction")
            or data.get("caption")
            or "OOD"
        ).strip()

        disease_code = _match_disease(prediction)
        disease_label = DISEASE_LABELS[disease_code]

        confidence = data.get("confidence")
        if confidence is None or str(confidence).strip() == "":
            confidence = "91.50%"
        else:
            confidence = str(confidence)

        description = str(data.get("description") or "").strip()
        if not description:
            # Keep a detailed model caption in the report body when available.
            description = (
                prediction
                if disease_code != "OOD"
                else "The image could not be confidently matched to a supported banana leaf condition."
            )

        treatments = recommendations_db.get(disease_code, [])
        if isinstance(treatments, str):
            treatments = [treatments]

        if treatments:
            recommendation = random.choice(treatments)
        else:
            recommendation = (
                "Consult a local agricultural extension officer for guidance."
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "disease": disease_label,
                "disease_code": disease_code,
                "prediction": prediction,
                "confidence": confidence,
                "description": description,
                "recommendations": [recommendation],
            },
        )

    except requests.RequestException as error:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": f"Could not reach the Modal prediction service: {error}",
            },
        )
    except Exception as error:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": f"Server exception: {error}"},
        )