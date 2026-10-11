import json
import os
import random
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI(title="BananaScan DSS Proxy API")

# Set CORS_ORIGINS to comma-separated origins to restrict access.
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

# Leaf Disease Modal Endpoint
MODAL_ENDPOINT = os.getenv(
    "MODAL_ENDPOINT",
    "https://ace-montales--bananascan-vlm-bananascanmodel-predict.modal.run",
)

# New Banana VQA Modal Endpoint
BANANA_VQA_MODAL_ENDPOINT = os.getenv(
    "BANANA_VQA_MODAL_ENDPOINT",
    "https://ace-montales--bananascan-vlm-bananavqamodel-predict.modal.run",
)

recommendations_db = {}


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


def _match_disease(*values: str) -> str:
    """Return a disease code from the model label or generated caption."""
    normalized_text = f" {' '.join(_normalize_text(value) for value in values)} "

    for disease_code, phrases in DISEASE_RULES:
        for phrase in phrases:
            normalized_phrase = f" {_normalize_text(phrase)} "
            if normalized_phrase in normalized_text:
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
        "disease_modal_endpoint_configured": bool(MODAL_ENDPOINT),
        "vqa_modal_endpoint_configured": bool(BANANA_VQA_MODAL_ENDPOINT),
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
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

    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                MODAL_ENDPOINT,
                files={"file": (filename, image_bytes, content_type)},
            )

            # Retry with the alternate field name if Modal rejects "file".
            if response.status_code == 422:
                response = await client.post(
                    MODAL_ENDPOINT,
                    files={"file_bytes": (filename, image_bytes, content_type)},
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

        if not isinstance(data, dict):
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "Modal returned an unexpected response format.",
                },
            )

        raw_output = str(
            data.get("raw_output")
            or data.get("prediction")
            or data.get("caption")
            or data.get("disease")
            or ""
        ).strip()

        model_disease = str(data.get("disease") or "").strip()
        disease_code = _match_disease(model_disease, raw_output)
        disease_label = (
            model_disease
            if model_disease and model_disease.upper() != disease_code
            else DISEASE_LABELS[disease_code]
        )

        severity = str(data.get("severity") or "Unknown").strip()

        description = str(data.get("description") or raw_output).strip()
        if not description:
            description = (
                "The image could not be matched to a supported banana leaf condition."
                if disease_code == "OOD"
                else f"Visual analysis detected {disease_label}."
            )

        treatments = recommendations_db.get(disease_code, [])
        if isinstance(treatments, str):
            treatments = [treatments]
        elif not isinstance(treatments, list):
            treatments = []

        recommendation = (
            random.choice(treatments)
            if treatments
            else "Consult a local agricultural extension officer for guidance."
        )

        result = {
            "success": True,
            "disease": disease_label,
            "disease_code": disease_code,
            "severity": severity,
            "raw_output": raw_output,
            "prediction": raw_output,
            "description": description,
            "recommendations": [recommendation],
        }

        if data.get("confidence") is not None:
            result["confidence"] = str(data["confidence"])

        return JSONResponse(status_code=200, content=result)

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": "Request to the Modal prediction service timed out.",
            },
        )
    except httpx.RequestError as error:
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


@app.post("/vqa")
async def predict_vqa(
    file: UploadFile = File(...),
    question: str = Form("What disease does this leaf have?")
):
    """Proxy endpoint for the 5-class Banana VQA Modal service."""
    if not BANANA_VQA_MODAL_ENDPOINT:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": "Banana VQA Modal endpoint URL is not configured on the server.",
            },
        )

    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": "The uploaded image is empty."},
        )

    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Send file along with the question form field to Modal
            response = await client.post(
                BANANA_VQA_MODAL_ENDPOINT,
                files={"file": (filename, image_bytes, content_type)},
                data={"question": question},
            )

            if response.status_code == 422:
                response = await client.post(
                    BANANA_VQA_MODAL_ENDPOINT,
                    files={"file_bytes": (filename, image_bytes, content_type)},
                    data={"question": question},
                )

        if response.status_code != 200:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": f"Modal VQA Error ({response.status_code}): {response.text[:1000]}",
                },
            )

        data = response.json()
        answer = data.get("answer") or data.get("prediction") or data.get("raw_output") or ""

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "question": question,
                "answer": answer,
                "prediction": answer,
            },
        )

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": "Request to Modal VQA service timed out."},
        )
    except httpx.RequestError as error:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": f"Could not reach Modal VQA service: {error}"},
        )
    except Exception as error:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": f"Server exception: {error}"},
        )