from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import base64
import httpx
import os
import re

app = FastAPI(title="DPD Google Vision OCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
VISION_URL = "https://vision.googleapis.com/v1/images:annotate"

def normalize_text(text):
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).strip()

def clean_field(text, field):
    t = normalize_text(text)

    if field == "numero":
        t = t.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
        groups = re.findall(r"\d+(?:[-_/]\d+)*", t)
        if groups:
            return max(groups, key=lambda x: len(re.sub(r"\D", "", x)))
        return t

    if field == "proceso":
        t = t.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
        groups = re.findall(r"\d{5,}", t)
        if groups:
            return max(groups, key=len)
        return t

    if field == "fecha":
        t2 = t.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
        m = re.search(r"(\d{1,2})[./\-\s](\d{1,2})[./\-\s](\d{2,4})", t2)
        if m:
            d, mo, y = m.groups()
            if len(y) == 2:
                y = "20" + y
            return f"{d.zfill(2)}-{mo.zfill(2)}-{y}"
        return t

    return t

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": "Google Cloud Vision",
        "api_key_configured": bool(VISION_API_KEY),
    }

@app.post("/ocr")
async def run_ocr(
    image: UploadFile = File(...),
    field: str = Form("texto")
):
    if not VISION_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Falta configurar GOOGLE_VISION_API_KEY en Render."}
        )

    try:
        raw = await image.read()
        if not raw:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "La imagen está vacía."}
            )

        encoded = base64.b64encode(raw).decode("utf-8")

        payload = {
            "requests": [
                {
                    "image": {"content": encoded},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": {"languageHints": ["es"]},
                }
            ]
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                VISION_URL,
                params={"key": VISION_API_KEY},
                json=payload,
            )

        data = response.json()

        if response.status_code >= 400:
            message = (
                data.get("error", {}).get("message")
                if isinstance(data, dict)
                else None
            )
            raise RuntimeError(message or f"Google Vision respondió HTTP {response.status_code}")

        responses = data.get("responses", [])
        if not responses:
            raise RuntimeError("Google Vision no devolvió una respuesta OCR.")

        first = responses[0]
        if first.get("error"):
            raise RuntimeError(first["error"].get("message", "Error de Google Vision"))

        raw_text = ""
        full = first.get("fullTextAnnotation", {})
        if isinstance(full, dict):
            raw_text = full.get("text", "") or ""

        if not raw_text:
            annotations = first.get("textAnnotations", [])
            if annotations:
                raw_text = annotations[0].get("description", "") or ""

        raw_text = normalize_text(raw_text)
        value = clean_field(raw_text, field)

        return {
            "ok": True,
            "field": field,
            "value": value,
            "raw": raw_text,
            "confidence": None,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
