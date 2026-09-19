from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import httpx
import os
import re

app = FastAPI(title="DPD OCR.Space OCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "").strip()
OCR_SPACE_URL = "https://api.ocr.space/parse/image"

def normalize_text(text):
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).strip()

def clean_field(text, field):
    t = normalize_text(text)

    if field == "numero":
        t2 = t.replace("O","0").replace("o","0").replace("I","1").replace("l","1")
        groups = re.findall(r"\d+(?:[-_/]\d+)*", t2)
        if groups:
            return max(groups, key=lambda x: len(re.sub(r"\D","",x)))
        return t

    if field == "proceso":
        t2 = t.replace("O","0").replace("o","0").replace("I","1").replace("l","1")
        groups = re.findall(r"\d{5,}", t2)
        if groups:
            return max(groups, key=len)
        return t

    if field == "fecha":
        t2 = t.replace("O","0").replace("o","0").replace("I","1").replace("l","1")
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
    return FileResponse(
        "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": "OCR.Space",
        "api_key_configured": bool(OCR_SPACE_API_KEY),
    }

@app.post("/ocr")
async def run_ocr(
    image: UploadFile = File(...),
    field: str = Form("texto")
):
    if not OCR_SPACE_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Falta configurar OCR_SPACE_API_KEY en Render."}
        )

    try:
        raw = await image.read()
        if not raw:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "La imagen está vacía."}
            )

        filename = image.filename or "recorte.jpg"
        content_type = image.content_type or "image/jpeg"

        headers = {"apikey": OCR_SPACE_API_KEY}
        data = {
            "language": "spa",
            "isOverlayRequired": "false",
            "OCREngine": "2",
            "scale": "true",
        }
        files = {
            "file": (filename, raw, content_type)
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OCR_SPACE_URL,
                headers=headers,
                data=data,
                files=files,
            )

        text_body = response.text
        try:
            payload = response.json()
        except Exception:
            raise RuntimeError(f"OCR.Space respondió HTTP {response.status_code}: {text_body[:250]}")

        if response.status_code >= 400:
            raise RuntimeError(payload.get("ErrorMessage") or f"HTTP {response.status_code}")

        if payload.get("IsErroredOnProcessing"):
            err = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "OCR.Space no pudo procesar la imagen."
            if isinstance(err, list):
                err = " | ".join(map(str, err))
            raise RuntimeError(str(err))

        parsed = payload.get("ParsedResults") or []
        if not parsed:
            raise RuntimeError("OCR.Space no devolvió texto.")

        raw_text = normalize_text(parsed[0].get("ParsedText", ""))
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
