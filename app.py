from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import io, re, os

app = FastAPI(title="DPD PaddleOCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Spanish OCR. PaddleOCR downloads its official models on first start.
ocr = PaddleOCR(
    lang="es",
    ocr_version="PP-OCRv6",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

def normalize_text(text):
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).strip()

def clean_field(text, field):
    t = normalize_text(text)

    if field == "numero":
        t = t.replace("O","0").replace("o","0").replace("I","1").replace("l","1")
        groups = re.findall(r"\d+(?:[-_/]\d+)*", t)
        if groups:
            return max(groups, key=lambda x: len(re.sub(r"\D","",x)))
        return t

    if field == "proceso":
        t = t.replace("O","0").replace("o","0").replace("I","1").replace("l","1")
        groups = re.findall(r"\d{5,}", t)
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

def extract_result(res):
    data = None

    # PaddleOCR result objects expose a json property in v3.x.
    try:
        data = res.json
        if callable(data):
            data = data()
    except Exception:
        data = None

    if not isinstance(data, dict):
        try:
            data = res.to_dict()
        except Exception:
            data = {}

    if "res" in data and isinstance(data["res"], dict):
        data = data["res"]

    texts = data.get("rec_texts", []) if isinstance(data, dict) else []
    scores = data.get("rec_scores", []) if isinstance(data, dict) else []

    texts = [str(x) for x in texts if str(x).strip()]
    numeric_scores = []
    try:
        numeric_scores = [float(x) for x in list(scores)]
    except Exception:
        numeric_scores = []

    return texts, numeric_scores

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"ok": True, "engine": "PaddleOCR", "lang": "es"}

@app.post("/ocr")
async def run_ocr(
    image: UploadFile = File(...),
    field: str = Form("texto")
):
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # Upscale small crops because OCR usually improves with larger glyphs.
        w, h = img.size
        if max(w, h) < 1600:
            scale = min(3.0, 1600 / max(w, h))
            img = img.resize((max(1,int(w*scale)), max(1,int(h*scale))))

        arr = np.array(img)
        results = ocr.predict(arr)

        all_texts = []
        all_scores = []
        for res in results:
            texts, scores = extract_result(res)
            all_texts.extend(texts)
            all_scores.extend(scores)

        raw_text = normalize_text(" ".join(all_texts))
        value = clean_field(raw_text, field)

        confidence = None
        if all_scores:
            confidence = round(sum(all_scores) / len(all_scores) * 100, 1)

        return {
            "ok": True,
            "field": field,
            "value": value,
            "raw": raw_text,
            "confidence": confidence,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
