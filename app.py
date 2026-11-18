"""FastAPI dashboard for the experimental wound-segmentation pipeline."""

from __future__ import annotations

import base64
import logging
import os
import socket
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from core.analysis import calculate_metrics, generate_human_report
from core.inference import MODE_MODEL, get_inference, model_ready

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("wound-analysis")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

app = FastAPI(title="Wound Analysis (research prototype)")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _encode(image: np.ndarray, ext: str) -> str:
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": model_ready()}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Never trust the client-supplied filename: keep only a whitelisted
    # extension and store the upload under a random name.
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")

    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    file_path.write_bytes(payload)

    try:
        result = get_inference(str(file_path))
        if result is None:
            raise HTTPException(status_code=400, detail="File is not a readable image")

        metrics = calculate_metrics(result.rgb, result.mask, result.prob)
        explanation = generate_human_report(metrics)

        bgr = cv2.cvtColor(result.rgb, cv2.COLOR_RGB2BGR)
        mask_u8 = (result.mask * 255).astype(np.uint8)

        # Red overlay of the mask on top of the original image.
        red = np.zeros_like(bgr)
        red[:] = (0, 0, 255)
        overlay_red = cv2.bitwise_and(red, red, mask=mask_u8)
        overlay = cv2.addWeighted(overlay_red, 0.5, bgr, 1.0, 0)

        return JSONResponse(
            {
                "success": True,
                "mode": result.mode,
                "model_used": result.mode == MODE_MODEL,
                "metrics": metrics,
                "explanation": explanation,
                "image": _encode(bgr, ".jpg"),
                "overlay": _encode(overlay, ".jpg"),
                "mask": _encode(mask_u8, ".png"),
            }
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Analysis failed")
        return JSONResponse({"success": False, "error": "Internal analysis error"}, status_code=500)
    finally:
        file_path.unlink(missing_ok=True)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    import uvicorn

    # Binds to localhost by default; set HOST=0.0.0.0 to expose on the network.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8001"))
    while _port_in_use(port):
        port += 1

    mode = "trained model" if model_ready() else "HEURISTIC FALLBACK (no trained model loaded)"
    print("\n" + "=" * 60)
    print("Wound Analysis dashboard (research prototype, not a medical device)")
    print(f"Segmentation: {mode}")
    print(f"URL:          http://127.0.0.1:{port}")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")
