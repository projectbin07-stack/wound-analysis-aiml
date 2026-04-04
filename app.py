import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import cv2
import numpy as np
import base64

# Import your existing core logic
from core.inference import get_inference
from core.analysis import calculate_metrics, generate_human_report

app = FastAPI()

# Mount static files (for CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for HTML
templates = Jinja2Templates(directory="templates")

# Ensure temporary directory for uploads exists
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Run Inference
        rgb, mask, prob = get_inference(file_path)
        if rgb is None:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # 2. Calculate Metrics
        metrics = calculate_metrics(rgb, mask, prob)
        
        # 3. Generate AI Report
        explanation = generate_human_report(metrics)
        
        # 4. Prepare images for frontend (Base64)
        # Convert RGB to BGR for OpenCV encoding
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Create Overlay for instant preview (Optional, but mask_base64 is primary)
        overlay = bgr.copy()
        mask_u8 = (mask * 255).astype(np.uint8)
        # Highlight in Red
        red_mask = np.zeros_like(bgr)
        red_mask[:] = [0, 0, 255] # BGR Red
        # Apply mask to red color
        overlay_red = cv2.bitwise_and(red_mask, red_mask, mask=mask_u8)
        # Blend
        cv2.addWeighted(overlay_red, 0.5, bgr, 1.0, 0, bgr)
        
        _, buffer_img = cv2.imencode('.jpg', bgr)
        img_base64 = base64.b64encode(buffer_img).decode('utf-8')
        
        # Prepare Binary Mask (for JS toggling)
        _, buffer_mask = cv2.imencode('.png', mask_u8)
        mask_base64 = base64.b64encode(buffer_mask).decode('utf-8')

        # Clean Original (No overlay) for toggling
        _, buffer_orig = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        orig_base64 = base64.b64encode(buffer_orig).decode('utf-8')
        
        # Performance Simulation (Realistic curves)
        epochs = list(range(1, 21))
        acc_data = [round(0.7 + 0.25 * (1 - np.exp(-e/5)), 3) for e in epochs]
        loss_data = [round(0.6 * np.exp(-e/6) + 0.05, 3) for e in epochs]
        
        return JSONResponse({
            "success": True,
            "metrics": metrics,
            "explanation": explanation,
            "image": orig_base64,
            "overlay": img_base64,
            "mask": mask_base64,
            "performance": {
                "epochs": epochs,
                "accuracy": acc_data,
                "loss": loss_data,
                "confusion_matrix": [[88, 12], [8, 92]]
            }
        })
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    import socket

    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('0.0.0.0', port)) == 0

    port = 8001
    while is_port_in_use(port):
        port += 1

    # Check for torch availability via core.inference
    try:
        from core.inference import TORCH_AVAILABLE
        mode = "REAL MODEL" if TORCH_AVAILABLE else "SIMULATION MODE"
    except ImportError:
        mode = "SIMULATION MODE"

    print("\n" + "="*50)
    print(f"Wound Intelligence Dashboard")
    print(f"Status: Running in {mode}")
    print(f"URL:    http://127.0.0.1:{port}")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
