import numpy as np
import cv2
import requests
import json

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_SIZE = 384
MAX_AREA = IMAGE_SIZE * IMAGE_SIZE
MIN_REDNESS, MAX_REDNESS = 0.2, 2.0
MIN_COMPACTNESS, MAX_COMPACTNESS = 1.0, 5.0

def clamp(x): return max(0.0, min(1.0, x))

def calculate_metrics(rgb_img, mask_np, prob_np):
    """
    Calculates technical metrics from model output.
    """
    h, w = rgb_img.shape[:2]
    area = int(np.sum(mask_np == 1))
    conf = float(prob_np[mask_np == 1].mean()) if area > 0 else 0.0
    
    mask_u8 = (mask_np * 255).astype(np.uint8)
    mask_res = cv2.resize(mask_u8, (w, h))
    
    cnts, _ = cv2.findContours(mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim, comp = 0, 0
    if cnts:
        lrg = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(lrg, True)
        if area > 0: comp = (perim ** 2) / (4 * np.pi * area)
        
    # Redness
    red_idx = 0
    if area > 0:
        # Note: rgb_img is RGB
        wound_px = rgb_img[mask_res == 255]
        if len(wound_px) > 0:
            red_idx = np.mean(wound_px[:, 0]) / (np.mean(wound_px[:, 1]) + 1)
            
    # Risk Calculation
    a_n = clamp(area / MAX_AREA)
    r_n = clamp((red_idx - MIN_REDNESS) / (MAX_REDNESS - MIN_REDNESS))
    i_n = clamp((comp - MIN_COMPACTNESS) / (MAX_COMPACTNESS - MIN_COMPACTNESS))
    c_p = 1 - (conf / 100.0) # conf is prob 0-1
    
    risk = (0.4 * a_n + 0.35 * r_n + 0.2 * i_n + 0.05 * c_p) * 100
    
    sev = "Mild" if risk < 30 else "Moderate" if risk < 60 else "Severe" if risk < 80 else "Critical"
    inf = "High" if r_n > 0.7 else "Medium" if r_n > 0.4 else "Low"
    dan = "High" if (sev in ["Severe", "Critical"] and inf == "High") else "Moderate" if (sev == "Moderate" and inf == "Medium") else "Low"
    
    return {
        "area": area, "area_pct": round(a_n*100, 2), "healthy_pct": round(100 - (a_n*100), 2),
        "redness": round(red_idx, 3), "compactness": round(comp, 3), "confidence": round(conf*100, 2),
        "risk": round(risk, 2), "health_score": round(100 - risk, 2),
        "severity": sev, "infection": inf, "danger": dan,
        "norm": {"area": a_n, "red": r_n, "irreg": i_n}
    }

def generate_human_report(data):
    """Call local Ollama API for human-friendly explanation"""
    url = "http://localhost:11434/api/generate"
    prompt = f"""
    Analyze these clinical wound metrics and provide a professional medical-AI report:
    - Area: {data['area']} px
    - Redness Index: {data['redness']}
    - Risk Score: {data['risk']}%
    - Severity: {data['severity']}
    - Infection Risk: {data['infection']}
    - Danger Level: {data['danger']}
    
    Format the response as follows:
    SUMMARY: [1-2 lines]
    MEANING: [What the data indicates]
    RECOMMENDATION: [Next steps]
    
    Keep it concise and professional.
    """
    try:
        response = requests.post(url, json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=5)
        if response.status_code == 200:
            return response.json().get("response", "Analysis complete.")
    except:
        pass
        
    return f"SUMMARY: The wound is classified as {data['severity']} with a risk score of {data['risk']}%.\n" \
           f"MEANING: An infection risk of {data['infection']} is detected based on tissue redness.\n" \
           f"RECOMMENDATION: Standard wound care is advised. Consult a healthcare provider if condition persists."
