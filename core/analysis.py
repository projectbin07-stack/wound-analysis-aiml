"""Geometric / colour metrics and the heuristic risk score.

The risk score is a hand-tuned weighted sum of image-derived features. It has
not been clinically validated and is not a diagnostic measure.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

MIN_REDNESS, MAX_REDNESS = 0.2, 2.0
MIN_COMPACTNESS, MAX_COMPACTNESS = 1.0, 5.0

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "5"))


def clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def calculate_metrics(rgb_img: np.ndarray, mask_np: np.ndarray, prob_np: np.ndarray) -> dict:
    """Compute area, shape and colour metrics plus the heuristic risk score.

    ``mask_np`` is a binary mask (values 0/1) and ``prob_np`` a probability map
    in [0, 1]; both may be at a different resolution from ``rgb_img`` and are
    measured at the image resolution.
    """
    h, w = rgb_img.shape[:2]

    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    prob = prob_np.astype(np.float32)
    if prob.shape[:2] != (h, w):
        prob = cv2.resize(prob, (w, h))

    wound = mask_u8 == 255
    area = int(np.count_nonzero(wound))
    conf = float(prob[wound].mean()) if area > 0 else 0.0  # mean probability, 0-1

    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comp = 0.0
    if cnts and area > 0:
        largest = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(largest, True)
        comp = float((perim**2) / (4 * np.pi * area))

    # Redness index: mean R / mean G inside the wound region (image is RGB).
    red_idx = 0.0
    if area > 0:
        wound_px = rgb_img[wound].astype(np.float32)
        red_idx = float(np.mean(wound_px[:, 0]) / (np.mean(wound_px[:, 1]) + 1))

    a_n = clamp(area / float(h * w))  # fraction of the image covered by the mask
    r_n = clamp((red_idx - MIN_REDNESS) / (MAX_REDNESS - MIN_REDNESS))
    i_n = clamp((comp - MIN_COMPACTNESS) / (MAX_COMPACTNESS - MIN_COMPACTNESS))
    c_p = 1.0 - clamp(conf)  # uncertainty penalty

    risk = (0.4 * a_n + 0.35 * r_n + 0.2 * i_n + 0.05 * c_p) * 100

    sev = "Mild" if risk < 30 else "Moderate" if risk < 60 else "Severe" if risk < 80 else "Critical"
    inf = "High" if r_n > 0.7 else "Medium" if r_n > 0.4 else "Low"
    if sev in ("Severe", "Critical") and inf == "High":
        dan = "High"
    elif sev == "Moderate" and inf == "Medium":
        dan = "Moderate"
    else:
        dan = "Low"

    return {
        "area": area,
        "area_pct": round(a_n * 100, 2),
        "healthy_pct": round(100 - (a_n * 100), 2),
        "redness": round(red_idx, 3),
        "compactness": round(comp, 3),
        "confidence": round(conf * 100, 2),
        "risk": round(risk, 2),
        "health_score": round(100 - risk, 2),
        "severity": sev,
        "infection": inf,
        "danger": dan,
        "norm": {"area": a_n, "red": r_n, "irreg": i_n},
    }


def template_report(data: dict) -> str:
    """Deterministic summary used when no local LLM is reachable."""
    return (
        f"SUMMARY: The heuristic score places this image in the {data['severity']} band "
        f"with a score of {data['risk']}%.\n"
        f"MEANING: The redness-based indicator is {data['infection']}. These values are "
        f"computed from image colour and shape only.\n"
        "RECOMMENDATION: This is research software and not a diagnosis. "
        "Consult a healthcare professional about any wound."
    )


def generate_human_report(data: dict) -> str:
    """Ask a local Ollama model for a plain-language summary; fall back to a template."""
    prompt = f"""
    You are summarising the output of an experimental image-analysis tool. It is not a
    medical device. Do not diagnose. Describe these image-derived wound metrics plainly:
    - Area: {data["area"]} px
    - Redness Index: {data["redness"]}
    - Heuristic Risk Score: {data["risk"]}%
    - Severity band: {data["severity"]}
    - Redness-based infection indicator: {data["infection"]}
    - Danger band: {data["danger"]}

    Format the response as follows:
    SUMMARY: [1-2 lines]
    MEANING: [What the data indicates]
    RECOMMENDATION: [Next steps, always including consulting a clinician]

    Keep it concise.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT_S,
        )
        if response.status_code == 200:
            text = response.json().get("response", "").strip()
            if text:
                return text
    except (requests.RequestException, ValueError) as exc:
        logger.info("Ollama unavailable (%s); using template report.", exc)

    return template_report(data)
