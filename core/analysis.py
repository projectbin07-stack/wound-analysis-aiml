"""Geometric / colour metrics and the heuristic risk score.

The risk score is a hand-tuned weighted sum of image-derived features. It has
not been clinically validated and is not a diagnostic measure. It is computed
only for masks from the trained model: in fallback mode (no model loaded) there
is no probability map, so confidence, the score and the bands are ``None``.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Redness index R / (G + 1): about 1.0 for any neutral grey, white or black region,
# so the band starts there. v0.1.0 started it at 0.2, which put plain grey in "Medium".
MIN_REDNESS, MAX_REDNESS = 1.0, 3.0
MIN_COMPACTNESS, MAX_COMPACTNESS = 1.0, 5.0

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "5"))


def clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def redness_band(red_norm: float) -> str:
    """Band for the normalised redness index (colour only; not an infection measure)."""
    return "High" if red_norm > 0.7 else "Medium" if red_norm > 0.4 else "Low"


def calculate_metrics(rgb_img: np.ndarray, mask_np: np.ndarray, prob_np: np.ndarray | None) -> dict:
    """Compute area, shape and colour metrics plus the heuristic risk score.

    ``mask_np`` is a binary mask (values 0/1) and ``prob_np`` the model's
    probability map in [0, 1], or ``None`` when no model produced the mask
    (fallback mode). Both may be at a different resolution from ``rgb_img`` and
    are measured at the image resolution. Without a probability map, the fields
    that need a model (``confidence``, ``risk``, ``health_score``, ``severity``,
    ``redness_band``, ``healthy_pct``) are ``None`` and ``scored`` is ``False``.
    """
    h, w = rgb_img.shape[:2]

    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    wound = mask_u8 == 255
    area = int(np.count_nonzero(wound))

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

    measured = {
        "area": area,
        "area_pct": round(a_n * 100, 2),
        "redness": round(red_idx, 3),
        "compactness": round(comp, 3),
        "norm": {"area": a_n, "red": r_n, "irreg": i_n},
    }
    if prob_np is None:
        # No model: nothing here can justify a confidence, a score or a band.
        return {
            **measured,
            "scored": False,
            "healthy_pct": None,
            "confidence": None,
            "risk": None,
            "health_score": None,
            "severity": None,
            "redness_band": None,
        }

    prob = prob_np.astype(np.float32)
    if prob.shape[:2] != (h, w):
        prob = cv2.resize(prob, (w, h))
    conf = float(prob[wound].mean()) if area > 0 else 0.0  # mean probability, 0-1
    c_p = 1.0 - clamp(conf)  # uncertainty penalty

    risk = (0.4 * a_n + 0.35 * r_n + 0.2 * i_n + 0.05 * c_p) * 100

    sev = "Mild" if risk < 30 else "Moderate" if risk < 60 else "Severe" if risk < 80 else "Critical"

    return {
        **measured,
        "scored": True,
        "healthy_pct": round(100 - (a_n * 100), 2),
        "confidence": round(conf * 100, 2),
        "risk": round(risk, 2),
        "health_score": round(100 - risk, 2),
        "severity": sev,
        "redness_band": redness_band(r_n),
    }


NOT_A_DIAGNOSIS = (
    "NOTE: This is research software, not a medical device. These values are "
    "image measurements, not a diagnosis, and give no medical advice."
)


def template_report(data: dict) -> str:
    """Deterministic summary used when no local LLM is reachable (or no model is loaded)."""
    if not data.get("scored"):
        return (
            "SUMMARY: No trained model is loaded, so no score, band or confidence is "
            "reported.\n"
            "MEANING: The outline comes from a simple brightness threshold near the "
            "image centre (heuristic fallback). It is not a wound detection.\n" + NOT_A_DIAGNOSIS
        )
    return (
        f"SUMMARY: The heuristic score places this image in the {data['severity']} band "
        f"with a score of {data['risk']}%.\n"
        f"MEANING: The redness band is {data['redness_band']}. These values are "
        f"computed from image colour and shape only.\n" + NOT_A_DIAGNOSIS
    )


def generate_human_report(data: dict) -> str:
    """Ask a local Ollama model for a plain-language summary; fall back to a template.

    Without a model-produced score there is nothing for the LLM to describe, so the
    template is used directly.
    """
    if not data.get("scored"):
        return template_report(data)
    prompt = f"""
    You are summarising the output of an experimental image-analysis tool. It is not a
    medical device. Do not diagnose. Describe these image-derived wound metrics plainly:
    - Area: {data["area"]} px
    - Redness Index: {data["redness"]}
    - Heuristic Risk Score: {data["risk"]}%
    - Severity band: {data["severity"]}
    - Redness band (colour only): {data["redness_band"]}

    Format the response as follows:
    SUMMARY: [1-2 lines]
    MEANING: [What the measurements describe]
    NOTE: [State that these are image measurements, not a diagnosis. Give no medical
    advice and do not tell the reader what to do.]

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
