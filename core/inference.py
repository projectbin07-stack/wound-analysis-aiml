"""Wound segmentation inference.

Two execution modes exist and the caller is always told which one ran:

* ``"model"``    - the trained U-Net (ResNet34 encoder) loaded from
  ``models/best_model.pth``.
* ``"fallback"`` - a classical Otsu-threshold heuristic used only when PyTorch
  or the weights are unavailable. It is NOT a trained model and its output
  must not be interpreted as a segmentation prediction.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODE_MODEL = "model"
MODE_FALLBACK = "fallback"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
IMAGE_SIZE = 384
PROB_THRESHOLD = 0.3

TORCH_AVAILABLE = False
try:
    import albumentations as A
    import torch
    from albumentations.pytorch import ToTensorV2

    from model import get_model

    TORCH_AVAILABLE = True
except (ImportError, OSError) as exc:  # OSError covers DLL load failures on Windows
    logger.warning("PyTorch stack unavailable (%s); heuristic fallback will be used.", exc)

_model = None
_model_load_failed = False
device = None
transform = None

if TORCH_AVAILABLE:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    except Exception as exc:
        logger.warning("Could not initialise the PyTorch pipeline (%s).", exc)
        TORCH_AVAILABLE = False


class InferenceResult(NamedTuple):
    """Output of :func:`get_inference`."""

    rgb: np.ndarray  # HxWx3 uint8, RGB
    mask: np.ndarray  # HxW float32, values in {0, 1}
    prob: np.ndarray  # HxW float32, values in [0, 1]
    mode: str  # MODE_MODEL or MODE_FALLBACK


def model_ready() -> bool:
    """True when the trained model can be used (PyTorch importable and weights present)."""
    return TORCH_AVAILABLE and os.path.exists(MODEL_PATH) and not _model_load_failed


def load_model():
    """Load the trained model once. Returns ``None`` if it cannot be loaded.

    A model without its trained weights is never returned: running a randomly
    initialised decoder would produce meaningless masks.
    """
    global _model, _model_load_failed
    if not TORCH_AVAILABLE or _model_load_failed:
        return None
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model weights not found at %s.", MODEL_PATH)
        _model_load_failed = True
        return None

    try:
        model = get_model()
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    except Exception as exc:
        logger.error("Failed to load model weights: %s", exc)
        _model_load_failed = True
        return None
    _model = model
    return _model


def postprocess_probability_map(prob_map: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Turn a probability map into a clean binary mask at the original resolution.

    Threshold, remove speckle with open/close, keep only the largest connected
    region, then resize with nearest-neighbour so the mask stays binary.
    """
    h_out, w_out = out_hw
    mask_bin = (prob_map > PROB_THRESHOLD).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask_clean)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid, [largest], -1, 255, -1)

    resized = cv2.resize(solid, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.float32)


def _run_model(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    model = load_model()
    if model is None or transform is None:
        return None
    h_orig, w_orig = rgb.shape[:2]
    try:
        img_t = transform(image=rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob_map = torch.sigmoid(model(img_t)).squeeze().cpu().numpy()
    except Exception as exc:
        logger.error("Model inference failed: %s", exc)
        return None

    logger.debug("prob_map range %.4f-%.4f", float(prob_map.min()), float(prob_map.max()))
    mask = postprocess_probability_map(prob_map, (h_orig, w_orig))
    prob = cv2.resize(prob_map, (w_orig, h_orig)).astype(np.float32)
    return mask, prob


def heuristic_fallback(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classical (non-learned) stand-in used when the model is unavailable.

    Otsu-thresholds the central half of the image and keeps the largest dark
    blob. The returned "probability" is the mask itself: there is no model
    confidence in this mode.
    """
    h_orig, w_orig = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    cy, cx = h_orig // 2, w_orig // 2
    roi_h, roi_w = h_orig // 2, w_orig // 2
    y1, y2 = max(0, cy - roi_h // 2), min(h_orig, cy + roi_h // 2)
    x1, x2 = max(0, cx - roi_w // 2), min(w_orig, cx + roi_w // 2)

    roi = gray[y1:y2, x1:x2]
    if roi.size:
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            solid = np.zeros_like(roi)
            cv2.drawContours(solid, [largest], -1, 255, -1)
            mask[y1:y2, x1:x2] = solid

    mask_f = (mask > 0).astype(np.float32)
    return mask_f, mask_f.copy()


def get_inference(image_path: str) -> InferenceResult | None:
    """Segment the image at ``image_path``. Returns ``None`` if it cannot be read."""
    original = cv2.imread(image_path)
    if original is None:
        return None
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    result = _run_model(rgb)
    if result is not None:
        mask, prob = result
        return InferenceResult(rgb, mask, prob, MODE_MODEL)

    mask, prob = heuristic_fallback(original)
    return InferenceResult(rgb, mask, prob, MODE_FALLBACK)
