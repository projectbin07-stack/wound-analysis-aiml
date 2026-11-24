import base64

import cv2
import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

import app as app_module
from core import analysis, inference


@pytest.fixture
def client(monkeypatch):
    # Force the labelled fallback path and keep the test offline.
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", False)

    def no_ollama(*args, **kwargs):
        raise requests.ConnectionError("offline test")

    monkeypatch.setattr(analysis.requests, "post", no_ollama)
    return TestClient(app_module.app)


def _png_bytes():
    bgr = np.full((120, 160, 3), 220, dtype=np.uint8)
    cv2.circle(bgr, (80, 60), 20, (30, 30, 120), -1)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def test_index_and_health(client):
    assert client.get("/").status_code == 200
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_ready"] is False


def test_analyze_returns_labelled_fallback_result(client):
    r = client.post("/analyze", files={"file": ("wound.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["mode"] == "fallback"
    assert body["model_used"] is False
    # No fabricated training curves or confusion matrix in the response.
    assert "performance" not in body
    assert body["metrics"]["area"] > 0
    # No model: no fabricated confidence, score or bands.
    assert body["metrics"]["scored"] is False
    assert body["metrics"]["confidence"] is None
    assert body["metrics"]["risk"] is None
    assert body["metrics"]["severity"] is None
    assert body["explanation"].startswith("SUMMARY: No trained model is loaded")
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(body["mask"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (120, 160)


def test_upload_is_removed_and_filename_is_not_trusted(client):
    before = set(app_module.UPLOAD_DIR.iterdir())
    r = client.post("/analyze", files={"file": ("../../evil.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert set(app_module.UPLOAD_DIR.iterdir()) == before
    assert not (app_module.BASE_DIR.parent / "evil.png").exists()


@pytest.mark.parametrize("name", ["notes.txt", "shell.py", "noextension"])
def test_rejects_unsupported_extensions(client, name):
    r = client.post("/analyze", files={"file": (name, b"data", "application/octet-stream")})
    assert r.status_code == 400


def test_rejects_unreadable_image(client):
    r = client.post("/analyze", files={"file": ("fake.png", b"not really a png", "image/png")})
    assert r.status_code == 400


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/analyze", files={"file": ("big.png", _png_bytes(), "image/png")})
    assert r.status_code == 413


def _png(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def test_plain_grey_image_is_not_reported_as_a_wound(client):
    # Regression for v0.1.0, which reported area 6.25 %, confidence 100 %, "Mild" and
    # infection "Medium" for this image with no model loaded.
    paper = np.full((400, 400, 3), 200, dtype=np.uint8)
    paper[150:250, 150:250] = 190
    for image in (paper, np.full((400, 400, 3), 128, dtype=np.uint8)):
        body = client.post("/analyze", files={"file": ("grey.png", _png(image), "image/png")}).json()
        assert body["mode"] == "fallback"
        m = body["metrics"]
        assert m["area"] == 0
        assert (m["confidence"], m["risk"], m["severity"], m["redness_band"]) == (None,) * 4
