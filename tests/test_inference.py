import cv2
import numpy as np

from core import inference
from core.inference import MODE_FALLBACK, get_inference, heuristic_fallback, postprocess_probability_map


def test_postprocess_keeps_largest_blob_and_drops_speckle():
    prob = np.zeros((384, 384), dtype=np.float32)
    prob[100:200, 100:220] = 0.9  # main region
    prob[300:320, 300:320] = 0.8  # smaller second region
    prob[10, 10] = 0.99  # single-pixel speckle
    mask = postprocess_probability_map(prob, (384, 384))

    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask[150, 150] == 1.0
    assert mask[310, 310] == 0.0
    assert mask[10, 10] == 0.0


def test_postprocess_respects_threshold_and_output_size():
    prob = np.full((384, 384), 0.2, dtype=np.float32)  # below PROB_THRESHOLD everywhere
    assert postprocess_probability_map(prob, (50, 80)).sum() == 0

    prob[96:288, 96:288] = 0.6
    mask = postprocess_probability_map(prob, (100, 200))
    assert mask.shape == (100, 200)
    assert 0.2 < mask.mean() < 0.3  # central quarter of the frame


def test_heuristic_fallback_finds_dark_central_blob():
    bgr = np.full((200, 200, 3), 220, dtype=np.uint8)
    cv2.circle(bgr, (100, 100), 25, (30, 30, 120), -1)
    mask, prob = heuristic_fallback(bgr)
    assert mask[100, 100] == 1.0
    assert mask[5, 5] == 0.0
    assert np.array_equal(mask, prob)


def test_get_inference_reports_fallback_mode_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", False)
    path = tmp_path / "img.png"
    bgr = np.full((120, 160, 3), 220, dtype=np.uint8)
    cv2.circle(bgr, (80, 60), 20, (30, 30, 120), -1)
    cv2.imwrite(str(path), bgr)

    result = get_inference(str(path))
    assert result is not None
    assert result.mode == MODE_FALLBACK
    assert result.rgb.shape == (120, 160, 3)
    assert result.mask.shape == (120, 160)


def test_get_inference_returns_none_for_non_image(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    assert get_inference(str(path)) is None


def test_missing_weights_never_yield_an_untrained_model(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(inference, "_model", None)
    monkeypatch.setattr(inference, "_model_load_failed", False)
    monkeypatch.setattr(inference, "MODEL_PATH", str(tmp_path / "absent.pth"))
    assert inference.load_model() is None
    assert inference.model_ready() is False
