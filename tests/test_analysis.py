import math

import numpy as np
import pytest
import requests

from core import analysis
from core.analysis import (
    calculate_metrics,
    clamp,
    generate_human_report,
    redness_band,
    template_report,
)


def test_clamp():
    assert clamp(-1) == 0.0
    assert clamp(0.25) == 0.25
    assert clamp(7) == 1.0


def test_metrics_on_known_disc(red_disc):
    img, mask = red_disc
    m = calculate_metrics(img, mask, mask * 0.9)

    area = int(mask.sum())
    assert m["area"] == area
    assert area == pytest.approx(math.pi * 40**2, rel=0.01)
    # Area is a fraction of the whole image, whatever its resolution.
    assert m["area_pct"] == pytest.approx(100 * area / (200 * 300), abs=0.01)
    assert m["healthy_pct"] == pytest.approx(100 - m["area_pct"], abs=0.01)
    # Mean probability inside the mask, reported as a percentage.
    assert m["confidence"] == pytest.approx(90.0, abs=0.01)
    # R=180, G=40 -> 180 / (40 + 1)
    assert m["redness"] == pytest.approx(180 / 41, abs=0.001)
    # A circle is the most compact shape: perimeter^2 / (4*pi*area) is close to 1.
    assert 0.85 < m["compactness"] < 1.25
    assert m["health_score"] == pytest.approx(100 - m["risk"], abs=0.01)


def test_risk_formula_matches_documented_weights(red_disc):
    img, mask = red_disc
    m = calculate_metrics(img, mask, mask)
    n = m["norm"]
    expected = (0.4 * n["area"] + 0.35 * n["red"] + 0.2 * n["irreg"] + 0.05 * 0.0) * 100
    assert m["risk"] == pytest.approx(expected, abs=0.01)


def test_low_confidence_increases_risk(red_disc):
    img, mask = red_disc
    confident = calculate_metrics(img, mask, mask * 1.0)
    unsure = calculate_metrics(img, mask, mask * 0.4)
    assert unsure["risk"] == pytest.approx(confident["risk"] + 0.05 * 0.6 * 100, abs=0.02)


def test_empty_mask_is_safe():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    zero = np.zeros((64, 64), dtype=np.float32)
    m = calculate_metrics(img, zero, zero)
    assert m["area"] == 0
    assert m["confidence"] == 0.0
    assert m["redness"] == 0.0
    assert m["severity"] == "Mild"


def test_mask_at_other_resolution_is_rescaled(red_disc):
    img, mask = red_disc
    small = mask[::2, ::2]
    m_full = calculate_metrics(img, mask, mask)
    m_small = calculate_metrics(img, small, small)
    assert m_small["area_pct"] == pytest.approx(m_full["area_pct"], rel=0.05)


def test_report_falls_back_to_template_when_ollama_is_down(monkeypatch, red_disc):
    img, mask = red_disc
    m = calculate_metrics(img, mask, mask)

    def boom(*args, **kwargs):
        raise requests.ConnectionError("no server")

    monkeypatch.setattr(analysis.requests, "post", boom)
    text = generate_human_report(m)
    assert text == template_report(m)
    assert "not a diagnosis" in text
    assert text.startswith("SUMMARY:")


def test_without_a_model_no_confidence_score_or_band_is_reported(red_disc):
    # Regression: v0.1.0 used the fallback mask as its own "probability" and reported
    # 100 % confidence plus a severity and an infection band.
    img, mask = red_disc
    m = calculate_metrics(img, mask, None)
    assert m["scored"] is False
    for key in ("confidence", "risk", "health_score", "severity", "redness_band", "healthy_pct"):
        assert m[key] is None, key
    assert "infection" not in m and "danger" not in m
    # Plain measurements of the outline are still reported.
    assert m["area"] == int(mask.sum())
    assert m["redness"] == pytest.approx(180 / 41, abs=0.001)


@pytest.mark.parametrize("grey", [40, 128, 220])
def test_neutral_grey_region_is_in_the_low_redness_band(grey):
    # Regression: v0.1.0 put any neutral grey region in the "Medium" infection band.
    img = np.full((64, 64, 3), grey, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1
    m = calculate_metrics(img, mask, mask)
    assert m["norm"]["red"] == 0.0
    assert m["redness_band"] == "Low"


def test_redness_band_edges():
    assert [redness_band(x) for x in (0.0, 0.4, 0.41, 0.7, 0.71)] == [
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
    ]


def test_reports_give_no_medical_directives(red_disc, monkeypatch):
    img, mask = red_disc
    monkeypatch.setattr(analysis.requests, "post", lambda *a, **k: pytest.fail("no LLM call"))
    unscored = generate_human_report(calculate_metrics(img, mask, None))
    assert "No trained model is loaded" in unscored
    for text in (unscored, template_report(calculate_metrics(img, mask, mask))):
        lowered = text.lower()
        assert "not a diagnosis" in lowered
        for phrase in ("consult", "recommend", "seek", "infection"):
            assert phrase not in lowered
