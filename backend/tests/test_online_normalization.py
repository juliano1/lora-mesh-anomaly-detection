"""Testes do ScoreNormalizer (normalização de score por percentis do treino)."""
import numpy as np
import pytest
from ai.online.normalization import ScoreNormalizer


def test_transform_after_fit_clamps_to_01():
    norm = ScoreNormalizer(invert=True)
    scores = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
    norm.fit(scores)
    result = norm.transform(-0.5)
    assert 0.0 <= result <= 1.0


def test_most_anomalous_maps_near_1_invert_true():
    norm = ScoreNormalizer(invert=True)
    scores = np.linspace(-0.5, 0.0, 100)  # IF-like: mais negativo = mais anômalo
    norm.fit(scores)
    assert norm.transform(-0.5) > 0.8
    assert norm.transform(0.0) < 0.2


def test_most_anomalous_maps_near_1_invert_false():
    norm = ScoreNormalizer(invert=False)
    scores = np.linspace(0.0, 1.0, 100)  # AE-like: mais alto = mais anômalo
    norm.fit(scores)
    assert norm.transform(1.0) > 0.8
    assert norm.transform(0.0) < 0.2


def test_fallback_before_fit_returns_float():
    norm = ScoreNormalizer(invert=True)
    assert not norm.is_calibrated
    result = norm.transform(-0.3)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_is_calibrated_false_before_fit():
    norm = ScoreNormalizer()
    assert norm.is_calibrated is False


def test_is_calibrated_true_after_fit():
    norm = ScoreNormalizer()
    norm.fit(np.array([0.1, 0.5, 0.9]))
    assert norm.is_calibrated is True


def test_fit_uses_percentiles():
    """p_low=0, p_high=100 → usa exatamente min/max."""
    norm = ScoreNormalizer(p_low=0, p_high=100, invert=False)
    scores = np.array([0.0, 0.5, 1.0])
    norm.fit(scores)
    assert norm.transform(0.0) == pytest.approx(0.0, abs=1e-6)
    assert norm.transform(1.0) == pytest.approx(1.0, abs=1e-6)
    assert norm.transform(0.5) == pytest.approx(0.5, abs=1e-6)


def test_if_scores_median_maps_near_middle():
    """Score mediano do IF (invertido) cai perto do meio da escala calibrada."""
    norm = ScoreNormalizer(p_low=5, p_high=95, invert=True)
    if_scores = np.linspace(-0.5, 0.0, 200)
    norm.fit(if_scores)
    mid = float(np.median(if_scores))
    result = norm.transform(mid)
    assert 0.3 < result < 0.7, f"score médio deve mapear perto de 0.5, mas foi {result}"
