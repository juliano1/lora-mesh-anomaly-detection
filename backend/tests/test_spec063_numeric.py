"""Finitude e escala de features constantes."""
import math

import numpy as np
import pytest


def test_require_finite_rejects_nan_and_inf():
    from ai.numeric_guards import require_finite
    require_finite(np.array([1.0, 2.0]), "x")
    with pytest.raises(ValueError, match="x"):
        require_finite(np.array([1.0, np.nan]), "x")
    with pytest.raises(ValueError):
        require_finite(np.array([np.inf]), "y")


def test_safe_feature_scale_uses_one_for_zero_variance():
    from ai.numeric_guards import safe_feature_scale, MIN_FEATURE_STD
    std = np.array([0.0, MIN_FEATURE_STD / 10, 2.5])
    scale = safe_feature_scale(std)
    assert scale.tolist() == [1.0, 1.0, 2.5]


def test_safe_feature_scale_keeps_small_variation_bounded():
    """Contraexemplo da auditoria: variação 0,01 não pode virar 1.000.000."""
    from ai.numeric_guards import safe_feature_scale
    train = np.full((50, 1), 3.0)
    scale = safe_feature_scale(train.std(axis=0))
    z = (np.array([3.01]) - train.mean(axis=0)) / scale
    assert abs(z[0] - 0.01) < 1e-9


def test_safe_feature_scale_rejects_non_finite():
    from ai.numeric_guards import safe_feature_scale
    with pytest.raises(ValueError):
        safe_feature_scale(np.array([np.nan]))


def test_normalizer_fit_rejects_non_finite_and_empty():
    from ai.online.normalization import ScoreNormalizer
    with pytest.raises(ValueError):
        ScoreNormalizer().fit(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        ScoreNormalizer().fit(np.array([]))


def test_normalizer_transform_non_finite_returns_nan():
    from ai.online.normalization import ScoreNormalizer
    n = ScoreNormalizer(invert=False)
    n.fit(np.linspace(0, 1, 100))
    assert math.isnan(n.transform(float("nan")))


def test_normalizer_detects_degenerate_calibration():
    from ai.online.normalization import ScoreNormalizer
    n = ScoreNormalizer(invert=True)
    n.fit(np.full(20, -0.5))
    assert n.is_degenerate is True
    m = ScoreNormalizer(invert=True)
    m.fit(np.linspace(-0.7, -0.3, 20))
    assert m.is_degenerate is False
