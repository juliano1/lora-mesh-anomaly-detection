"""IsolationForestModel."""
import numpy as np
import pytest


def _X(n=60, seed=0):
    return np.random.default_rng(seed).normal(0, 1, size=(n, 11))


def test_invalid_input_raises():
    from ai.anomaly_ml import IsolationForestModel
    m = IsolationForestModel(contamination=0.05, n_estimators=20)
    with pytest.raises(ValueError):
        m.train(np.zeros((0, 11)))
    X = _X()
    X[:, 3] = np.nan          # coluna inteira ausente (sklearn 1.8 aceitaria)
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.train(X)


def test_failed_refit_preserves_previous_scaler_and_forest(monkeypatch):
    import ai.anomaly_ml as mod
    m = mod.IsolationForestModel(contamination=0.05, n_estimators=20)
    assert m.train(_X()) is True
    scaler, forest = m.scaler, m.model
    ref = m.score(_X(5, seed=9))

    class _Boom:
        def __init__(self, *a, **k): pass
        def fit(self, X): raise RuntimeError("falha controlada")

    monkeypatch.setattr(mod, "IsolationForest", _Boom)
    assert m.train(_X(seed=3)) is False
    assert "falha controlada" in m.last_train_error
    assert m.is_trained and m.scaler is scaler and m.model is forest
    np.testing.assert_allclose(m.score(_X(5, seed=9)), ref)


def test_degenerate_reference_is_not_published():
    from ai.anomaly_ml import IsolationForestModel
    m = IsolationForestModel(contamination=0.05, n_estimators=20)
    assert m.train(np.ones((30, 11))) is False
    assert m.last_train_error.startswith("degenerate_reference")
    assert m.is_trained is False


def test_predict_matches_negative_margin_and_offset_exposed():
    from ai.anomaly_ml import IsolationForestModel
    m = IsolationForestModel(contamination=0.1, n_estimators=50)
    assert m.train(_X(200)) is True
    X = np.vstack([_X(50, seed=4), np.full((3, 11), 8.0)])
    margin = m.margin(X)
    np.testing.assert_array_equal(m.predict(X) == -1, margin < 0)
    np.testing.assert_allclose(margin, m.score(X) - m.offset)


def test_random_state_is_configurable_and_reported():
    from ai.anomaly_ml import IsolationForestModel
    m = IsolationForestModel(contamination=0.05, n_estimators=20, random_state=7)
    assert m.train(_X()) is True
    assert m.get_info()["parameters"]["random_state"] == 7
