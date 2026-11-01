"""LSTM-AE: entrada inválida, candidato isolado, escala segura."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai.lstm_autoencoder import LSTMAutoencoderModel  # noqa: E402

W, F = 6, 11


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, size=(n, W, F))).astype(np.float32)


def _trained():
    m = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=2)
    assert m.train(_data()[:32], X_val=_data()[32:]) is True
    return m


def test_wrong_shape_raises_value_error():
    m = LSTMAutoencoderModel(window_size=W, hidden_size=8)
    with pytest.raises(ValueError):
        m.train(np.ones((20, W + 1, F), dtype=np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_training_or_validation_raises(bad):
    m = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=2)
    X = _data()
    X_val = _data(8, seed=1)
    X_val[0, 0, 0] = bad
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.train(X, X_val=X_val)
    X[3, 2, 1] = bad
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.train(X)


def test_empty_validation_raises():
    m = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=2)
    with pytest.raises(ValueError):
        m.train(_data(), X_val=np.zeros((0, W, F), dtype=np.float32))


def test_insufficient_sequences_returns_false_with_reason():
    m = LSTMAutoencoderModel(window_size=W, hidden_size=8)
    assert m.train(_data(5)) is False
    assert m.last_train_error.startswith("insufficient_sequences")
    assert m.is_trained is False


def test_failed_retrain_preserves_active_state():
    m = _trained()
    threshold, net = m.threshold, m._net
    with pytest.raises(ValueError):
        m.train(np.full((20, W, F), np.nan, dtype=np.float32))
    assert m.is_trained is True and m.threshold == threshold and m._net is net


def test_non_finite_loss_rejected_and_state_preserved(monkeypatch):
    import ai.lstm_autoencoder as mod

    class _NaNLoss(torch.nn.Module):
        def forward(self, a, b):
            return ((a - b) ** 2).mean() * float("nan")

    m = _trained()
    threshold = m.threshold
    monkeypatch.setattr(mod.nn, "MSELoss", _NaNLoss)
    assert m.train(_data()[:32], X_val=_data()[32:]) is False
    assert m.last_train_error == "non_finite_validation_loss"
    assert m.threshold == threshold and math.isfinite(m.threshold)


def test_constant_feature_gets_unit_scale():
    X = _data()
    X[:, :, 4] = 7.0
    m = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=2)
    assert m.train(X[:32], X_val=X[32:]) is True
    assert float(m._scaler_std[4]) == 1.0
    probe = X[0].copy()
    probe[:, 4] = 7.01
    z = (probe - m._scaler_mean) / m._scaler_std
    assert abs(float(z[0, 4]) - 0.01) < 1e-4


def test_threshold_and_weights_are_finite_after_success():
    m = _trained()
    assert math.isfinite(m.threshold)
    assert all(torch.isfinite(t).all() for t in m._net.state_dict().values())


# ─── Validação da API pública de inferência ───────────────────────────────────

@pytest.mark.parametrize("bad_shape", [(W, 1), (1, F), (W + 1, F), (2, W, F)])
def test_score_rejects_wrong_shapes(bad_shape):
    m = _trained()
    with pytest.raises(ValueError):
        m.score(np.zeros(bad_shape, dtype=np.float32))


@pytest.mark.parametrize("bad_shape", [(5, W, 1), (5, 1, F), (W, F)])
def test_score_batch_rejects_wrong_shapes(bad_shape):
    m = _trained()
    with pytest.raises(ValueError):
        m.score_batch(np.zeros(bad_shape, dtype=np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_inference_rejects_non_finite_input(bad):
    m = _trained()
    seq = _data(1)[0]
    seq[1, 2] = bad
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.score(seq)
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.predict(seq)
    batch = _data(3)
    batch[0, 0, 0] = bad
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.score_batch(batch)


def test_predict_rejects_non_finite_threshold():
    m = _trained()
    m._threshold = float("nan")
    with pytest.raises(ValueError, match="limiar"):
        m.predict(_data(1)[0])


def test_untrained_model_refuses_to_score():
    """Modelo não treinado não pontua: ausência de modelo não pode virar score normal."""
    m = LSTMAutoencoderModel(window_size=W, hidden_size=8)
    with pytest.raises(RuntimeError, match="não treinado"):
        m.score(_data(1)[0])


# ─── Constante decimal com acumulação estável ─────────────────────────────────

@pytest.mark.parametrize("value", [-83.2, 0.1, 7.3, 0.3333])
@pytest.mark.parametrize("n", [40, 100, 400])
def test_constant_decimal_feature_gets_unit_scale(value, n):
    X = _data(n)
    X[:, :, 0] = value
    m = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=1)
    split = int(n * 0.8)
    assert m.train(X[:split], X_val=X[split:]) is True
    assert float(m._scaler_std[0]) == 1.0
    probe = X[0].copy()
    probe[:, 0] = np.float32(value + 0.01)
    z = (probe - m._scaler_mean) / m._scaler_std
    assert abs(float(z[0, 0]) - 0.01) < 1e-4
