"""
Testes unitários para LSTMAutoencoderModel.

Cobre:
- Treino sobre dataset sintético (retorna True, threshold calibrado)
- Score de sequência normal vs. anômala (anomalous > normal)
- predict: True para anômalo, False para normal
- get_residuals: shape correto
- Três agregações S1/S2/S3
- Persistência: save/load preserva threshold e scores
- Validação de shape incorreta levanta ValueError
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Importar de forma segura — pula testes se torch ausente
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not torch_available, reason="PyTorch não instalado"
)

from ai.lstm_autoencoder import LSTMAutoencoderModel, MODELS_DIR, N_FEATURES


N_SEQS = 60
W = 6   # Janela pequena para testes rápidos


@pytest.fixture
def normal_dataset() -> np.ndarray:
    """Dataset normal sintético (N, W, N_FEATURES)."""
    rng = np.random.default_rng(42)
    base = np.zeros(N_FEATURES, dtype=np.float32)
    data = base + rng.normal(0, 0.1, size=(N_SEQS, W, N_FEATURES))
    return data.astype(np.float32)


@pytest.fixture
def anomalous_seq() -> np.ndarray:
    """Sequência anômala com valores extremos."""
    seq = np.ones((W, N_FEATURES), dtype=np.float32) * 10.0
    return seq


@pytest.fixture
def trained_model(normal_dataset) -> LSTMAutoencoderModel:
    """Modelo treinado sobre dataset normal."""
    model = LSTMAutoencoderModel(
        window_size=W,
        hidden_size=8,     # Pequeno para testes rápidos
        epochs=5,          # Poucas épocas
        patience=3,
    )
    ok = model.train(normal_dataset)
    assert ok, "Treino falhou no fixture"
    return model


# ── Treino ────────────────────────────────────────────────────────────────────


class TestThresholdCalibration:
    def test_threshold_comes_from_calibration_block_not_validation(self, normal_dataset):
        """Validação escolhe a época; o p99 sai de um bloco posterior e separado."""
        rng = np.random.default_rng(7)
        X_train, X_val = normal_dataset[:40], normal_dataset[40:]
        X_calib = (rng.normal(0, 0.5, size=(30, W, N_FEATURES))).astype(np.float32)  # mais ruidoso
        model = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=3, patience=2)
        assert model.train(X_train, X_val, X_calib) is True
        assert model.threshold_source == "calibration"
        calib_scores = model.score_batch(X_calib)
        assert model.threshold == pytest.approx(np.percentile(calib_scores, 99.0), rel=1e-5)
        assert model.threshold > np.percentile(model.score_batch(X_val), 99.0)

    def test_without_calibration_block_threshold_is_marked_as_validation(self, normal_dataset):
        model = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=2)
        assert model.train(normal_dataset[:40], normal_dataset[40:]) is True
        assert model.threshold_source == "validation"

    def test_empty_calibration_block_is_refused(self, normal_dataset):
        model = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=2)
        empty = np.zeros((0, W, N_FEATURES), dtype=np.float32)
        with pytest.raises(ValueError):
            model.train(normal_dataset[:40], normal_dataset[40:], empty)


class TestTrain:
    def test_train_returns_true(self, normal_dataset):
        model = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=3)
        assert model.train(normal_dataset) is True

    def test_is_trained_after_train(self, trained_model):
        assert trained_model.is_trained is True

    def test_threshold_calibrated(self, trained_model):
        assert trained_model.threshold < float("inf")
        assert trained_model.threshold > 0.0

    def test_train_fails_on_too_few_sequences(self):
        model = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=3)
        X_tiny = np.ones((5, W, N_FEATURES), dtype=np.float32)
        assert model.train(X_tiny) is False

    def test_train_fails_on_wrong_shape(self):
        model = LSTMAutoencoderModel(window_size=W, hidden_size=8)
        X_wrong = np.ones((N_SEQS, W + 1, N_FEATURES), dtype=np.float32)
        with pytest.raises(ValueError):
            model.train(X_wrong)

    def test_accepts_explicit_val_set(self, normal_dataset):
        X_train, X_val = normal_dataset[:40], normal_dataset[40:]
        model = LSTMAutoencoderModel(window_size=W, hidden_size=8, epochs=3)
        assert model.train(X_train, X_val=X_val) is True


# ── Inferência ────────────────────────────────────────────────────────────────


class TestScore:
    def test_score_returns_float(self, trained_model, normal_dataset):
        seq = normal_dataset[0]  # (W, N_FEATURES)
        s = trained_model.score(seq)
        assert isinstance(s, float)
        assert s >= 0.0

    def test_anomalous_score_higher_than_normal(self, trained_model, normal_dataset, anomalous_seq):
        """No geral, a sequência anômala deve ter score maior que a normal."""
        scores_normal = [trained_model.score(normal_dataset[i]) for i in range(10)]
        score_anomalous = trained_model.score(anomalous_seq)
        avg_normal = np.mean(scores_normal)
        assert score_anomalous > avg_normal, (
            f"Score anômalo ({score_anomalous:.5f}) não maior que médio normal ({avg_normal:.5f})"
        )

    def test_predict_normal_is_false(self, trained_model, normal_dataset):
        """Sequências normais devem geralmente não ser previstas como anômalas."""
        predictions = [trained_model.predict(normal_dataset[i]) for i in range(20)]
        # Com threshold p=99, no máximo 1% de FP esperado
        false_positive_rate = sum(predictions) / len(predictions)
        assert false_positive_rate <= 0.2, (
            f"Taxa de FP elevada: {false_positive_rate:.0%}"
        )

    def test_score_batch_shape(self, trained_model, normal_dataset):
        scores = trained_model.score_batch(normal_dataset[:10])
        assert scores.shape == (10,)
        assert np.all(scores >= 0.0)

    def test_untrained_model_refuses_to_score(self):
        model = LSTMAutoencoderModel(window_size=W, hidden_size=8)
        seq = np.ones((W, N_FEATURES), dtype=np.float32)
        with pytest.raises(RuntimeError, match="não treinado"):
            model.score(seq)
        with pytest.raises(RuntimeError, match="não treinado"):
            model.score_batch(seq[np.newaxis, ...])

    def test_get_residuals_shape(self, trained_model, normal_dataset):
        residuals = trained_model.get_residuals(normal_dataset[0])
        assert residuals.shape == (W, N_FEATURES)
        assert np.all(residuals >= 0.0)

    def test_get_residuals_vector_shape(self, trained_model, normal_dataset):
        vec = trained_model.get_residuals_vector(normal_dataset[0])
        assert vec.shape == (W * N_FEATURES,)


# ── Agregações S1/S2/S3 ───────────────────────────────────────────────────────


class TestScoreAggregations:
    @pytest.mark.parametrize("aggregation", ["mean", "last", "max"])
    def test_each_aggregation_runs(self, normal_dataset, aggregation):
        model = LSTMAutoencoderModel(
            window_size=W, hidden_size=8, epochs=3, score_aggregation=aggregation
        )
        ok = model.train(normal_dataset)
        assert ok
        s = model.score(normal_dataset[0])
        assert isinstance(s, float)

    def test_invalid_aggregation_raises_on_init(self):
        with pytest.raises(ValueError):
            LSTMAutoencoderModel(window_size=W, score_aggregation="invalid")


# ── Persistência ──────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load_preserves_threshold(self, trained_model, normal_dataset, tmp_path, monkeypatch):
        """save/load deve preservar threshold e scaler."""
        monkeypatch.setattr("ai.lstm_autoencoder.MODELS_DIR", tmp_path)

        saved_path = trained_model.save(version=999)
        assert saved_path.exists()

        # Atualizar o caminho dos metadados para apontar para tmp_path
        meta_path = tmp_path / "lstm_ae_v999.json"
        assert meta_path.exists()

        loaded = LSTMAutoencoderModel.load.__func__(
            LSTMAutoencoderModel,  # noqa — workaround monkeypatch classmethod
            999,
        )
        assert loaded.threshold == pytest.approx(trained_model.threshold)
        np.testing.assert_allclose(loaded.score_batch(normal_dataset[:5]),
                                   trained_model.score_batch(normal_dataset[:5]), rtol=1e-5)

    def test_load_latest_returns_none_when_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai.lstm_autoencoder.MODELS_DIR", tmp_path)
        result = LSTMAutoencoderModel.load_latest()
        assert result is None
