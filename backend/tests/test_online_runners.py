"""Testes dos runners online"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_ctx(features_vec=None, node_ids=None):
    from ai.online.registry import DetectionContext
    return DetectionContext(
        session=None,
        node_id=1,
        feature_extractor=None,
        features_vec=features_vec if features_vec is not None else np.zeros(10),
        network_node_ids=node_ids or [1, 2, 3],
        network_id=1,
    )


def _make_spec(id="test", scaling="self"):
    from ai.online.registry import AlgorithmSpec
    return AlgorithmSpec(
        id=id, label="Test", input_kind="point",
        scaling=scaling, incremental=True, factory=lambda _: None,
    )


def _ready(runner, calib=(-0.6, -0.2)):
    """Simula runner já treinado: prontidão composta exige normalizador calibrado."""
    runner._ready = True
    runner._normalizer.fit(np.array(calib))
    return runner


def test_point_runner_score_one_returns_algo_result():
    """PointRunner.score_one() retorna AlgoResult com score_norm ∈ [0,1]."""
    from ai.online.runners import PointRunner
    from ai.online.registry import AlgoResult

    model = MagicMock()
    model.is_trained = True
    model.predict.return_value = np.array([-1])   # anomalia
    model.score.return_value = np.array([-0.3])   # raw score IF-like

    spec = _make_spec(scaling="self")
    runner = _ready(PointRunner(model=model, spec=spec))

    ctx = _make_ctx(features_vec=np.zeros(10))
    result = runner.score_one(ctx)

    assert isinstance(result, AlgoResult)
    assert result.is_anomaly is True
    assert 0.0 <= result.score_norm <= 1.0


def test_point_runner_normal_returns_is_anomaly_false():
    from ai.online.runners import PointRunner

    model = MagicMock()
    model.is_trained = True
    model.predict.return_value = np.array([1])    # normal
    model.score.return_value = np.array([0.0])

    runner = _ready(PointRunner(model=model, spec=_make_spec()))
    result = runner.score_one(_make_ctx())
    assert result.is_anomaly is False



def test_point_runner_ensure_trained_returns_false_when_untrained():
    from ai.online.runners import PointRunner

    model = MagicMock()
    model.is_trained = False
    model.train.return_value = False  # treino falha

    runner = PointRunner(model=model, spec=_make_spec())

    fe = MagicMock()
    fe.extract_batch.return_value = []

    ctx = _make_ctx()
    ctx = ctx.__class__(
        session=MagicMock(), node_id=1, feature_extractor=fe,
        features_vec=np.zeros(10), network_node_ids=[1], network_id=1
    )
    result = runner.ensure_trained(ctx)
    assert result is False


def test_point_runner_model_flag_alone_does_not_make_runner_ready():
    """is_trained do modelo não basta — normalizador e etapas do runner também."""
    from ai.online.runners import PointRunner

    model = MagicMock()
    model.is_trained = True

    runner = PointRunner(model=model, spec=_make_spec())
    assert runner.ensure_trained(_make_ctx()) is False      # sem sessão/FE: não treina
    assert runner.is_ready() is False
    assert _ready(runner).ensure_trained(_make_ctx()) is True


def _ml_settings(**over):
    base = dict(ml_lstm_window=12, ml_if_contamination=0.05, ml_if_n_estimators=100)
    base.update(over)
    return type("S", (), base)()


def test_lstm_factory_builds_without_typeerror():
    from ai.online.pipeline import _register_defaults  # garante registro
    from ai.online.registry import get
    runner = get("lstm_ae").factory(_ml_settings())
    # window_size propagado ao modelo (não 'window')
    assert runner._model.window_size == 12


def test_sequence_runner_trains_on_synthetic_sequences(monkeypatch):
    from ai.online.runners import SequenceRunner
    from ai.lstm_autoencoder import LSTMAutoencoderModel
    from ai.online.registry import DetectionContext
    from ai.reference_policy import ReferencePolicy
    from datetime import datetime

    runner = SequenceRunner(model=LSTMAutoencoderModel(window_size=4, epochs=2, n_features=1))
    X = np.random.RandomState(0).rand(24, 4, 1).astype("float32")

    class _DS:
        def __init__(self):
            self.X, self.meta = X, list(range(24))
        def summary(self):
            return {"n_sequences": 24}

    class _SB:
        def __init__(self, **kw): pass
        def build_reference_dataset(self, *a, **k): return _DS()
        def split_train_val_calib(self, meta, val_fraction=0.2, calib_fraction=0.2):
            return (list(range(14)), list(range(14, 19)), list(range(19, 24)),
                    datetime(2026, 1, 1), datetime(2026, 1, 1, 12))

    monkeypatch.setattr("ai.sequence_builder.SequenceBuilder", _SB)
    ctx = DetectionContext(session=object(), node_id=1, feature_extractor=object(),
                           features_vec=np.zeros(1, dtype="float32"),
                           network_node_ids=[1, 2], network_id=7,
                           reference_time=datetime(2026, 1, 2),
                           reference_policy=ReferencePolicy.empty())
    assert runner.ensure_trained(ctx) is True
    assert runner._model.is_trained is True
    # Limiar no bloco de calibração (19–23), separado da validação da parada antecipada.
    assert runner._model.threshold_source == "calibration"
    assert runner.training_summary["threshold_source"] == "calibration"
    assert runner.training_summary["n_calib"] == 5 and len(runner.calibration_set.X) == 5


def test_point_runner_trains_with_network_nodes_only():
    """Treino usa extract_sliding_windows dos nós da rede, não extract_batch global."""
    from ai.online.runners import PointRunner
    from ai.online.registry import AlgorithmSpec, DetectionContext
    from ai.anomaly_ml import IsolationForestModel

    calls = {"sliding_nodes": [], "extract_batch": 0}

    class _FE:
        from ai.sequence_builder import FEATURE_NAMES as feature_names  # Nomes canônicos
        def extract_sliding_windows(self, session, node_id, **kw):
            calls["sliding_nodes"].append(node_id)
            return [{n: 0.1 * (node_id + i) for n in self.feature_names} for i in range(6)]
        def get_feature_matrix(self, batch):
            return np.array([[f[n] for n in self.feature_names] for _, f in batch], dtype="float32")
        def extract_batch(self, *a, **k):
            calls["extract_batch"] += 1
            return []

    class _Session:
        def get(self, model, node_id):
            return type("N", (), {"sim_mode": False})()

    spec = AlgorithmSpec(id="isolation_forest", label="IF", input_kind="point",
                         scaling="self", incremental=True, factory=lambda s: None)
    runner = PointRunner(model=IsolationForestModel(contamination=0.1), spec=spec)
    from ai.reference_policy import ReferencePolicy
    ctx = DetectionContext(session=_Session(), node_id=1, feature_extractor=_FE(),
                           features_vec=None, network_node_ids=[10, 11], network_id=7,
                           reference_policy=ReferencePolicy.empty())
    assert runner.ensure_trained(ctx) is True
    assert sorted(set(calls["sliding_nodes"])) == [10, 11]   # só nós da rede
    assert calls["extract_batch"] == 0                        # NÃO usa o global


def test_point_runner_excludes_sim_mode_nodes_from_training():
    """Nós em sim_mode não entram no treino do PointRunner."""
    from ai.online.runners import PointRunner
    from ai.online.registry import AlgorithmSpec, DetectionContext
    from ai.anomaly_ml import IsolationForestModel

    sliding = []

    class _FE:
        from ai.sequence_builder import FEATURE_NAMES as feature_names  # Nomes canônicos
        def extract_sliding_windows(self, session, node_id, **kw):
            sliding.append(node_id)
            return [{n: 0.1 * (node_id + i) for n in self.feature_names} for i in range(6)]
        def get_feature_matrix(self, batch):
            return np.array([[f[n] for n in self.feature_names] for _, f in batch], dtype="float32")

    class _Session:
        def get(self, model, node_id):
            return type("N", (), {"sim_mode": node_id == 11})()  # nó 11 em sim_mode

    spec = AlgorithmSpec(id="isolation_forest", label="IF", input_kind="point",
                         scaling="self", incremental=True, factory=lambda s: None)
    runner = PointRunner(model=IsolationForestModel(contamination=0.1), spec=spec)
    from ai.reference_policy import ReferencePolicy
    ctx = DetectionContext(session=_Session(), node_id=1, feature_extractor=_FE(),
                           features_vec=None, network_node_ids=[10, 11], network_id=7,
                           reference_policy=ReferencePolicy.empty())
    runner.ensure_trained(ctx)
    assert sliding == [10]  # nó 11 (sim_mode) foi pulado


# ─── Calibração de τ do fusion ──────────────────────────────────────────────

def test_fusion_tau_calibrated_from_validation_percentile():
    from ai.online.pipeline import OnlinePipeline
    p = OnlinePipeline("isolation_forest", "lstm_ae", "fusion", fusion_alpha=0.5)
    s_final = np.linspace(0.0, 1.0, 100)
    tau = p.calibrate_tau(s_final, contamination=0.05)
    assert 0.94 <= tau <= 0.96
    assert p.calibrate_tau(np.array([]), contamination=0.05) is None


def test_point_runner_score_vectors_uses_given_online_vectors():
    """O pontual pontua os vetores da janela online (não um frame de 5 min)."""
    from ai.online.runners import PointRunner
    from ai.online.registry import AlgorithmSpec

    seen = {}
    class _M:
        is_trained = True
        def score(self, X):
            seen["X"] = np.asarray(X)
            return np.zeros(len(X))
        def predict(self, X):
            return np.array([-1, 1, 1])
    spec = AlgorithmSpec(id="isolation_forest", label="IF", input_kind="point",
                         scaling="self", incremental=True, factory=lambda s: None)
    runner = PointRunner(model=_M(), spec=spec)
    X = np.arange(3 * 2, dtype="float32").reshape(3, 2)
    scores, decisions, raw = runner.score_vectors(X)
    assert scores.shape == (3,) and raw.tolist() == [0.0, 0.0, 0.0]
    assert decisions.tolist() == [True, False, False]
    assert np.array_equal(seen["X"], X)


def test_sequence_runner_score_val_scores_full_batch():
    from ai.online.runners import SequenceRunner
    seen = {}
    class _M:
        is_trained = True
        threshold = 0.5
        def score_batch(self, X):
            seen["X"] = np.asarray(X)
            return np.array([0.1, 0.9, 0.2, 0.7, 0.3])
    runner = SequenceRunner(model=_M())
    X_val = np.random.RandomState(0).rand(5, 4, 2).astype("float32")
    scores, decisions, raw = runner.score_val(X_val)
    assert scores.shape == (5,) and raw.tolist() == [0.1, 0.9, 0.2, 0.7, 0.3]
    assert decisions.tolist() == [False, True, False, True, False]
    assert np.array_equal(seen["X"], X_val)


def test_fusion_calibration_aligns_point_window_with_sequence_end():
    """Τ sobre o holdout do LSTM; o IF pontua a janela de 15 min que
    termina no fim de cada sequência de validação (a mesma da inferência)."""
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from ai.online.effective_config import resolve_effective_config
    from ai.online.pipeline import OnlinePipeline
    from ai.online.registry import DetectionContext
    from ai.online.runners import HoldoutSet
    from ai.sequence_builder import FEATURE_NAMES

    p = OnlinePipeline("isolation_forest", "lstm_ae", "fusion", fusion_alpha=0.5)
    p._contamination = 0.1
    p._config = resolve_effective_config(None)
    t0 = datetime(2026, 10, 19, 12)
    meta = [SimpleNamespace(node_id=1 + i % 2, end=t0 + timedelta(minutes=5 * i)) for i in range(30)]

    class _Seq:                      # bruto maior = mais anômalo
        input_kind = "sequence"
        calibration_set = HoldoutSet(X=np.zeros((30, 4, 1)), meta=meta, cutoff=t0)
        def score_val(self, X):
            return np.full(len(X), 1.0), np.zeros(len(X), dtype=bool), np.arange(len(X), dtype=float)

    class _Point:                    # IF: bruto menor = mais anômalo (normalizador invertido)
        input_kind = "point"
        _normalizer = SimpleNamespace(invert=True)
        def score_vectors(self, X):
            return np.full(len(X), 1.0), np.zeros(len(X), dtype=bool), -np.arange(len(X), dtype=float)

    calls = []
    class _FE:
        def extract_features(self, session, node_id, window_minutes, reference_time):
            calls.append((node_id, window_minutes, reference_time))
            return {name: 1.0 for name in FEATURE_NAMES}

    p._primary_runner, p._secondary_runner = _Point(), _Seq()
    ctx = DetectionContext(session=object(), node_id=1, feature_extractor=_FE(),
                           features_vec=None, network_node_ids=[1, 2], network_id=7)
    p.calibrate(ctx)
    # score_norm todo em 1 (o empate que saturava τ); os postos dos brutos não empatam:
    # posto_i = (i+1)/30 nos dois componentes → s_final = (i+1)/30 → p90 ≈ 0.9
    assert p.tau == pytest.approx(np.percentile(np.arange(1, 31) / 30, 90))
    assert p.tau < 1 and p.tau_status == "ok"
    assert p.tau_stats["frac_primary_score_norm_at_1"] == 1.0
    assert p._tau_calibrated is True
    assert calls[3] == (meta[3].node_id, p._config.if_window_minutes, meta[3].end)
    assert p.calibration["n_used"] == 30
    assert p.calibration["alignment"] == f"point_window_{p._config.if_window_minutes}min_at_sequence_end"


def test_fusion_combiner_returns_empty_when_tau_none():
    from ai.online.combiner import Combiner
    from ai.online.registry import AlgoResult
    c = Combiner("fusion")
    pr = AlgoResult(is_anomaly=True, score_norm=0.9, evidence={})
    sr = AlgoResult(is_anomaly=True, score_norm=0.9, evidence={})
    out = c.combine(primary=pr, secondary=sr, primary_id="isolation_forest",
                    secondary_id="lstm_ae", alpha=0.5, tau=None)
    assert out == []
