"""Runners online."""
from collections import Counter
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

T = datetime(2026, 9, 1, 12, 0, 0)


def _cfg(**over):
    from ai.online.effective_config import resolve_effective_config
    base = dict(ml_if_contamination=0.1, ml_if_n_estimators=30, ml_lstm_window=4,
                ml_lstm_epochs=2, ml_lstm_hidden=8)
    base.update(over)
    return resolve_effective_config(type("ML", (), base)())


def _ctx(session, fe, vec=None, policy=None, reason=None):
    from ai.online.registry import DetectionContext
    from ai.reference_policy import ReferencePolicy
    return DetectionContext(session=session, node_id=1, feature_extractor=fe, features_vec=vec,
                            network_node_ids=[1, 2], network_id=7, reference_time=T,
                            reference_policy=policy or ReferencePolicy.empty(),
                            features_reason=reason)


class _Session:
    def get(self, model, node_id):
        return SimpleNamespace(sim_mode=False)


class _PointFE:
    from ai.sequence_builder import FEATURE_NAMES as feature_names  # Nomes canônicos

    def __init__(self, per_node=8):
        self.per_node = per_node
        self.kwargs = []

    def extract_sliding_windows(self, session, node_id, **kw):
        self.kwargs.append(kw)
        rng = np.random.default_rng(node_id)
        return [{n: float(v) for n, v in zip(self.feature_names, rng.normal(0, 1, 11))}
                for _ in range(self.per_node)]

    def get_feature_matrix(self, batch):
        return np.array([[f[n] for n in self.feature_names] for _, f in batch])


def _if_runner(cfg):
    from ai.online.runners import PointRunner
    from ai.online.registry import AlgorithmSpec
    from ai.anomaly_ml import IsolationForestModel
    spec = AlgorithmSpec(id="isolation_forest", label="IF", input_kind="point",
                         scaling="self", incremental=True, factory=lambda c: None)
    model = IsolationForestModel(contamination=cfg.if_contamination,
                                 n_estimators=cfg.if_n_estimators)
    return PointRunner(model=model, spec=spec, config=cfg)


def test_point_runner_uses_effective_minimum_and_cutoff():
    cfg = _cfg()
    fe = _PointFE(per_node=4)                     # 8 janelas < mínimo 10
    r = _if_runner(cfg)
    assert r.ensure_trained(_ctx(_Session(), fe)) is False
    assert r.last_error == "insufficient_reference: 8 < 10"
    assert fe.kwargs[0]["end_time"] == T and fe.kwargs[0]["window_minutes"] == 15
    assert fe.kwargs[0]["history_hours"] == 24.0 and fe.kwargs[0]["reference_policy"] is not None


def test_point_runner_ready_only_after_model_and_normalizer(monkeypatch):
    cfg = _cfg()
    r = _if_runner(cfg)
    import ai.online.runners as mod

    class _BadNorm(mod.ScoreNormalizer):
        def fit(self, scores):
            raise ValueError("normalizador falhou")

    monkeypatch.setattr(mod, "ScoreNormalizer", _BadNorm)
    ctx = _ctx(_Session(), _PointFE())
    assert r.ensure_trained(ctx) is False
    assert r.is_ready() is False
    assert "normalizador falhou" in r.last_error
    res = r.score_one(_ctx(_Session(), _PointFE(), vec=np.zeros(11)))
    assert res.status == "warming_up"                      # nada de fallback sigmoid
    monkeypatch.undo()
    # Sem nova tentativa imediata após falha (espera por runner)
    assert r.ensure_trained(ctx) is False and "normalizador falhou" in r.last_error
    r.allow_retry()
    assert r.ensure_trained(ctx) is True and r.is_ready()


def test_point_runner_unavailable_features_and_non_finite():
    cfg = _cfg()
    r = _if_runner(cfg)
    assert r.ensure_trained(_ctx(_Session(), _PointFE())) is True
    res = r.score_one(_ctx(_Session(), _PointFE(), vec=None, reason="no_data"))
    assert res.status == "insufficient_history" and res.reason == "no_data"
    bad = np.zeros(11); bad[2] = np.nan
    assert r.score_one(_ctx(_Session(), _PointFE(), vec=bad)).status == "error"
    ok = r.score_one(_ctx(_Session(), _PointFE(), vec=np.zeros(11)))
    assert ok.is_valid
    assert ok.evidence["margin"] < 0 if ok.is_anomaly else ok.evidence["margin"] >= 0
    assert "offset" in ok.evidence


def test_sequence_runner_no_sequence_is_insufficient_history_not_zero():
    pytest.importorskip("torch")
    from ai.online.runners import SequenceRunner
    from ai.lstm_autoencoder import LSTMAutoencoderModel
    cfg = _cfg()
    r = SequenceRunner(model=LSTMAutoencoderModel(window_size=4, hidden_size=8, epochs=2), config=cfg)
    r._ready = True
    r._model._is_trained = True

    class _FE:
        def extract_for_window(self, session, node_id, s, e, **kw):
            return None
    res = r.score_one(_ctx(None, _FE()))
    assert res.status == "insufficient_history" and res.reason == "missing_initial_context"
    assert res.score_norm is None and res.is_anomaly is None


def test_sequence_runner_trains_with_reference_dataset_and_global_split(monkeypatch):
    pytest.importorskip("torch")
    from ai.online.runners import SequenceRunner
    from ai.lstm_autoencoder import LSTMAutoencoderModel
    from ai.sequence_builder import SequenceBuilder
    cfg = _cfg(ml_lstm_history_hours=10.0)
    seen = {}
    real_build = SequenceBuilder.build_reference_dataset

    def _spy(self, session, node_ids, fe, end_time, history_hours, reference_policy):
        seen.update(end_time=end_time, history=history_hours, W=self.window_size,
                    gap=self.max_impute_gap, policy=reference_policy)
        return real_build(self, session, node_ids, fe, end_time, history_hours, reference_policy)

    monkeypatch.setattr(SequenceBuilder, "build_reference_dataset", _spy)

    class _FE:
        def extract_for_window(self, session, node_id, s, e, provenance=None):
            if provenance is not None:
                provenance.update({"synthetic_rows": 0, "source_node_ids": {node_id},
                                   "defaulted_features": []})
            rng = np.random.default_rng(int(s.timestamp()) % 1000 + node_id)
            from ai.sequence_builder import FEATURE_NAMES
            return {k: float(v) for k, v in zip(FEATURE_NAMES, rng.normal(0, 1, 11))}

    r = SequenceRunner(model=LSTMAutoencoderModel(window_size=4, hidden_size=8, epochs=2), config=cfg)
    assert r.ensure_trained(_ctx(object(), _FE())) is True
    assert seen["end_time"] == T and seen["history"] == 10.0 and seen["W"] == 4 and seen["gap"] == 2
    assert r.training_summary["cutoff"] is not None and r.training_summary["n_train"] >= 10


def test_sequence_runner_non_finite_threshold_is_error_and_cascade_unavailable():
    """Limiar NaN injetado não pode virar normalidade válida."""
    pytest.importorskip("torch")
    from ai.online.runners import SequenceRunner
    from ai.online.combiner import Combiner
    from ai.online.registry import AlgoResult
    from ai.lstm_autoencoder import LSTMAutoencoderModel
    from ai.sequence_builder import FEATURE_NAMES
    m = LSTMAutoencoderModel(window_size=4, hidden_size=8, epochs=2)
    X = np.random.default_rng(0).normal(0, 1, (40, 4, 11)).astype(np.float32)
    assert m.train(X[:32], X_val=X[32:])
    m._threshold = float("nan")                       # injeção controlada
    r = SequenceRunner(model=m, config=_cfg())
    r._ready = True
    r._normalizer.fit(m.score_batch(X[:32]))

    class _FE:
        def extract_for_window(self, session, node_id, s, e, **kw):
            return {k: 0.0 for k in FEATURE_NAMES}
    res = r.score_one(_ctx(None, _FE()))
    assert res.status == "error" and res.is_anomaly is None and res.score_norm is None
    cascade = Combiner("cascade").evaluate(
        AlgoResult(is_anomaly=True, score_norm=0.9, evidence={}), res)
    assert cascade.status == "unavailable"
