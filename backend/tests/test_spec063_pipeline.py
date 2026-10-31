"""Pipeline online."""
from unittest.mock import MagicMock

import numpy as np
import pytest


def _ctx():
    from ai.online.registry import DetectionContext
    return DetectionContext(session=MagicMock(), node_id=1, feature_extractor=MagicMock(),
                            features_vec=np.zeros(11), network_node_ids=[1], network_id=1)


def _runner(trained=True, result=None, exc=None):
    from ai.online.registry import AlgoResult
    r = MagicMock()
    r.last_error = "insufficient_reference: 3 < 10"
    r.ensure_trained.return_value = trained
    if exc is not None:
        r.score_one.side_effect = exc
    else:
        r.score_one.return_value = result or AlgoResult(is_anomaly=True, score_norm=0.9, evidence={})
    return r


@pytest.mark.parametrize("order", ["if_fails", "lstm_fails"])
def test_shadow_evaluates_each_component_independently(order):
    from ai.online.pipeline import OnlinePipeline
    p = OnlinePipeline("isolation_forest", "lstm_ae", "shadow", 0.5)
    ok, bad = _runner(), _runner(trained=False)
    p._primary_runner, p._secondary_runner = (bad, ok) if order == "if_fails" else (ok, bad)
    out = p.detect_detailed(_ctx())
    failing = out.primary if order == "if_fails" else out.secondary
    working = out.secondary if order == "if_fails" else out.primary
    assert failing.status == "warming_up" and failing.reason == "insufficient_reference: 3 < 10"
    assert working.is_valid and working.is_anomaly is True
    assert len(out.anomalies) == 1


def test_runner_exception_becomes_error_and_other_still_scored():
    from ai.online.pipeline import OnlinePipeline
    p = OnlinePipeline("isolation_forest", "lstm_ae", "shadow", 0.5)
    p._primary_runner = _runner(exc=RuntimeError("boom"))
    p._secondary_runner = _runner()
    out = p.detect_detailed(_ctx())
    assert out.primary.status == "error" and "boom" in out.primary.reason
    assert out.secondary.is_valid


def test_cascade_with_unavailable_lstm_is_unavailable_not_if_alert():
    from ai.online.pipeline import OnlinePipeline
    from ai.online.registry import AlgoResult
    p = OnlinePipeline("isolation_forest", "lstm_ae", "cascade", 0.5)
    p._primary_runner = _runner()
    p._secondary_runner = _runner(result=AlgoResult.unavailable("insufficient_history", "missing_initial_context"))
    out = p.detect_detailed(_ctx())
    assert out.combination.status == "unavailable"
    assert out.primary.is_anomaly is True          # alerta IF preservado no resultado
    assert out.anomalies == []


def test_build_runners_uses_effective_config_and_sets_signature(monkeypatch):
    from ai.online.pipeline import OnlinePipeline
    import ai.online.registry as reg
    seen = {}
    spec = MagicMock()
    spec.factory = lambda cfg: seen.setdefault("cfg", cfg) and MagicMock()
    monkeypatch.setitem(reg._registry, "isolation_forest", spec)
    p = OnlinePipeline("isolation_forest", None, "single", 0.5)
    p.build_runners(type("ML", (), {"ml_if_contamination": 0.07, "ml_lstm_window": 6})())
    assert p.is_ready()
    from ai.online.effective_config import EffectiveMLConfig
    assert isinstance(seen["cfg"], EffectiveMLConfig) and seen["cfg"].if_contamination == 0.07
    assert p.signature == seen["cfg"].pipeline_signature("isolation_forest", None, "single", 0.5)


def test_build_runners_rejects_invalid_config(monkeypatch):
    from ai.online.pipeline import OnlinePipeline
    p = OnlinePipeline("isolation_forest", None, "single", 0.5)
    p.build_runners(type("ML", (), {"ml_lstm_threshold_strategy": "dynamic"})())
    assert p.is_ready() is False and p.signature is None


def test_factories_propagate_lstm_parameters():
    pytest.importorskip("torch")
    from ai.online.pipeline import _register_defaults  # noqa: F401
    from ai.online.registry import get
    ml = type("ML", (), {"ml_lstm_window": 6, "ml_lstm_hidden": 16, "ml_lstm_epochs": 7,
                         "ml_lstm_patience": 3, "ml_lstm_threshold_percentile": 97.5,
                         "ml_lstm_score_aggregation": "max"})()
    runner = get("lstm_ae").factory(ml)
    m = runner._model
    assert (m.window_size, m.hidden_size, m.epochs, m.patience) == (6, 16, 7, 3)
    assert m.threshold_percentile == 97.5 and m.score_aggregation == "max"
    if_runner = get("isolation_forest").factory(type("ML", (), {"ml_if_contamination": 0.1,
                                                                 "ml_if_n_estimators": 60})())
    assert (if_runner._model.contamination, if_runner._model.n_estimators) == (0.1, 60)
