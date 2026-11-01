"""Limiar do IF, metadados da janela e decisão contrafactual da fusão."""
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

NOW = datetime(2026, 10, 21, 12)
FEATURES = dict(rssi_mean=-75.0, rssi_std=1.0, snr_mean=8.0, snr_std=1.0, pdr_mean=8.0,
                battery_delta=0.0, packet_loss_rate=0.0, rssi_snr_divergence=0.0,
                beacon_interval_std=1.0, route_change_rate=0.0, link_asymmetry=0.0)
REQUIRED = {"_unavailable", "_window_minutes", "_rules_version", "_definitions_version",
            "_feature_schema_version", "_registro_version"}


def _ctx(vector):
    from ai.online.registry import DetectionContext
    return DetectionContext(session=None, node_id=1, feature_extractor=None,
                            features_vec=vector, network_node_ids=[1], network_id=1,
                            reference_time=NOW)


def _ready_runner(model, training):
    from ai.online.registry import AlgorithmSpec
    from ai.online.runners import PointRunner
    spec = AlgorithmSpec(id="isolation_forest", label="IF", input_kind="point",
                         scaling="self", incremental=True, factory=lambda cfg: None)
    runner = PointRunner(model, spec)
    runner._normalizer.fit(model.score(training))
    runner._ready = True
    return runner


@pytest.fixture
def if_results():
    from ai.anomaly_ml import IsolationForestModel
    rng = np.random.default_rng(69)
    training = rng.normal(size=(64, 11))
    model = IsolationForestModel(contamination=0.1, n_estimators=16)
    assert model.train(training)
    runner = _ready_runner(model, training)
    vectors = np.vstack([training[:20], rng.normal(scale=6.0, size=(10, 11))])
    return model, [runner.score_one(_ctx(vector)) for vector in vectors]


# ─── R1 ─────────────────────────────────────────────────────────────────────

def test_t9_if_threshold_recorded_as_offset(if_results):
    from ai.anomaly_detector import _component_fields
    model, results = if_results
    assert all(r.is_valid for r in results)
    assert {bool(r.is_anomaly) for r in results} == {True, False}
    for result in results:
        fields = _component_fields("primary", result)
        assert fields["primary_threshold"] == model.offset
        assert (fields["primary_raw_score"] < fields["primary_threshold"]) == \
            fields["primary_is_anomaly"]


def test_r1_keeps_algo_result_and_evidence_unchanged(if_results):
    from ai.online.combiner import _component_summary
    _, results = if_results
    assert all(r.threshold is None for r in results)
    assert all(_component_summary("isolation_forest", r)["threshold"] is None for r in results)


class _NoOffsetModel:
    def predict(self, X):
        return np.where(np.asarray(X, dtype=float)[:, 0] > 0, -1, 1)

    def score(self, X):
        return -np.asarray(X, dtype=float)[:, 0]


def test_r1_model_without_offset_keeps_threshold_null():
    from ai.anomaly_detector import _component_fields
    training = np.random.default_rng(1).normal(size=(32, 11))
    runner = _ready_runner(_NoOffsetModel(), training)
    result = runner.score_one(_ctx(training[0]))
    assert result.is_valid
    assert _component_fields("primary", result)["primary_threshold"] is None


def test_r1_unavailable_and_own_threshold():
    from ai.anomaly_detector import _component_fields
    from ai.online.registry import AlgoResult
    missing = AlgoResult.unavailable("warming_up", "not_trained", evidence={"offset": -0.5})
    assert _component_fields("primary", missing)["primary_threshold"] is None
    lstm = AlgoResult(is_anomaly=False, score_norm=0.4, raw_score=0.2, threshold=0.5,
                      evidence={"offset": -0.5})
    assert _component_fields("secondary", lstm)["secondary_threshold"] == 0.5


# ─── R2 e R3 ────────────────────────────────────────────────────────────────

def _pipeline(ranks=(0.9, 0.4), alpha=0.5, tau=0.6, outcome=None):
    attrs = dict(fusion_alpha=alpha, tau=tau, tau_status="ok", bundle_id="b1",
                 model_signature=("sig",), _secondary_runner=SimpleNamespace(input_kind="sequence"),
                 fusion_ranks=lambda p, s: ranks)
    if outcome is not None:
        attrs["detect_detailed"] = lambda ctx: outcome
    return SimpleNamespace(**attrs)


def _outcome(combo="shadow", primary=None, secondary=None, alpha=0.5, tau=None, ranks=None):
    from ai.online.combiner import Combiner
    from ai.online.pipeline import PipelineOutcome
    from ai.online.registry import AlgoResult
    primary = primary or AlgoResult(is_anomaly=False, score_norm=0.2, raw_score=0.1)
    secondary = secondary or AlgoResult(is_anomaly=False, score_norm=0.3, raw_score=0.1,
                                        threshold=0.5)
    combination = Combiner(combo).evaluate(primary, secondary, alpha, tau, ranks)
    return PipelineOutcome("isolation_forest", "lstm_ae", combo, primary, secondary,
                           combination, [])


def test_t10_features_json_records_window_metadata():
    from ai.anomaly_definitions import DEFINITIONS_VERSION, RULES_VERSION
    from ai.anomaly_detector import _inference_row
    from ai.feature_engineering import FEATURE_SCHEMA_VERSION
    features = FEATURES | {"_unavailable": {"battery_delta": "menos de duas leituras"},
                           "_link_stats": {"nao_persistir": True}, "_battery_first_at": "x"}
    row = _inference_row(1, 1, NOW, _pipeline(), _outcome(), features, None, [],
                         window_minutes=15)
    stored = json.loads(row.features_json)
    assert REQUIRED <= stored.keys()
    assert set(stored) - REQUIRED == set(FEATURES)
    assert stored["_unavailable"] == {"battery_delta": "menos de duas leituras"}
    assert stored["_window_minutes"] == 15
    assert (stored["_rules_version"], stored["_definitions_version"],
            stored["_feature_schema_version"]) == (RULES_VERSION, DEFINITIONS_VERSION,
                                                   FEATURE_SCHEMA_VERSION)
    assert stored["_registro_version"] == "registro-v1"


def test_t10_empty_unavailable_and_missing_features():
    from ai.anomaly_detector import _inference_row
    row = _inference_row(1, 1, NOW, _pipeline(), _outcome(), dict(FEATURES), None, [],
                         window_minutes=15)
    assert json.loads(row.features_json)["_unavailable"] == {}
    row = _inference_row(1, 1, NOW, _pipeline(), _outcome(), None, "no_point_features", [],
                         window_minutes=15)
    assert row.features_json is None and row.features_reason == "no_point_features"


@pytest.fixture
def session():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    with Session(eng) as db:
        from models import MLSettings, Node
        db.add(Node(node_id=1, network_id=7, name="N1", status="online"))
        db.add(MLSettings(id=1))
        db.commit()
        yield db


def test_t10_pipeline_records_effective_window(session, monkeypatch):
    import ai.online.effective_config as effective
    from models import OnlineInference
    from tests.test_ciclo_modelos import _detector
    monkeypatch.setattr(effective, "resolve_effective_config",
                        lambda ml: SimpleNamespace(if_window_minutes=20, lstm_window_size=12,
                                                     lstm_window_minutes=5))
    d = _detector()
    d._get_pipeline_for_network = MagicMock(return_value=_pipeline(outcome=_outcome()))
    ml_det = MagicMock()
    ml_det.feature_extractor.extract_features.return_value = dict(FEATURES)
    d._run_online_pipeline(session, 7, ml_det.feature_extractor, [1], None, reference_time=NOW)
    row = session.exec(select(OnlineInference)).one()
    assert json.loads(row.features_json)["_window_minutes"] == 20
    assert ml_det.feature_extractor.extract_features.call_args.kwargs["window_minutes"] == 20


@pytest.mark.parametrize("rank, expected", [(0.6000004, True), (0.6, False), (0.5999996, False)])
def test_t10b_fusion_decision_uses_unrounded_score(rank, expected):
    from ai.anomaly_detector import _inference_row
    from ai.online.registry import AlgoResult
    primary = AlgoResult(is_anomaly=True, score_norm=0.8, raw_score=-0.7)
    secondary = AlgoResult(is_anomaly=False, score_norm=0.4, raw_score=0.2, threshold=0.5)
    outcome = _outcome("fusion", primary, secondary, alpha=0.5, tau=0.6, ranks=(rank, rank))
    row = _inference_row(1, 1, NOW, _pipeline(ranks=(rank, rank), tau=0.6), outcome,
                         dict(FEATURES), None, [], window_minutes=15)
    assert row.combination_is_anomaly is expected
    assert row.fusion_decision is expected
    assert row.fusion_score == round(rank, 6)
