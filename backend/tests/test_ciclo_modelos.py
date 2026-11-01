"""Ciclo de vida dos modelos online para a campanha shadow/fusão/cascata."""
import json
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

NOW = datetime(2026, 10, 19, 12)
CUT = NOW - timedelta(hours=5)
NEUTRAL = dict(rssi_mean=-80., rssi_std=1., snr_mean=8., snr_std=1., pdr_mean=10.,
               battery_delta=0., packet_loss_rate=0., rssi_snr_divergence=0.,
               beacon_interval_std=1., route_change_rate=0., link_asymmetry=1.)


# Segundo componente pontual só para estes testes de ciclo de vida (pacotes, troca de
# modo, espera de retreino), que independem do algoritmo: um IF com outra semente.
SECOND = "if_teste"


@pytest.fixture(autouse=True)
def second_point_algorithm(monkeypatch):
    import ai.online.pipeline  # noqa: F401 - registro padrão
    import ai.online.registry as reg
    from ai.anomaly_ml import IsolationForestModel
    from ai.online.effective_config import resolve_effective_config
    from ai.online.runners import PointRunner

    def factory(ml_settings):
        cfg = resolve_effective_config(ml_settings)
        model = IsolationForestModel(contamination=cfg.if_contamination,
                                     n_estimators=cfg.if_n_estimators, random_state=7)
        return PointRunner(model=model, spec=reg.get(SECOND), config=cfg)

    monkeypatch.setitem(reg._registry, SECOND, reg.AlgorithmSpec(
        id=SECOND, label="IF (teste)", input_kind="point", scaling="self",
        incremental=True, factory=factory))


@pytest.fixture
def session():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    with Session(eng) as db:
        from models import MLSettings, Node
        db.add(Node(node_id=1, network_id=7, name="N1", status="online"))
        db.add(MLSettings(id=1))
        db.commit()
        yield db


def _settings(**over):
    from models import MLSettings
    base = dict(id=1, ml_online_primary="isolation_forest", ml_online_secondary=SECOND,
                ml_online_combo="shadow", ml_online_fusion_alpha=0.5)
    base.update(over)
    return MLSettings(**base)


def _detector(ml=None):
    from ai.anomaly_detector import AnomalyDetector
    ml = ml or _settings()
    d = object.__new__(AnomalyDetector)
    d._models_lock = threading.Lock()
    d._network_cycle_counts = {7: 1}
    d._network_models = {}
    d._online_pipelines = {}
    d._online_primary, d._online_secondary = ml.ml_online_primary, ml.ml_online_secondary
    d._online_combo, d._online_fusion_alpha = ml.ml_online_combo, ml.ml_online_fusion_alpha
    d._ml_if_contamination, d._ml_if_n_estimators = ml.ml_if_contamination, ml.ml_if_n_estimators
    d._feature_extractor = MagicMock()
    d.detect_simulated = True
    d.stats = {"total_detected": 0, "by_type": {}, "last_detection": None,
               "by_severity": {"low": 0, "medium": 0, "high": 0}}
    return d


def _trained_pair(ml=None):
    """Dois componentes pontuais treinados em dados sintéticos, com calibração de τ arquivada."""
    from ai.online.pipeline import OnlinePipeline
    ml = ml or _settings()
    p = OnlinePipeline("isolation_forest", SECOND, ml.ml_online_combo, 0.5)
    p.build_runners(ml)
    X = np.random.RandomState(0).normal(size=(200, 11))
    for runner in (p._primary_runner, p._secondary_runner):
        assert runner._model.train(X)
        runner._normalizer.fit(runner._model.score(X))
        runner._ready = True
    p.calibration = {"s_primary": list(np.linspace(0, 1, 100)), "s_secondary": [0.2] * 100,
                     "d_primary": [False] * 100, "d_secondary": [False] * 100,
                     "orientation": {"primary": -1.0, "secondary": -1.0},
                     "r_primary": list(np.linspace(0, 1, 100)), "r_secondary": list(np.linspace(1, 2, 100))}
    p._apply_tau()
    return p, ml, X


# ─── R1: tentativa de treino com espera e candidato fora do ciclo ────────────

def test_runner_waits_before_rebuilding_dataset_again(monkeypatch):
    from config import settings
    from ai.online.runners import SequenceRunner
    monkeypatch.setattr(settings, "ML_ONLINE_TRAIN_RETRY_SECONDS", 600)
    runner = SequenceRunner(model=object())
    calls = []
    monkeypatch.setattr(runner, "_train", lambda ctx: calls.append(ctx) or False)
    assert runner.ensure_trained("node-1") is False
    assert runner.ensure_trained("node-2") is False     # outro nó, mesmo ciclo: não refaz
    assert len(calls) == 1
    assert 0 < runner.retry_in_seconds() <= 600
    runner.allow_retry()
    runner.ensure_trained("node-3")
    assert len(calls) == 2


def test_candidate_trains_in_background_without_blocking_the_cycle(monkeypatch):
    import ai.online.pipeline as pmod
    d = _detector()
    d.engine = create_engine("sqlite://")
    d._training_context = MagicMock(return_value=object())
    gate = threading.Event()
    built = []
    candidate = MagicMock(combo="shadow", fusion_alpha=0.5)
    candidate.is_ready.return_value = True
    candidate.train.side_effect = lambda ctx: gate.wait(5)
    monkeypatch.setattr(pmod, "OnlinePipeline", lambda **kw: built.append(kw) or candidate)
    ml = _settings()
    assert d._get_pipeline_for_network(7, ml) is None     # ciclo segue sem esperar o treino
    assert d._get_pipeline_for_network(7, ml) is None
    assert len(built) == 1 and d._training(7)["status"] == "training"
    gate.set()
    d._training(7)["thread"].join(5)
    assert d._online_pipelines[7] is candidate and d._training(7)["status"] == "ready"


def test_failed_candidate_waits_but_manual_retrain_runs(monkeypatch):
    import ai.online.pipeline as pmod
    d = _detector()
    d._train_inline = True
    d.engine = create_engine("sqlite://")
    d._training_context = MagicMock(return_value=object())
    built = []
    candidate = MagicMock()
    candidate.is_ready.return_value = True
    candidate.train.return_value = False
    candidate._primary_runner.last_error = "insufficient_split: train=0 val=10"
    candidate._secondary_runner = None
    monkeypatch.setattr(pmod, "OnlinePipeline", lambda **kw: built.append(kw) or candidate)
    ml = _settings()
    assert d._rebuild_online_pipeline(7, ml, reason="initial") is False
    state = d._training(7)
    assert state["status"] == "failed" and "insufficient_split" in state["last_error"]
    assert d._rebuild_online_pipeline(7, ml, reason="periodic") is False
    assert len(built) == 1                                  # espera respeitada
    d.request_online_retrain(ml, [7])
    assert len(built) == 2                                  # manual ignora a espera


def test_frozen_models_skip_periodic_retrain(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "ML_ONLINE_FREEZE_MODELS", True)
    d = _detector()
    d._rebuild_online_pipeline = MagicMock()
    d._maybe_retrain_online(7, 60, _settings())
    d._rebuild_online_pipeline.assert_not_called()


# ─── R4: holdout e calibração de τ ──────────────────────────────────────────

def _fake_runner(kind, log, ok=True, calibration_set=None):
    runner = SimpleNamespace(input_kind=kind, last_error=None, calibration_set=calibration_set)
    runner.ensure_trained = lambda ctx: log.append((kind, ctx.reference_time)) or ok
    runner.is_ready = lambda: ok
    return runner


def _ctx(**over):
    from ai.online.registry import DetectionContext
    base = dict(session=object(), node_id=1, feature_extractor=object(), features_vec=None,
                network_node_ids=[1], network_id=7, reference_time=NOW)
    base.update(over)
    return DetectionContext(**base)


def test_point_component_trains_only_until_sequence_cutoff():
    from ai.online.pipeline import OnlinePipeline
    from ai.online.runners import HoldoutSet
    p = OnlinePipeline("isolation_forest", "lstm_ae", "shadow", 0.5)
    log = []
    p._secondary_runner = _fake_runner(
        "sequence", log, calibration_set=HoldoutSet(X=np.zeros((0, 4, 11)), meta=[], cutoff=CUT))
    p._primary_runner = _fake_runner("point", log)
    assert p.train(_ctx()) is True
    assert log == [("sequence", NOW), ("point", CUT)]       # holdout fora dos dois treinos
    assert p.tau is None and p.tau_status == "no_calibration_set"


def test_point_component_is_not_trained_when_sequence_fails():
    from ai.online.pipeline import OnlinePipeline
    p = OnlinePipeline("isolation_forest", "lstm_ae", "shadow", 0.5)
    log = []
    p._secondary_runner = _fake_runner("sequence", log, ok=False)
    p._primary_runner = _fake_runner("point", log)
    assert p.train(_ctx()) is False
    assert log == [("sequence", NOW)]


def _calibration(r_primary, r_secondary, s=1.0):
    n = len(r_primary)
    return {"s_primary": [s] * n, "s_secondary": [s] * n, "d_primary": [True] * n,
            "d_secondary": [True] * n, "orientation": {"primary": -1.0, "secondary": 1.0},
            "r_primary": list(r_primary), "r_secondary": list(r_secondary)}


def test_score_norm_ties_at_one_no_longer_saturate_tau():
    """Empates com os dois score_norm em 1 numa fração da validação maior que a
    contaminação (3%) levariam τ a 1. Na escala de posto, brutos distintos não empatam
    e τ fica < 1."""
    from ai.online.pipeline import OnlinePipeline
    p = OnlinePipeline("isolation_forest", "lstm_ae", "fusion", 0.5)
    p._contamination = 0.03
    rng = np.random.RandomState(1)
    p.calibration = _calibration(rng.normal(size=265), rng.normal(size=265), s=1.0)
    p._apply_tau()
    assert p.tau_status == "ok" and 0.5 < p.tau < 1
    assert p.tau_stats["frac_primary_score_norm_at_1"] == 1.0


def test_saturated_tau_is_unavailable_and_explicit():
    from ai.online.combiner import Combiner
    from ai.online.pipeline import OnlinePipeline
    from ai.online.registry import AlgoResult
    p = OnlinePipeline("isolation_forest", "lstm_ae", "fusion", 0.5)
    p._contamination = 0.03
    p.calibration = _calibration([5.0] * 100, [3.0] * 100)      # brutos todos empatados
    p._apply_tau()
    assert p.tau is None and p.tau_status == "saturated"
    assert p.tau_stats["frac_fused_at_max"] == pytest.approx(1.0)
    both = AlgoResult(is_anomaly=True, score_norm=1.0)
    outcome = Combiner("fusion").evaluate(both, both, alpha=0.5, tau=p.tau)
    assert outcome.status == "unavailable" and outcome.reason == "tau_unavailable"


# ─── R3: troca de modo sem retreino ─────────────────────────────────────────

def test_set_combination_reuses_models_and_recomputes_tau():
    from ai.online.pipeline import OnlinePipeline
    p = OnlinePipeline("isolation_forest", "lstm_ae", "shadow", 0.5)
    p._primary_runner, p._secondary_runner = object(), object()
    p._contamination = 0.1
    p.calibration = _calibration(-np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    p._apply_tau()
    tau_half = p.tau
    runners = (p._primary_runner, p._secondary_runner)
    p.set_combination("fusion", 1.0)
    assert (p._primary_runner, p._secondary_runner) == runners
    assert p.combo == "fusion" and p._combiner.combo == "fusion"
    assert p.tau == pytest.approx(0.9, abs=0.01) and p.combo == "fusion"
    with pytest.raises(ValueError):
        p.set_combination("single", 0.5)


def test_apply_settings_switches_mode_without_retraining():
    from ai.online.pipeline import OnlinePipeline
    initial = _settings()
    d = _detector(initial)
    pipe = OnlinePipeline("isolation_forest", SECOND, "shadow", 0.5)
    pipe.build_runners(initial)
    d._online_pipelines = {7: pipe}
    d._rebuild_online_pipeline = MagicMock()
    runners = (pipe._primary_runner, pipe._secondary_runner)
    d.apply_settings(_settings(ml_online_combo="cascade"))
    d._rebuild_online_pipeline.assert_not_called()
    assert pipe.combo == "cascade" and (pipe._primary_runner, pipe._secondary_runner) == runners
    d.apply_settings(_settings(ml_online_combo="cascade", ml_if_n_estimators=10))
    d._rebuild_online_pipeline.assert_called_once()
    assert d._online_pipelines[7] is pipe             # anterior segue até o candidato ficar pronto


# ─── R2: pacotes congelados ─────────────────────────────────────────────────

def test_bundle_roundtrip_restores_models_and_calibration(tmp_path):
    from ai.online.bundles import load_latest_bundle, save_bundle
    from ai.online.effective_config import resolve_effective_config
    from ai.online.pipeline import OnlinePipeline
    p, ml, X = _trained_pair()
    manifest = save_bundle(p, 7, str(tmp_path), "manual")
    assert manifest["payload_sha256"] and manifest["feature_schema_version"]
    loaded_manifest, bundle = load_latest_bundle(7, p.model_signature, str(tmp_path))
    q = OnlinePipeline.from_bundle(bundle, ml, "cascade", 0.5)
    np.testing.assert_allclose(q._primary_runner.score_vectors(X[:5])[0],
                               p._primary_runner.score_vectors(X[:5])[0])
    np.testing.assert_allclose(q._secondary_runner.score_vectors(X[:5])[0],
                               p._secondary_runner.score_vectors(X[:5])[0])
    assert q.tau == p.tau and q.combo == "cascade" and loaded_manifest == manifest
    other = resolve_effective_config(_settings(ml_if_n_estimators=7))
    assert load_latest_bundle(7, other.models_signature("isolation_forest", SECOND),
                              str(tmp_path)) is None
    (tmp_path / "network_7" / manifest["bundle_id"] / "bundle.joblib").write_bytes(b"x")
    assert load_latest_bundle(7, p.model_signature, str(tmp_path)) is None


def test_bundle_accepts_old_version_prefix_but_rejects_other_feature_versions(tmp_path):
    from ai.online.bundles import load_latest_bundle, save_bundle
    p, ml, X = _trained_pair()
    manifest = save_bundle(p, 7, str(tmp_path), "manual")
    path = tmp_path / "network_7" / manifest["bundle_id"] / "manifest.json"
    manifest["feature_schema_version"] = "legacy065-" + manifest["feature_schema_version"]
    path.write_text(json.dumps(manifest))
    assert load_latest_bundle(7, p.model_signature, str(tmp_path)) is not None
    manifest["feature_schema_version"] = "legacy065-features-v1"
    path.write_text(json.dumps(manifest))
    assert load_latest_bundle(7, p.model_signature, str(tmp_path)) is None



def test_bundle_loader_refuses_redirected_payloads(tmp_path):
    """O manifesto não escolhe o arquivo carregado e links simbólicos não são seguidos."""
    import json
    import shutil
    from ai.online.bundles import load_latest_bundle, save_bundle
    p, _, _ = _trained_pair()
    manifest = save_bundle(p, 7, str(tmp_path), "manual")
    folder = tmp_path / "network_7" / manifest["bundle_id"]
    outside = tmp_path / "fora.joblib"
    shutil.copy(folder / "bundle.joblib", outside)

    data = json.loads((folder / "manifest.json").read_text())
    data["payload"] = "../../fora.joblib"
    (folder / "manifest.json").write_text(json.dumps(data))
    assert load_latest_bundle(7, p.model_signature, str(tmp_path)) is None

    data["payload"] = "bundle.joblib"
    (folder / "manifest.json").write_text(json.dumps(data))
    (folder / "bundle.joblib").unlink()
    assert load_latest_bundle(7, p.model_signature, str(tmp_path)) is None  # ausente: ignora
    (folder / "bundle.joblib").symlink_to(outside)
    assert load_latest_bundle(7, p.model_signature, str(tmp_path)) is None

def test_frozen_startup_publishes_saved_bundle_without_training(tmp_path, monkeypatch):
    from config import settings
    from ai.online.bundles import save_bundle
    monkeypatch.setattr(settings, "ML_ONLINE_FREEZE_MODELS", True)
    monkeypatch.setattr(settings, "ML_ONLINE_BUNDLE_DIR", str(tmp_path))
    p, ml, _ = _trained_pair()
    manifest = save_bundle(p, 7, str(tmp_path), "initial")
    d = _detector(ml)
    d._rebuild_online_pipeline = MagicMock()
    published = d._get_pipeline_for_network(7, ml)
    d._rebuild_online_pipeline.assert_not_called()
    assert published.bundle_id == manifest["bundle_id"] and published.tau == p.tau


def test_publication_saves_bundle_when_frozen(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "ML_ONLINE_FREEZE_MODELS", True)
    monkeypatch.setattr(settings, "ML_ONLINE_BUNDLE_DIR", str(tmp_path))
    p, ml, _ = _trained_pair()
    d = _detector(ml)
    d._publish_pipeline(7, p, "initial")
    assert d._online_pipelines[7] is p and p.bundle_id
    assert (tmp_path / "network_7" / p.bundle_id / "manifest.json").is_file()


# ─── R5/R6/R8: registro por ciclo, deduplicação por modo e estado ───────────

def _published(outcome, **over):
    attrs = dict(fusion_alpha=0.5, tau=0.6, tau_status="ok", bundle_id="b1",
                 model_signature=("sig",), _secondary_runner=SimpleNamespace(input_kind="sequence"),
                 fusion_ranks=lambda primary, secondary: (0.9, 0.4))
    attrs.update(over)
    return SimpleNamespace(detect_detailed=lambda ctx: outcome, **attrs)


def _shadow_outcome():
    from ai.online.combiner import Combiner
    from ai.online.pipeline import PipelineOutcome
    from ai.online.registry import AlgoResult
    primary = AlgoResult(is_anomaly=True, score_norm=0.9, raw_score=-0.6, evidence={"raw_score": -0.6})
    secondary = AlgoResult(is_anomaly=False, score_norm=0.4, raw_score=0.5, threshold=0.8,
                           evidence={"context_end": "2026-10-19T11:59:00"})
    combiner = Combiner("shadow")
    return PipelineOutcome("isolation_forest", "lstm_ae", "shadow", primary, secondary,
                           combiner.evaluate(primary, secondary),
                           combiner.combine(primary, secondary, "isolation_forest", "lstm_ae"))


def test_cycle_logs_both_components_and_counterfactual_fusion(session):
    from models import OnlineInference
    d = _detector()
    d._get_pipeline_for_network = MagicMock(return_value=_published(_shadow_outcome()))
    ml_det = MagicMock()
    ml_det.feature_extractor.extract_features.return_value = NEUTRAL | {"rssi_mean": -105.}
    d._run_online_pipeline(session, 7, ml_det.feature_extractor, [1], None, reference_time=NOW)
    row = session.exec(select(OnlineInference)).one()
    assert (row.primary_is_anomaly, row.secondary_is_anomaly) == (True, False)
    assert row.secondary_threshold == 0.8 and row.sequence_context_end == "2026-10-19T11:59:00"
    assert row.fusion_score == pytest.approx(0.65) and row.fusion_decision is True
    assert (row.primary_rank, row.secondary_rank) == (0.9, 0.4)
    assert row.combo == "shadow" and row.combination_status == "not_applicable"
    assert row.emitted == 1 and json.loads(row.outcomes_json)[0]["status"] == "saved"
    assert json.loads(row.rule_hits) == ["A1"] and row.bundle_id == "b1"
    assert d.online_status()["networks"][7]["last_cycle"]["emitted"] == 1


def test_cycle_without_pipeline_logs_unavailability(session):
    from models import OnlineInference
    d = _detector()
    d._get_pipeline_for_network = MagicMock(return_value=None)
    d._training(7).update(status="training")
    d._run_online_pipeline(session, 7, MagicMock(), [1], None, reference_time=NOW)
    row = session.exec(select(OnlineInference)).one()
    assert row.pipeline_status == "no_pipeline" and row.training_status == "training"


def test_dedup_does_not_cross_modes(session):
    from models import Anomaly
    d = _detector()
    event = {"anomaly_type": "A2", "category": "radio", "severity": "high",
             "description": "perda", "evidence": {}}
    first = d._save_and_notify(session, node_id=1, anomaly=dict(event),
                               detector_algorithm="shadow:if", detector_combo="shadow")
    cascade = d._save_and_notify(session, node_id=1, anomaly=dict(event),
                                 detector_algorithm="cascade:if>lstm", detector_combo="cascade")
    repeat = d._save_and_notify(session, node_id=1, anomaly=dict(event),
                                detector_algorithm="cascade:if>lstm", detector_combo="cascade")
    assert first["status"] == "saved" and cascade["status"] == "saved"
    assert repeat == {"status": "dedup", "duplicate_of": cascade["anomaly_id"]}
    assert len(session.exec(select(Anomaly)).all()) == 2


def test_online_status_exposes_tau_runners_and_training():
    p, _, _ = _trained_pair()
    d = _detector()
    d._online_pipelines = {7: p}
    d._training(7).update(status="ready")
    status = d.online_status()
    published = status["networks"][7]["published"]
    assert published["tau_status"] == "ok" and published["primary"]["ready"] is True
    assert status["networks"][7]["training"]["status"] == "ready"
    assert status["selection"]["combo"] == "shadow" and status["frozen"] is False
    json.dumps(status)                                      # serializável pela API


def test_live_fusion_uses_calibration_ranks():
    """A inferência converte o bruto de cada componente no posto da validação."""
    from ai.online.combiner import Combiner
    from ai.online.pipeline import OnlinePipeline
    from ai.online.registry import AlgoResult
    p = OnlinePipeline("isolation_forest", "lstm_ae", "fusion", 0.5)
    p._contamination = 0.1
    # validação já orientada (maior = mais anômalo): IF brutos 0..−99 → 0..99
    p.calibration = _calibration(np.arange(100.0), np.arange(100.0))
    p._apply_tau()
    # IF bruto −95 (menor = mais anômalo) → posto 0.96; LSTM bruto 500 > máximo → posto 1
    if_result = AlgoResult(is_anomaly=True, score_norm=1.0, raw_score=-95.0)
    lstm_result = AlgoResult(is_anomaly=True, score_norm=1.0, raw_score=500.0, threshold=99.0)
    assert p.fusion_ranks(if_result, lstm_result) == pytest.approx((0.96, 1.0))
    [event] = Combiner("fusion").combine(if_result, lstm_result, "isolation_forest", "lstm_ae",
                                         alpha=0.5, tau=p.tau,
                                         fusion_scores=p.fusion_ranks(if_result, lstm_result))
    assert event["evidence"]["fusion_scale"] == "calibration_rank"
    assert event["evidence"]["s_fusion"] == pytest.approx(0.98)
    median = AlgoResult(is_anomaly=False, score_norm=0.5, raw_score=-50.0)
    assert Combiner("fusion").combine(median, lstm_result, "isolation_forest", "lstm_ae",
                                      alpha=0.5, tau=p.tau,
                                      fusion_scores=p.fusion_ranks(median, lstm_result)) == []

