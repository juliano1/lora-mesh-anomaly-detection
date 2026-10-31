"""Seleção do pipeline de detecção: o pipeline online prevalece sobre a flag híbrida
legada (ml_use_hybrid), e o retreino só publica pipelines treinados e prontos."""

import threading
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        from models import MLSettings, Node

        db.add(Node(node_id=1, network_id=7, name="N1", status="online"))
        db.add(MLSettings(id=1))
        db.commit()
        yield db


def _anomaly(algorithm=None):
    result = {
        "anomaly_type": "A1",
        "category": "signal",
        "severity": "medium",
        "description": "anomalia de teste",
        "detection_method": "ml",
        "confidence": 80,
    }
    if algorithm:
        result["detector_algorithm"] = algorithm
    return result


def _detector(use_hybrid):
    from ai.anomaly_detector import AnomalyDetector
    from ai.sequence_builder import FEATURE_NAMES

    detector = object.__new__(AnomalyDetector)
    detector._models_lock = threading.Lock()
    detector._network_cycle_counts = {7: 1}
    detector._online_combo = "cascade"
    detector._online_primary = "isolation_forest"
    detector.detect_simulated = True
    detector.fast_detection_enabled = False
    detector.stats = {
        "total_detected": 0,
        "by_type": {},
        "by_severity": {"low": 0, "medium": 0, "high": 0},
        "last_detection": None,
    }

    ml_detector = MagicMock()
    ml_detector.model.is_trained = True
    ml_detector.feature_extractor.extract_features.return_value = {
        name: 0.0 for name in FEATURE_NAMES
    }
    detector._feature_extractor = ml_detector.feature_extractor
    return detector, ml_detector


def _outcome(anomalies=()):
    """O laço consome detect_detailed (componentes + combinação + emissões)."""
    from ai.online.combiner import ComboOutcome
    from ai.online.pipeline import PipelineOutcome
    return PipelineOutcome("isolation_forest", None, "single", None, None,
                           ComboOutcome("valid"), list(anomalies))


def _published(**over):
    """Dublê do pipeline publicado, com os atributos lidos pelo registro por ciclo."""
    from unittest.mock import MagicMock
    attrs = dict(fusion_alpha=0.5, tau=None, tau_status=None, bundle_id=None,
                 model_signature=None, _secondary_runner=None)
    attrs.update(over)
    return MagicMock(**attrs)


def test_online_cascade_is_persisted_when_hybrid_is_disabled(session):
    from models import Anomaly, MLSettings

    ml = session.get(MLSettings, 1)
    ml.ml_use_hybrid = False
    ml.ml_online_primary = "isolation_forest"
    ml.ml_online_secondary = "lstm_ae"
    ml.ml_online_combo = "cascade"
    session.add(ml)
    session.commit()

    detector, _ = _detector(use_hybrid=False)
    pipeline = _published()
    pipeline.detect_detailed.return_value = _outcome([{
        "score_norm": 0.9,
        "detector_algorithm": "cascade:if>lstm",
        "detector_combo": "cascade",
        "detector_primary": "isolation_forest",
        "detector_secondary": "lstm_ae",
    }])
    detector._get_pipeline_for_network = MagicMock(return_value=pipeline)
    with patch("ai.anomaly_ml.classify_anomaly_type", return_value=[_anomaly()]):
        detector._detect_for_network(session, network_id=7)

    persisted = session.exec(select(Anomaly)).one()
    assert persisted.detector_combo == "cascade"
    assert persisted.detector_algorithm == "cascade:if>lstm"
    assert persisted.detector_primary == "isolation_forest"
    assert persisted.detector_secondary == "lstm_ae"


def _prep_online_attrs(detector):
    detector._online_secondary = None
    detector._online_combo = "single"
    detector._online_fusion_alpha = 0.5


def test_online_rebuild_swaps_when_new_pipeline_ready(monkeypatch):
    """No retreino, o candidato TREINADO substitui o anterior."""
    import ai.online.pipeline as pmod
    detector, _ = _detector(use_hybrid=False)
    _prep_online_attrs(detector)
    detector._train_inline = True
    detector.engine = create_engine("sqlite://")
    detector._training_context = MagicMock(return_value=object())
    detector._online_pipelines = {7: MagicMock()}        # pipeline "velho"
    new = MagicMock(combo="single", fusion_alpha=0.5)
    new.is_ready.return_value = True
    new.train.return_value = True
    monkeypatch.setattr(pmod, "OnlinePipeline", lambda **kw: new)
    detector._maybe_retrain_online(7, 60, ml_settings=object())
    new.train.assert_called_once()                        # treinado antes de publicar
    assert detector._online_pipelines[7] is new          # substituído pelo novo


def test_online_rebuild_keeps_old_when_candidate_does_not_train(monkeypatch):
    """Candidato instanciado mas não treinado nunca é publicado."""
    import ai.online.pipeline as pmod
    detector, _ = _detector(use_hybrid=False)
    _prep_online_attrs(detector)
    detector._train_inline = True
    detector.engine = create_engine("sqlite://")
    detector._training_context = MagicMock(return_value=object())
    old = MagicMock()
    detector._online_pipelines = {7: old}
    new = MagicMock()
    new.is_ready.return_value = True
    new.train.return_value = False
    monkeypatch.setattr(pmod, "OnlinePipeline", lambda **kw: new)
    detector._maybe_retrain_online(7, 60, ml_settings=object())
    assert detector._online_pipelines[7] is old
    assert detector._training(7)["status"] == "failed"


def test_online_rebuild_keeps_old_when_new_not_ready(monkeypatch):
    """Rollback: se o novo pipeline não subir, mantém o anterior."""
    import ai.online.pipeline as pmod
    detector, _ = _detector(use_hybrid=False)
    _prep_online_attrs(detector)
    old = MagicMock()
    detector._online_pipelines = {7: old}
    new = MagicMock(); new.is_ready.return_value = False
    monkeypatch.setattr(pmod, "OnlinePipeline", lambda **kw: new)
    detector._maybe_retrain_online(7, 60, ml_settings=object())
    assert detector._online_pipelines[7] is old          # rollback: anterior preservado


def test_online_rebuild_skipped_off_cadence(monkeypatch):
    """Fora da cadência (%60 != 0) não reconstrói."""
    import ai.online.pipeline as pmod
    detector, _ = _detector(use_hybrid=False)
    _prep_online_attrs(detector)
    old = MagicMock()
    detector._online_pipelines = {7: old}
    monkeypatch.setattr(pmod, "OnlinePipeline", lambda **kw: (_ for _ in ()).throw(AssertionError("não deveria construir")))
    detector._maybe_retrain_online(7, 59, ml_settings=object())
    assert detector._online_pipelines[7] is old


def test_pipeline_build_is_atomic_when_secondary_fails(monkeypatch):
    """Secundário falhando não produz meio-pipeline."""
    from ai.online.pipeline import OnlinePipeline
    from unittest.mock import MagicMock
    import ai.online.registry as reg

    p = OnlinePipeline("isolation_forest", "lstm_ae", "cascade", fusion_alpha=0.5)
    ok_runner = MagicMock()
    p_spec = MagicMock(); p_spec.factory = lambda ml: ok_runner
    s_spec = MagicMock()
    def boom(_ml): raise RuntimeError("falha no secundário")
    s_spec.factory = boom
    monkeypatch.setitem(reg._registry, "isolation_forest", p_spec)
    monkeypatch.setitem(reg._registry, "lstm_ae", s_spec)

    p.build_runners(ml_settings=object())
    assert p.is_ready() is False              # não publica meio-pipeline
    assert p._secondary_runner is None
    assert p._primary_runner is None          # nada publicado quando o build falha


def test_pipeline_build_is_atomic_when_primary_fails(monkeypatch):
    """Primário falhando deixa o pipeline não-pronto."""
    from ai.online.pipeline import OnlinePipeline
    from unittest.mock import MagicMock
    import ai.online.registry as reg

    p = OnlinePipeline("isolation_forest", None, "single", fusion_alpha=0.5)
    bad = MagicMock()
    def boom(_ml): raise RuntimeError("falha no primário")
    bad.factory = boom
    monkeypatch.setitem(reg._registry, "isolation_forest", bad)
    p.build_runners(ml_settings=object())
    assert p.is_ready() is False


def test_live_always_runs_online_even_when_use_hybrid_true(session):
    """Com ml_use_hybrid=True, a operação ao vivo ainda usa o OnlinePipeline."""
    from unittest.mock import MagicMock
    from models import MLSettings

    ml = session.get(MLSettings, 1)
    ml.ml_use_hybrid = True
    ml.ml_online_combo = "shadow"
    session.add(ml); session.commit()

    detector, _ = _detector(use_hybrid=True)
    detector._run_online_pipeline = MagicMock()
    detector._detect_for_network(session, network_id=7)
    detector._run_online_pipeline.assert_called()


def test_pipeline_build_publishes_when_all_runners_ok(monkeypatch):
    """Build bem-sucedido fica pronto e captura settings (contamination/window)."""
    from ai.online.pipeline import OnlinePipeline
    from unittest.mock import MagicMock
    import ai.online.registry as reg

    p = OnlinePipeline("isolation_forest", "lstm_ae", "shadow", fusion_alpha=0.5)
    pr, sr = MagicMock(), MagicMock()
    ps = MagicMock(); ps.factory = lambda ml: pr
    ss = MagicMock(); ss.factory = lambda ml: sr
    monkeypatch.setitem(reg._registry, "isolation_forest", ps)
    monkeypatch.setitem(reg._registry, "lstm_ae", ss)

    ml = type("S", (), {"ml_if_contamination": 0.07, "ml_lstm_window": 9})()
    p.build_runners(ml_settings=ml)
    assert p.is_ready() is True
    assert p._primary_runner is pr and p._secondary_runner is sr
    assert p._contamination == 0.07 and p._window == 9
