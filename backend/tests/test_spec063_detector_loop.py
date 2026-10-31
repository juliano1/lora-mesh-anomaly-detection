"""Loop de detecção contínua."""
import threading
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine


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


def _detector():
    from ai.anomaly_detector import AnomalyDetector
    d = object.__new__(AnomalyDetector)
    d._models_lock = threading.Lock()
    d._network_cycle_counts = {7: 1}
    d._online_pipelines = {}
    d._online_primary, d._online_secondary = "isolation_forest", "lstm_ae"
    d._online_combo, d._online_fusion_alpha = "shadow", 0.5
    d.detect_simulated = True
    d.fast_detection_enabled = False
    d.stats = {"total_detected": 0, "by_type": {}, "last_detection": None}
    d._feature_extractor = MagicMock()
    return d


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


def test_pipeline_runs_even_without_point_features(session):
    from models import MLSettings
    d = _detector()
    fe = MagicMock()
    fe.extract_features.return_value = None
    ml_det = MagicMock(); ml_det.feature_extractor = fe
    pipeline = _published(); pipeline.detect_detailed.return_value = _outcome()
    d._get_pipeline_for_network = MagicMock(return_value=pipeline)
    t = datetime(2026, 9, 1, 12, 0, 0)
    d._run_online_pipeline(session, 7, fe, [1], session.get(MLSettings, 1),
                           reference_time=t, reference_policy="P")
    ctx = pipeline.detect_detailed.call_args.args[0]
    assert ctx.features_vec is None and ctx.features_reason == "no_data"
    assert ctx.reference_time == t and ctx.reference_policy == "P"
    assert fe.extract_features.call_args.kwargs["reference_time"] == t
    assert fe.extract_features.call_args.kwargs["window_minutes"] == 15


def test_missing_feature_key_is_not_zero_filled(session):
    from models import MLSettings
    from ai.sequence_builder import FEATURE_NAMES
    d = _detector()
    fe = MagicMock()
    fe.extract_features.return_value = {k: 1.0 for k in FEATURE_NAMES[:-1]}
    ml_det = MagicMock(); ml_det.feature_extractor = fe
    pipeline = _published(); pipeline.detect_detailed.return_value = _outcome()
    d._get_pipeline_for_network = MagicMock(return_value=pipeline)
    d._run_online_pipeline(session, 7, fe, [1], session.get(MLSettings, 1))
    ctx = pipeline.detect_detailed.call_args.args[0]
    assert ctx.features_vec is None
    assert ctx.features_reason == "invalid_feature:missing:link_asymmetry"


def test_config_change_rebuilds_pipeline(session, monkeypatch):
    from models import MLSettings
    from ai.online.effective_config import resolve_effective_config
    d = _detector()
    ml = session.get(MLSettings, 1)
    old = MagicMock()
    old.model_signature = resolve_effective_config(ml).models_signature("isolation_forest", "lstm_ae")
    d._online_pipelines[7] = old
    ml.ml_lstm_window = 6
    rebuilt = MagicMock()
    # O candidato é treinado fora do ciclo; aqui o dublê publica na hora
    d._rebuild_online_pipeline = MagicMock(
        side_effect=lambda nid, ms, reason=None: d._online_pipelines.__setitem__(nid, rebuilt) or True)
    assert d._get_pipeline_for_network(7, ml) is rebuilt
    d._rebuild_online_pipeline.assert_called_once()


def test_same_config_keeps_pipeline(session):
    from models import MLSettings
    from ai.online.effective_config import resolve_effective_config
    d = _detector()
    ml = session.get(MLSettings, 1)
    old = MagicMock(combo="shadow", fusion_alpha=0.5)
    old.model_signature = resolve_effective_config(ml).models_signature("isolation_forest", "lstm_ae")
    d._online_pipelines[7] = old
    d._rebuild_online_pipeline = MagicMock()
    assert d._get_pipeline_for_network(7, ml) is old
    d._rebuild_online_pipeline.assert_not_called()


def test_detect_for_network_passes_single_cutoff_and_policy(session):
    d = _detector()
    d._run_online_pipeline = MagicMock()
    d._maybe_retrain_online = MagicMock()
    d._detect_for_network(session, network_id=7)
    kw = d._run_online_pipeline.call_args.kwargs
    from ai.reference_policy import ReferencePolicy
    assert isinstance(kw["reference_time"], datetime)
    assert isinstance(kw["reference_policy"], ReferencePolicy)


def test_apply_settings_rebuilds_when_only_model_params_change():
    """Auditoria IF: contamination/árvores (ou W) mudam → transição observável já no apply."""
    from models import MLSettings
    from ai.online.pipeline import OnlinePipeline
    d = _detector()
    d._online_secondary, d._online_combo = None, "single"
    initial = MLSettings(id=1, ml_online_primary="isolation_forest",
                         ml_if_contamination=0.03, ml_if_n_estimators=50)
    pipe = OnlinePipeline("isolation_forest", None, "single", 0.5)
    pipe.build_runners(initial)
    d._online_pipelines = {1: pipe}
    d._rebuild_online_pipeline = MagicMock(return_value=True)
    assert d.apply_settings(MLSettings(id=1, ml_online_primary="isolation_forest",
                                       ml_if_contamination=0.1, ml_if_n_estimators=100)) is True
    d._rebuild_online_pipeline.assert_called_once()
    d._rebuild_online_pipeline.reset_mock()
    d.apply_settings(initial)                     # mesma config do pipeline ativo
    d._rebuild_online_pipeline.assert_not_called()
