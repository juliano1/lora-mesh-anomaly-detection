"""Camada fast_rule descontinuada e não reativável."""
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine


def test_fast_rule_code_removed():
    import ai.anomaly_ml as anomaly_ml
    # O detector legado (e a regra rápida dentro dele) saiu; só o pipeline online decide.
    assert not hasattr(anomaly_ml, "MLAnomalyDetector")


def _detector(combo):
    from ai.anomaly_detector import AnomalyDetector
    d = object.__new__(AnomalyDetector)
    d._models_lock = threading.Lock()
    d._network_cycle_counts = {7: 1}
    d._online_pipelines = {}
    d._online_primary, d._online_secondary = "isolation_forest", "lstm_ae"
    d._online_combo, d._online_fusion_alpha = combo, 0.5
    d.detect_simulated = True
    d.fast_detection_enabled = True          # flag legada: ignorada
    d.stats = {"total_detected": 0, "by_type": {}, "last_detection": None}
    return d


@pytest.fixture
def session():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    with Session(eng) as db:
        from models import MLSettings, Node
        db.add(Node(node_id=1, network_id=7, name="N1", status="online"))
        db.add(MLSettings(id=1, ml_fast_detection=True))
        db.commit()
        yield db


@pytest.mark.parametrize("combo", ["single", "shadow", "cascade"])
@pytest.mark.parametrize("trained", [True, False])
def test_loop_never_calls_fast_rule(session, combo, trained):
    d = _detector(combo)
    assert d.fast_detection_enabled is False
    d._feature_extractor = MagicMock()
    d._run_online_pipeline = MagicMock()
    d._maybe_retrain_online = MagicMock()
    d._save_and_notify = MagicMock()
    d._detect_for_network(session, network_id=7)
    # Sem detector legado no laço: nada de regra rápida nem de IF paralelo.
    assert not hasattr(d, "_get_detector_for_network")
    d._save_and_notify.assert_not_called()
    d._run_online_pipeline.assert_called_once()


def test_setter_keeps_disabled():
    from ai.anomaly_detector import AnomalyDetector
    d = object.__new__(AnomalyDetector)
    d.set_fast_detection(True)
    assert d.fast_detection_enabled is False


def test_load_ml_settings_forces_false(session):
    from mqtt_service_sync import _load_ml_settings
    assert _load_ml_settings(session).ml_fast_detection is False


def _client(with_settings=True):
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa: F401
    SQLModel.metadata.create_all(eng)
    import main
    from database import get_session
    from auth import get_current_user, get_current_admin
    import jwt
    from config import settings

    def _get_session():
        with Session(eng) as s:
            yield s

    with Session(eng) as s:
        from models import MLSettings, User
        s.add(User(id=1, username="admin", password_hash="x", is_admin=True))
        if with_settings:
            s.add(MLSettings(id=1))
        s.commit()
    main.app.dependency_overrides[get_session] = _get_session
    main.app.dependency_overrides[get_current_user] = lambda: "admin"
    main.app.dependency_overrides[get_current_admin] = lambda: type("U", (), {"id": 1, "is_admin": True})()
    token = jwt.encode({"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                       settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    tc = TestClient(main.app)
    tc.headers.update({"Authorization": f"Bearer {token}"})
    return tc, eng, main


@pytest.mark.parametrize("with_settings", [True, False])
def test_reactivation_rejected_without_persisting(with_settings):
    client, eng, main = _client(with_settings)
    try:
        r1 = client.patch("/api/settings/fast-detection", params={"enabled": True})
        r2 = client.patch("/api/settings/ml", json={"fast_detection": True})
        assert r1.status_code == 409 and r2.status_code == 409
        with Session(eng) as s:
            from models import MLSettings
            ml = s.get(MLSettings, 1)
            assert (ml is None) if not with_settings else (ml.ml_fast_detection is False)
        ok = client.patch("/api/settings/fast-detection", params={"enabled": False})
        assert ok.status_code == 200 and ok.json() == {"fast_detection_enabled": False, "deprecated": True}
        assert client.get("/api/settings/fast-detection").json() == {"fast_detection_enabled": False, "deprecated": True}
        body = client.get("/api/settings/ml").json()
        assert body["fast_detection"] is False and body["fast_detection_deprecated"] is True
    finally:
        main.app.dependency_overrides.clear()
