"""Issues 1 & 2 — proveniência (evidência fundida) e métrica (onset).

Issue 1: s_fusion/alpha/tau do combiner devem ser persistidos no evidence da
         anomalia, sem perder as features do classificador.
Issue 2: latência do detector (detected_at - onset) separada da latência total
         (detected_at - started_at).
"""
import pytest
from datetime import datetime, timedelta

from ai.anomaly_detector import _merge_detection_evidence


# ─── Issue 1: merge de evidência ─────────────────────────────────────────────

def test_merge_keeps_fusion_stats_and_classifier_features():
    classifier_ev = {"features": {"rssi_mean": -95.0}, "anomaly_score": -0.19}
    combiner_ev = {
        "s_fusion": 0.73,
        "alpha": 0.6,
        "tau": 0.5,
        "secondary": {"score": 0.4},
    }
    merged = _merge_detection_evidence(classifier_ev, combiner_ev)
    # stats da fusão persistidos
    assert merged["s_fusion"] == 0.73
    assert merged["alpha"] == 0.6
    assert merged["tau"] == 0.5
    # features do classificador preservadas
    assert merged["features"] == {"rssi_mean": -95.0}
    assert merged["anomaly_score"] == -0.19


def test_merge_classifier_features_win_on_collision():
    # combiner pode carregar 'features' do primary.evidence; o classificador vence
    classifier_ev = {"features": {"src": "classifier"}, "anomaly_score": -0.1}
    combiner_ev = {"features": {"src": "primary"}, "s_fusion": 0.9, "tau": 0.5}
    merged = _merge_detection_evidence(classifier_ev, combiner_ev)
    assert merged["features"] == {"src": "classifier"}
    assert merged["s_fusion"] == 0.9


def test_merge_handles_missing_combiner_evidence():
    classifier_ev = {"features": {"rssi_mean": -95.0}, "anomaly_score": -0.19}
    merged = _merge_detection_evidence(classifier_ev, None)
    assert merged == classifier_ev
    # cópia defensiva — não muta o original
    merged["features"]["x"] = 1
    assert "x" not in classifier_ev["features"]


def test_merge_handles_missing_classifier_evidence():
    combiner_ev = {"s_fusion": 0.7, "tau": 0.5}
    merged = _merge_detection_evidence(None, combiner_ev)
    assert merged == combiner_ev


# ─── Issue 2: onset da anomalia e latência do detector ───────────────────────

@pytest.fixture
def sim_engine():
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel
    from models import SimulationRun, Node, Network  # noqa: F401
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


def test_first_synthetic_telemetry_sets_onset(sim_engine):
    from sqlmodel import Session
    from models import SimulationRun
    from mqtt_service_sync import _increment_telemetry_count

    with Session(sim_engine) as session:
        run = SimulationRun(
            node_id=1, anomaly="A1",
            started_at=datetime.utcnow() - timedelta(seconds=84),
            duration_planned=1800,
        )
        session.add(run)
        session.commit()
        run_id = run.id

        # 1ª telemetria sintética → onset registrado
        _increment_telemetry_count(run_id, session)
        session.expire_all()
        run = session.get(SimulationRun, run_id)
        assert run.anomaly_onset_at is not None
        assert run.telemetry_count == 1
        first_onset = run.anomaly_onset_at

        # telemetrias seguintes não alteram o onset (latch)
        _increment_telemetry_count(run_id, session)
        session.expire_all()
        run = session.get(SimulationRun, run_id)
        assert run.anomaly_onset_at == first_onset
        assert run.telemetry_count == 2


def test_get_simulation_run_reports_detector_latency():
    from fastapi.testclient import TestClient
    from sqlmodel import Session, SQLModel, create_engine
    from sqlalchemy.pool import StaticPool
    import models  # noqa: F401

    eng = create_engine("sqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    import main
    from database import get_session
    from auth import get_current_user, get_current_admin

    def _get_session():
        with Session(eng) as s:
            yield s

    started = datetime(2026, 6, 24, 12, 0, 0)
    onset = started + timedelta(seconds=84)     # firmware demora a emitir
    detected = onset + timedelta(seconds=120)   # detector leva 120s a partir do onset

    with Session(eng) as s:
        from models import SimulationRun
        s.add(SimulationRun(
            id=1, node_id=1, anomaly="A1",
            started_at=started, anomaly_onset_at=onset, detected_at=detected,
            duration_planned=1800,
        ))
        s.commit()

    main.app.dependency_overrides[get_session] = _get_session
    main.app.dependency_overrides[get_current_user] = lambda: "admin"
    main.app.dependency_overrides[get_current_admin] = lambda: "admin"
    try:
        import jwt
        from config import settings
        from datetime import timezone
        token = jwt.encode(
            {"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM,
        )
        client = TestClient(main.app)
        client.headers.update({"Authorization": f"Bearer {token}"})
        r = client.get("/api/simulation/run/1")
        assert r.status_code == 200
        body = r.json()
        # latência total (do comando) inclui o atraso do firmware
        assert body["ttd_seconds"] == 84 + 120
        # latência do detector (do onset) é a métrica honesta
        assert body["detector_latency_seconds"] == 120
        assert body["anomaly_onset_at"] is not None
    finally:
        main.app.dependency_overrides.clear()
