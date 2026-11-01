"""AX descartado no fluxo contínuo (só a faixa determinística A1-A7
gera registro de anomalia)."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select


@pytest.fixture
def engine():
    # Importar models antes de create_all para registrar as tabelas no metadata.
    from models import Anomaly, Node, Network, SimulationRun, LLMAnalysis, MLSettings  # noqa: F401
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def seeded_session(engine):
    from models import Node, Network
    with Session(engine) as session:
        session.add(Network(id=1, name="Rede Teste", description="", location=""))
        session.add(Node(node_id=1, network_id=1, lora_address="0x01", name="Node 1"))
        session.commit()
        yield session


def _make_service(engine):
    from ai.anomaly_detector import AnomalyDetector
    return AnomalyDetector(engine=engine, ml_settings=None)


def _ax_anomaly():
    return {
        "anomaly_type": "AX",
        "category": "statistical",
        "severity": "low",
        "description": "Outlier estatístico não classificado (score=-0.300)",
        "suggested_action": "Monitorar",
        "evidence": {"features": {}, "anomaly_score": -0.3},
        "detection_method": "ml",
        "confidence": 30,
    }


def test_stats_expose_ax_discard_counters(engine):
    svc = _make_service(engine)
    stats = svc.get_stats()
    assert stats["ax_discarded_total"] == 0
    assert stats["ax_discarded_by_algo"] == {}


def test_ax_discarded_before_any_db_access(engine):
    svc = _make_service(engine)
    fake_session = MagicMock()

    svc._save_and_notify(fake_session, node_id=1, anomaly=_ax_anomaly(),
                         confidence=30, detector_algorithm="isolation_forest")

    # Guarda antes do dedup: nenhuma query, nenhum insert, nenhum commit.
    fake_session.exec.assert_not_called()
    fake_session.add.assert_not_called()
    fake_session.commit.assert_not_called()


def test_ax_does_not_increment_total_detected(engine):
    svc = _make_service(engine)
    fake_session = MagicMock()

    svc._save_and_notify(fake_session, node_id=1, anomaly=_ax_anomaly(),
                         confidence=30, detector_algorithm="isolation_forest")

    assert svc.stats["total_detected"] == 0


def test_ax_discard_counter_by_algo(engine):
    svc = _make_service(engine)
    fake_session = MagicMock()

    svc._save_and_notify(fake_session, node_id=1, anomaly=_ax_anomaly(),
                         confidence=30, detector_algorithm="isolation_forest")
    svc._save_and_notify(fake_session, node_id=2, anomaly=_ax_anomaly(),
                         confidence=30, detector_algorithm="isolation_forest")
    svc._save_and_notify(fake_session, node_id=3, anomaly=_ax_anomaly(),
                         confidence=30, detector_algorithm="lstm_ae")

    assert svc.stats["ax_discarded_total"] == 3
    assert svc.stats["ax_discarded_by_algo"] == {
        "isolation_forest": 2,
        "lstm_ae": 1,
    }


def test_ax_discard_counter_falls_back_to_dict_algo(engine):
    """Sem o param detector_algorithm, usa anomaly['detector_algorithm'];
    ausente em ambos, cai em '?'."""
    svc = _make_service(engine)
    fake_session = MagicMock()

    anomaly = _ax_anomaly()
    anomaly["detector_algorithm"] = "single:lstm"
    svc._save_and_notify(fake_session, node_id=1, anomaly=anomaly, confidence=30)

    svc._save_and_notify(fake_session, node_id=2, anomaly=_ax_anomaly(), confidence=30)

    assert svc.stats["ax_discarded_by_algo"] == {"single:lstm": 1, "?": 1}


@pytest.mark.parametrize("atype", ["A1", "A2", "A3", "A4", "A5", "A6", "A7"])
def test_deterministic_types_are_persisted(engine, seeded_session, atype):
    from ai.anomaly_ml import build_anomaly
    from models import Anomaly

    svc = _make_service(engine)
    anomaly = build_anomaly(
        atype, confidence=80, description=f"teste {atype}",
        suggested_action="acao", features={"rssi_mean": -90.0}, score=-0.4,
    )

    svc._save_and_notify(seeded_session, node_id=1, anomaly=anomaly,
                         confidence=80, detector_algorithm="isolation_forest")

    rows = seeded_session.exec(
        select(Anomaly).where(Anomaly.anomaly_type == atype)
    ).all()
    assert len(rows) == 1
    assert svc.stats["total_detected"] == 1
    assert svc.stats["ax_discarded_total"] == 0


def _make_sim_run(session, node_id=1):
    from datetime import datetime, timedelta
    from models import SimulationRun
    run = SimulationRun(
        node_id=node_id,
        anomaly="A1",
        started_at=datetime.utcnow() - timedelta(minutes=5),
        ended_at=None,
        duration_planned=600,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def test_ax_does_not_latch_detected_at(engine, seeded_session):
    from models import SimulationRun
    svc = _make_service(engine)
    run = _make_sim_run(seeded_session, node_id=1)
    assert run.detected_at is None

    svc._save_and_notify(seeded_session, node_id=1, anomaly=_ax_anomaly(),
                         confidence=30, detector_algorithm="isolation_forest")

    seeded_session.refresh(run)
    assert run.detected_at is None  # AX descartado não carimba TTD


def test_a1_latches_detected_at(engine, seeded_session):
    from ai.anomaly_ml import build_anomaly
    svc = _make_service(engine)
    run = _make_sim_run(seeded_session, node_id=1)

    a1 = build_anomaly("A1", confidence=80, description="deg sinal",
                       suggested_action="acao",
                       features={"rssi_mean": -105.0}, score=-0.5)
    svc._save_and_notify(seeded_session, node_id=1, anomaly=a1,
                         confidence=80, detector_algorithm="isolation_forest")

    seeded_session.refresh(run)
    assert run.detected_at is not None  # A1 carimba TTD normalmente


def _neutral_features():
    return {
        "rssi_mean": -80.0, "rssi_std": 1.0, "snr_mean": 8.0, "snr_std": 1.0,
        "pdr_mean": 9.0, "battery_delta": 0.0, "packet_loss_rate": 0.0,
        "rssi_snr_divergence": 0.0, "beacon_interval_std": 0.0,
        "route_change_rate": 0.0, "link_asymmetry": 0.0,
    }


def test_real_classifier_ax_output_is_discarded(engine, seeded_session):
    from ai.anomaly_ml import classify_anomaly_type
    from models import Anomaly

    # Features saudáveis não disparam A1-A7 → classificador emite AX.
    classified = classify_anomaly_type(_neutral_features(), -0.30)
    assert len(classified) == 1
    assert classified[0]["anomaly_type"] == "AX"

    svc = _make_service(engine)
    svc._save_and_notify(seeded_session, node_id=1, anomaly=classified[0],
                         confidence=30, detector_algorithm="isolation_forest")

    rows = seeded_session.exec(select(Anomaly)).all()
    assert rows == []
    assert svc.stats["ax_discarded_total"] == 1
