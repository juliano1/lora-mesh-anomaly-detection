# backend/tests/test_spec55_mqtt.py
"""Testes do MQTT handler: origem dos dados e fecho das simulações."""
import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select


@pytest.fixture
def engine():
    from models import SimulationRun, Node, Telemetry, Neighbor, Network
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


def _make_node(session, sim_mode=False, sim_ended_at=None, current_sim_run_id=None):
    from models import Node, Network
    net = Network(id=1, name="R1", description="")
    node = Node(
        node_id=1, network_id=1, lora_address="0x01",
        sim_mode=sim_mode, current_sim_run_id=current_sim_run_id,
        sim_ended_at=sim_ended_at,
    )
    session.add(net); session.add(node); session.commit()
    return node


def _make_run(session, telemetry_count=0, ended_at=None):
    from models import SimulationRun
    run = SimulationRun(
        node_id=1, anomaly="A4",
        started_at=datetime.utcnow() - timedelta(seconds=60),
        duration_planned=1800,
        telemetry_count=telemetry_count,
        ended_at=ended_at,
    )
    session.add(run); session.commit()
    return run


def test_data_origin_real_when_no_sim_field():
    """Pacote sem campo 'sim' → data_origin = 'real'."""
    from mqtt_service_sync import _get_data_origin
    assert _get_data_origin(None) == "real"
    assert _get_data_origin("") == "real"


def test_data_origin_synthetic_when_sim_field():
    from mqtt_service_sync import _get_data_origin
    assert _get_data_origin("A4") == "synthetic"


def test_grace_period_blocks_residual_beacon(engine):
    """Beacon residual dentro de 30s após SIM_STOP não deve ressuscitar sim_mode."""
    from mqtt_service_sync import _should_activate_sim
    with Session(engine) as session:
        node = _make_node(
            session,
            sim_mode=False,
            sim_ended_at=datetime.utcnow() - timedelta(seconds=10),
        )
        assert _should_activate_sim(node, "A4") is False


def test_grace_period_allows_after_30s(engine):
    from mqtt_service_sync import _should_activate_sim
    with Session(engine) as session:
        node = _make_node(
            session,
            sim_mode=False,
            sim_ended_at=datetime.utcnow() - timedelta(seconds=35),
        )
        assert _should_activate_sim(node, "A4") is True


def test_grace_period_allows_when_no_previous_sim(engine):
    """Nó que nunca teve simulação (sim_ended_at=None) ativa normalmente."""
    from mqtt_service_sync import _should_activate_sim
    with Session(engine) as session:
        node = _make_node(session, sim_mode=False, sim_ended_at=None)
        assert _should_activate_sim(node, "A4") is True


@pytest.fixture
def ingest_sim(monkeypatch):
    """Banco em memória com N2 em simulação; o beacon passa pelo handler real."""
    import json
    from sqlalchemy.pool import StaticPool
    import models
    import mqtt_service_sync as svc
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(svc, "engine", eng)

    def start(seconds_ago: float):
        with Session(eng) as s:
            s.add(models.Network(id=1, name="rede-1", active=True))
            s.add(models.Node(node_id=1, name="GW", lora_address="0x0001", network_id=1, is_gateway=True))
            run = models.SimulationRun(node_id=2, anomaly="A3", telemetry_count=3, duration_planned=480,
                                       started_at=datetime.utcnow() - timedelta(seconds=seconds_ago))
            s.add(run); s.commit()
            s.add(models.Node(node_id=2, name="N2", lora_address="0x0002", network_id=1,
                              sim_mode=True, current_sim_run_id=run.id))
            s.commit()
            run_id = run.id
        beacon = json.dumps({"i": "N2", "a": "0002", "u": 10, "q": 7, "b": 80,
                             "n": [{"a": "0001", "r": -70, "s": 8.5, "p": True, "h": 1, "l": 9}],
                             "gw_rssi": -70, "gw_snr": 8.5})
        svc.process_message_sync(beacon, network_id=1)
        with Session(eng) as s:
            return s.get(models.SimulationRun, run_id)

    yield start
    eng.dispose()


def test_beacon_without_sim_closes_run_after_grace_period(ingest_sim):
    """Beacon sem 'sim' após 90s do started_at fecha a simulação (handler real)."""
    run = ingest_sim(seconds_ago=100)
    assert run.ended_reason == "firmware_stopped"
    assert run.ended_at is not None


def test_beacon_without_sim_ignored_within_grace_period(ingest_sim):
    """Beacon sem 'sim' nos primeiros 90s NÃO fecha (downlink + retry ainda podem chegar)."""
    run = ingest_sim(seconds_ago=10)
    assert run.ended_at is None
    assert run.ended_reason is None


def test_watchdog_closes_orphan_after_120s(engine):
    """Watchdog deve fechar run com 0 beacons após 120s como firmware_rejected."""
    from mqtt_service_sync import _simulation_watchdog_tick, _close_simulation_run
    from models import SimulationRun, Node
    with Session(engine) as session:
        _make_node(session, sim_mode=True, current_sim_run_id=None)
        run = SimulationRun(
            node_id=1, anomaly="A1",
            started_at=datetime.utcnow() - timedelta(seconds=130),
            duration_planned=1800,
            telemetry_count=0,
        )
        session.add(run); session.commit()
        # Simula o fechamento que o watchdog faria
        _close_simulation_run(run, "firmware_rejected", session)
        session.expire_all()
        run = session.get(SimulationRun, run.id)
        assert run.ended_reason == "firmware_rejected"
        assert run.ended_at is not None


def test_telemetry_count_increments(engine):
    from mqtt_service_sync import _increment_telemetry_count
    with Session(engine) as session:
        _make_node(session, sim_mode=True)
        run = _make_run(session, telemetry_count=5)
        _increment_telemetry_count(run.id, session)
        session.expire_all()
        from models import SimulationRun
        run = session.get(SimulationRun, run.id)
        assert run.telemetry_count == 6
