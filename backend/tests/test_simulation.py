# backend/tests/test_simulation.py
"""Testes de _close_simulation_run e do watchdog das simulações."""
import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select


@pytest.fixture
def engine():
    from models import SimulationRun, Node, Network
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session_with_run(engine):
    from models import Node, Network, SimulationRun
    with Session(engine) as session:
        net = Network(id=1, name="R1", description="")
        node = Node(node_id=1, network_id=1, lora_address="0x01",
                    sim_mode=True, sim_anomaly="A4", current_sim_run_id=None)
        session.add(net); session.add(node); session.commit()
        run = SimulationRun(
            node_id=1, anomaly="A4",
            started_at=datetime.utcnow() - timedelta(seconds=60),
            duration_planned=1800,
        )
        session.add(run); session.commit()
        node.current_sim_run_id = run.id
        session.add(node); session.commit()
        yield session, run.id


def test_close_simulation_run_sets_fields(session_with_run):
    from mqtt_service_sync import _close_simulation_run
    from models import SimulationRun, Node
    session, run_id = session_with_run
    run = session.get(SimulationRun, run_id)
    _close_simulation_run(run, "operator_stop", session)
    session.expire_all()
    run = session.get(SimulationRun, run_id)
    assert run.ended_at is not None
    assert run.ended_reason == "operator_stop"
    node = session.get(Node, 1)
    assert node.sim_mode is False
    assert node.sim_anomaly is None
    assert node.current_sim_run_id is None
    assert node.sim_ended_at is not None


@pytest.mark.parametrize("published", [False, True])
def test_stop_simulation_only_closes_run_after_publication(session_with_run, monkeypatch, published):
    import main
    from fastapi import HTTPException
    from models import Node, SimulationRun
    session, run_id = session_with_run
    sent = []
    monkeypatch.setattr(main, "publish_to_downlink",
                        lambda payload, network_id: sent.append((payload, network_id)) or published)
    if published:
        assert main.stop_simulation(1, session)["status"] == "ok"
    else:
        with pytest.raises(HTTPException) as exc:
            main.stop_simulation(1, session)
        assert exc.value.status_code == 503
    session.expire_all()
    assert session.get(Node, 1).sim_mode is (not published)
    assert (session.get(SimulationRun, run_id).ended_at is not None) is published
    assert len(sent) == 1 and sent[0][1] == 1


def test_watchdog_expires_old_run(engine):
    from models import Node, Network, SimulationRun
    from mqtt_service_sync import _simulation_watchdog_tick
    with Session(engine) as session:
        net = Network(id=1, name="R1", description="")
        node = Node(node_id=1, network_id=1, lora_address="0x01",
                    sim_mode=True, sim_anomaly="A4")
        session.add(net); session.add(node); session.commit()
        run = SimulationRun(
            node_id=1, anomaly="A4",
            started_at=datetime.utcnow() - timedelta(seconds=1900),
            duration_planned=1800,
            telemetry_count=10,
        )
        session.add(run); session.commit()
        _simulation_watchdog_tick(engine)
        session.expire_all()
        run = session.get(SimulationRun, run.id)
        assert run.ended_at is not None
        assert run.ended_reason == "timeout"


def _orphan_run(engine, seconds_ago):
    """Run sem nenhum beacon (telemetry_count=0) iniciado há `seconds_ago` segundos."""
    from models import Node, Network, SimulationRun
    with Session(engine) as session:
        session.add(Network(id=1, name="R1", description=""))
        session.add(Node(node_id=1, network_id=1, lora_address="0x01", sim_mode=True))
        session.commit()
        run = SimulationRun(
            node_id=1, anomaly="A7",
            started_at=datetime.utcnow() - timedelta(seconds=seconds_ago),
            duration_planned=1800,
            telemetry_count=0,
        )
        session.add(run); session.commit()
        return run.id


def test_watchdog_retries_sim_start_between_60_and_120s(engine, monkeypatch):
    """Entre 60 e 120 s sem beacons o watchdog reenvia SIM_START e mantém o run aberto
    (o downlink LoRa pode falhar em silêncio). Chama o watchdog real."""
    import mqtt_service_sync
    from models import SimulationRun
    sent = []
    monkeypatch.setattr(mqtt_service_sync, "publish_to_downlink",
                        lambda payload, network_id: sent.append((payload, network_id)) or True)
    run_id = _orphan_run(engine, seconds_ago=90)
    mqtt_service_sync._simulation_watchdog_tick(engine)
    with Session(engine) as session:
        run = session.get(SimulationRun, run_id)
        assert run.ended_at is None and run.ended_reason is None
    assert sent == [({"cmd": "send", "dest": "0x01", "msg": "SIM_START:A7:1800", "msg_id": 9039}, 1)]


def test_watchdog_closes_orphan_as_firmware_rejected_after_120s(engine, monkeypatch):
    """Sem beacons após 120 s (retry já ocorreu no tick anterior): encerra como firmware_rejected,
    sem novo reenvio, e libera o nó."""
    import mqtt_service_sync
    from models import Node, SimulationRun
    sent = []
    monkeypatch.setattr(mqtt_service_sync, "publish_to_downlink",
                        lambda payload, network_id: sent.append((payload, network_id)) or True)
    run_id = _orphan_run(engine, seconds_ago=130)
    mqtt_service_sync._simulation_watchdog_tick(engine)
    with Session(engine) as session:
        run = session.get(SimulationRun, run_id)
        assert run.ended_reason == "firmware_rejected" and run.ended_at is not None
        assert session.get(Node, 1).sim_mode is False
    assert sent == []


def test_watchdog_does_not_close_active_run(engine):
    from models import Node, Network, SimulationRun
    from mqtt_service_sync import _simulation_watchdog_tick
    with Session(engine) as session:
        net = Network(id=1, name="R1", description="")
        node = Node(node_id=1, network_id=1, lora_address="0x01", sim_mode=True)
        session.add(net); session.add(node); session.commit()
        run = SimulationRun(
            node_id=1, anomaly="A4",
            started_at=datetime.utcnow() - timedelta(seconds=300),
            duration_planned=1800,
            telemetry_count=15,
        )
        session.add(run); session.commit()
        _simulation_watchdog_tick(engine)
        session.expire_all()
        run = session.get(SimulationRun, run.id)
        assert run.ended_at is None


def test_duration_override_clipped_below(engine):
    """Override abaixo de 900s é clipado para 900s."""
    from main import _calculate_duration
    assert _calculate_duration("A4", 100) == 900


def test_duration_override_clipped_above(engine):
    from main import _calculate_duration
    assert _calculate_duration("A4", 5000) == 3600


def test_duration_override_valid(engine):
    from main import _calculate_duration
    assert _calculate_duration("A4", 1200) == 1200


def test_a3_ignores_override(engine):
    """A3 sempre retorna 480s."""
    from main import _calculate_duration
    assert _calculate_duration("A3", 1800) == 480


def test_duration_no_override_uses_default(engine):
    from main import _calculate_duration
    assert _calculate_duration("A1", None) == 1800


def test_get_active_simulation_includes_telemetry_count(engine):
    from models import Node, Network, SimulationRun
    from sqlmodel import Session
    from datetime import datetime
    with Session(engine) as session:
        net = Network(id=1, name="R1", description="")
        node = Node(node_id=1, network_id=1, lora_address="0x01",
                    name="N1", sim_mode=True, sim_anomaly="A4",
                    sim_started_at=datetime.utcnow())
        session.add(net); session.add(node); session.commit()
        run = SimulationRun(
            node_id=1, anomaly="A4",
            started_at=datetime.utcnow(),
            duration_planned=1800,
            telemetry_count=12,
        )
        session.add(run); session.commit()
        node.current_sim_run_id = run.id
        session.add(node); session.commit()

        import main
        body = main.get_active_simulation(session=session)
        assert body["active"] is True
        assert body["telemetry_count"] == 12
        assert body["sim_run_id"] == run.id


def test_get_simulation_run_ttd(engine):
    from models import Node, Network, SimulationRun
    from sqlmodel import Session
    from datetime import datetime, timedelta
    with Session(engine) as session:
        net = Network(id=1, name="R1", description="")
        node = Node(node_id=1, network_id=1, lora_address="0x01")
        session.add(net); session.add(node); session.commit()
        started = datetime.utcnow() - timedelta(seconds=300)
        detected = started + timedelta(seconds=120)
        run = SimulationRun(
            node_id=1, anomaly="A4",
            started_at=started,
            ended_at=datetime.utcnow(),
            duration_planned=1800,
            ended_reason="timeout",
            telemetry_count=15,
            detected_at=detected,
        )
        session.add(run); session.commit()
        import main
        body = main.get_simulation_run(run.id, session=session)
        assert body["ttd_seconds"] == 120
        assert body["telemetry_count"] == 15
