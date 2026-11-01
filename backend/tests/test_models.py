# backend/tests/test_models.py
"""Testes dos campos de simulação e de origem dos dados nos modelos."""
import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel


@pytest.fixture
def engine():
    from models import SimulationRun, Node, Telemetry, Neighbor, Anomaly, Network
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


def test_simulation_run_model_fields():
    from models import SimulationRun
    run = SimulationRun(
        node_id=1,
        anomaly="A4",
        started_at=__import__("datetime").datetime.utcnow(),
        duration_planned=1800,
    )
    assert run.ended_at is None
    assert run.telemetry_count == 0
    assert run.ended_reason is None
    assert run.detected_at is None
    assert run.detector_primary is None
    assert run.detector_secondary is None
    assert run.detector_combo is None


def test_simulation_run_tablename():
    from models import SimulationRun
    assert SimulationRun.__tablename__ == "simulation_run"


def test_telemetry_has_data_origin():
    from models import Telemetry
    t = Telemetry(node_id=1, rssi=-80, snr=5.0, payload_raw="")
    assert t.data_origin == "real"


def test_neighbor_has_data_origin():
    from models import Neighbor
    n = Neighbor(node_id=1, neighbor_address="0x01", rssi=-80, snr=5.0)
    assert n.data_origin == "real"


def test_anomaly_has_sim_run_id():
    from models import Anomaly
    a = Anomaly(
        node_id=1, anomaly_type="A4", category="radio",
        severity="medium", description="test", detection_method="ml",
    )
    assert a.sim_run_id is None


def test_node_has_current_sim_run_id():
    from models import Node
    n = Node(node_id=1)
    assert n.current_sim_run_id is None


def test_simulation_run_persists(engine):
    from models import SimulationRun, Node, Network
    from datetime import datetime
    with Session(engine) as session:
        net = Network(id=1, name="R1", description="")
        node = Node(node_id=1, network_id=1, lora_address="0x01")
        session.add(net); session.add(node); session.commit()
        run = SimulationRun(
            node_id=1, anomaly="A4",
            started_at=datetime.utcnow(), duration_planned=1800,
        )
        session.add(run); session.commit()
        assert run.id is not None
        assert run.telemetry_count == 0


def test_migration_idempotent():
    """Migration aplicada 2x não levanta exceção."""
    from sqlalchemy import create_engine
    from migrations.add_data_origin_and_simulation_run import upgrade
    eng = create_engine("sqlite:///:memory:")
    from sqlalchemy import text
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS node (
                node_id INTEGER PRIMARY KEY,
                sim_mode BOOLEAN DEFAULT FALSE NOT NULL,
                sim_anomaly VARCHAR(3),
                sim_started_at TIMESTAMP,
                sim_ended_at TIMESTAMP,
                current_sim_run_id INTEGER,
                name VARCHAR,
                lora_address VARCHAR,
                status VARCHAR DEFAULT 'offline',
                last_seen TIMESTAMP,
                battery_level INTEGER,
                led_state VARCHAR DEFAULT 'off',
                network_id INTEGER,
                is_gateway BOOLEAN DEFAULT FALSE,
                device_model VARCHAR
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY,
                node_id INTEGER,
                timestamp TIMESTAMP,
                rssi INTEGER NOT NULL,
                snr REAL NOT NULL,
                packet_id INTEGER,
                parent_id INTEGER,
                uptime INTEGER,
                sequence INTEGER,
                battery_level INTEGER,
                temperature REAL,
                humidity REAL,
                payload_raw TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS neighbor (
                id INTEGER PRIMARY KEY,
                node_id INTEGER NOT NULL,
                neighbor_address VARCHAR NOT NULL,
                rssi INTEGER NOT NULL,
                snr REAL NOT NULL,
                is_parent BOOLEAN DEFAULT FALSE,
                hops INTEGER,
                distance_m REAL,
                link_quality INTEGER,
                timestamp TIMESTAMP NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS anomaly (
                id INTEGER PRIMARY KEY,
                node_id INTEGER NOT NULL,
                timestamp TIMESTAMP,
                anomaly_type VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                severity VARCHAR DEFAULT 'medium',
                description TEXT NOT NULL,
                detection_method VARCHAR DEFAULT 'ml',
                payload_raw TEXT NOT NULL
            )
        """))
    upgrade(eng)   # primeira aplicação
    upgrade(eng)   # idempotente — não deve levantar
