# backend/tests/test_spec55_detector.py
"""Testes de sim_run_id e detected_at no anomaly_detector."""
import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select


@pytest.fixture
def engine():
    from models import SimulationRun, Node, Telemetry, Neighbor, Anomaly, Network
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


def _seed(session):
    from models import Node, Network, SimulationRun
    net = Network(id=1, name="R1", description="")
    node = Node(node_id=1, network_id=1, lora_address="0x01", sim_mode=True)
    session.add(net); session.add(node); session.commit()
    run = SimulationRun(
        node_id=1, anomaly="A4",
        started_at=datetime.utcnow() - timedelta(seconds=60),
        duration_planned=1800,
    )
    session.add(run); session.commit()
    return run


def test_find_active_sim_run():
    """_find_active_sim_run retorna run ativo para o nó no momento da detecção."""
    from ai.anomaly_detector import _find_active_sim_run
    eng = create_engine("sqlite:///:memory:")
    from models import SimulationRun, Node, Network
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        run = _seed(session)
        found = _find_active_sim_run(1, datetime.utcnow(), session)
        assert found is not None
        assert found.id == run.id


def test_find_active_sim_run_returns_none_when_no_run():
    from ai.anomaly_detector import _find_active_sim_run
    eng = create_engine("sqlite:///:memory:")
    from models import Node, Network
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        net = Network(id=1, name="R1", description="")
        node = Node(node_id=1, network_id=1, lora_address="0x01")
        session.add(net); session.add(node); session.commit()
        assert _find_active_sim_run(1, datetime.utcnow(), session) is None


def test_find_active_sim_run_within_tolerance():
    """Detecção até 15 min após encerramento ainda vincula ao run (atraso ML)."""
    from ai.anomaly_detector import _find_active_sim_run
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        run = _seed(session)
        run.ended_at = datetime.utcnow() - timedelta(minutes=5)
        session.add(run); session.commit()
        # Detecção 5 min após encerramento → dentro da tolerância de 15 min
        assert _find_active_sim_run(1, datetime.utcnow(), session) is not None


def test_find_active_sim_run_after_tolerance_expired():
    """Detecção mais de 15 min após encerramento não vincula ao run."""
    from ai.anomaly_detector import _find_active_sim_run
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        run = _seed(session)
        run.ended_at = datetime.utcnow() - timedelta(minutes=20)
        session.add(run); session.commit()
        assert _find_active_sim_run(1, datetime.utcnow(), session) is None


def test_is_window_contaminated_with_active_run():
    """Janela sobreposta com SimulationRun ativo é considerada contaminada."""
    from ai.feature_engineering import _is_window_contaminated
    eng = create_engine("sqlite:///:memory:")
    from models import SimulationRun, Node, Network
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        run = _seed(session)
        now = datetime.utcnow()
        # Janela dentro do período do run
        result = _is_window_contaminated(1, now - timedelta(minutes=2), now, session)
        assert result is True


def test_is_window_contaminated_no_run():
    from ai.feature_engineering import _is_window_contaminated
    eng = create_engine("sqlite:///:memory:")
    from models import Node, Network
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        net = Network(id=1, name="R1", description="")
        node = Node(node_id=1, network_id=1, lora_address="0x01")
        session.add(net); session.add(node); session.commit()
        now = datetime.utcnow()
        result = _is_window_contaminated(1, now - timedelta(minutes=2), now, session)
        assert result is False


def test_is_window_contaminated_after_run_ended():
    """Janela após o encerramento do run NÃO é contaminada."""
    from ai.feature_engineering import _is_window_contaminated
    eng = create_engine("sqlite:///:memory:")
    from models import SimulationRun, Node, Network
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        run = _seed(session)
        run.ended_at = datetime.utcnow() - timedelta(seconds=120)
        session.add(run); session.commit()
        now = datetime.utcnow()
        # Janela que começa 60s após fim do run
        result = _is_window_contaminated(
            1,
            now - timedelta(seconds=60),
            now,
            session
        )
        assert result is False



def test_find_active_sim_run_uses_detector_context():
    """Com contexto de 60 min (sequência do LSTM), alerta 30 min após o fim ainda vincula."""
    from ai.anomaly_detector import _find_active_sim_run
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        run = _seed(session)
        run.ended_at = datetime.utcnow() - timedelta(minutes=30)
        session.add(run); session.commit()
        assert _find_active_sim_run(1, datetime.utcnow(), session) is None
        assert _find_active_sim_run(1, datetime.utcnow(), session, tolerance_minutes=60) is not None
