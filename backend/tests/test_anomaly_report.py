"""Testes do contrato de dados dos endpoints de anomalia — spec AnomalyReportModal."""
import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel


@pytest.fixture
def in_memory_session():
    from models import Anomaly, Node, Network
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        net = Network(id=1, name="Rede Teste", description="", location="")
        node = Node(node_id=1, network_id=1, lora_address="0x01", name="Node 1")
        session.add(net)
        session.add(node)
        session.commit()
        anomaly = Anomaly(
            node_id=1,
            anomaly_type="A1",
            category="radio",
            severity="medium",
            description="test",
            detection_method="ml",
            payload_raw="",
            detector_combo="cascade",
            detector_primary="isolation_forest",
            detector_secondary="lstm_ae",
        )
        session.add(anomaly)
        session.commit()
        yield session


def test_anomaly_model_has_detector_fields(in_memory_session):
    """Anomaly model must expose detector_combo, detector_primary, detector_secondary."""
    from sqlmodel import select
    from models import Anomaly
    anomaly = in_memory_session.exec(select(Anomaly)).first()
    assert anomaly.detector_combo == "cascade"
    assert anomaly.detector_primary == "isolation_forest"
    assert anomaly.detector_secondary == "lstm_ae"


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_export_identifies_the_emitting_detector(in_memory_session, fmt):
    """Em shadow, cada modelo grava sua linha: a exportação precisa dizer qual foi."""
    import asyncio
    import csv
    import io
    import json
    import main

    response = main.export_anomalies(format=fmt, hours=24, from_date=None, to_date=None,
                                     network_id=None, session=in_memory_session)

    async def _read():
        return b"".join([chunk async for chunk in response.body_iterator]).decode()

    body = asyncio.run(_read())
    rows = json.loads(body) if fmt == "json" else list(csv.DictReader(io.StringIO(body)))
    assert len(rows) == 1
    assert rows[0]["detector_combo"] == "cascade"
    assert rows[0]["detector_primary"] == "isolation_forest"
    assert rows[0]["detector_secondary"] == "lstm_ae"
    assert "sim_run_id" in rows[0] and "classification_mode_used" in rows[0]
