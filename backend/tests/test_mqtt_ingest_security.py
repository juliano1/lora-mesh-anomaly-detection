"""Entrada MQTT: limites, valores não finitos, tópico e vínculo nó↔rede (MQTT-01)."""
import json
from types import SimpleNamespace

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select


@pytest.fixture
def ingest(monkeypatch):
    import models
    import mqtt_service_sync as svc
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(models.Network(id=1, name="rede-1", active=True))
        s.add(models.Network(id=2, name="rede-2", active=True))
        s.add(models.Node(node_id=1, name="GW", lora_address="0x0001", network_id=1, is_gateway=True))
        s.add(models.Node(node_id=2, name="N2", lora_address="0x0002", network_id=1))
        s.add(models.Node(node_id=5, name="N5", lora_address="0x0005", network_id=2))
        s.commit()
    monkeypatch.setattr(svc, "engine", engine)
    yield svc, engine, models
    engine.dispose()


def _beacon(**over):
    data = {"i": "N2", "a": "0002", "u": 10, "q": 7, "b": 80,
            "n": [{"a": "0001", "r": -70, "s": 8.5, "p": True, "h": 1, "l": 9}],
            "gw_rssi": -70, "gw_snr": 8.5}
    data.update(over)
    return json.dumps(data)


def _telemetry(engine, models, node_id=None):
    with Session(engine) as s:
        query = select(models.Telemetry)
        if node_id is not None:
            query = query.where(models.Telemetry.node_id == node_id)
        return s.exec(query).all()


def test_valid_beacon_is_stored(ingest):
    svc, engine, models = ingest
    svc.process_message_sync(_beacon(), network_id=1)
    rows = _telemetry(engine, models, 2)
    assert len(rows) == 1 and rows[0].rssi == -70 and rows[0].battery_level == 80


@pytest.mark.parametrize("payload", [
    "[]",                                                   # não é objeto
    '{"i": "N2", "a": "0002", "gw_snr": NaN}',              # constante não finita
    '{"i": "N2", "a": "0002", "n": [{"a": "0001", "r": -70, "s": Infinity}]}',
    "{",                                                     # JSON inválido
])
def test_malformed_or_non_finite_json_is_rejected(ingest, payload):
    svc, engine, models = ingest
    svc.process_message_sync(payload, network_id=1)
    assert _telemetry(engine, models) == []


@pytest.mark.parametrize("over", [
    {"n": [{"a": "0001", "r": -70, "s": 1.0}] * 17},       # vizinhos acima do limite
    {"b": 250},                                              # bateria fora de 0–100/-1/-2
    {"gw_snr": 900.0},
    {"n": [{"a": "0001", "r": -70, "s": 1.0, "l": 42}]},    # PDR fora de 0–10
    {"a": "0002; DROP"},                                     # endereço não hexadecimal
    {"sim": "A9"},
])
def test_out_of_range_fields_are_rejected(ingest, over):
    svc, engine, models = ingest
    svc.process_message_sync(_beacon(**over), network_id=1)
    assert _telemetry(engine, models) == []


def test_node_of_another_network_cannot_be_written(ingest):
    svc, engine, models = ingest
    # N5 pertence à rede 2: o pacote na rede 1 não pode gravar nele, com ou sem endereço.
    svc.process_message_sync(_beacon(i="N5", a="0005"), network_id=1)
    svc.process_message_sync(json.dumps({"i": "N5", "gw_rssi": -60}), network_id=1)
    assert _telemetry(engine, models, 5) == []
    with Session(engine) as s:
        assert s.get(models.Node, 5).network_id == 2


def test_non_finite_inner_payload_values_are_dropped(ingest):
    svc, engine, models = ingest
    # O literal 1e400 vira inf em json.loads sem passar por parse_constant.
    payload = _beacon(payload={"temp": "quente", "hum": "HUGE"}).replace('"HUGE"', "1e400")
    svc.process_message_sync(payload, network_id=1)
    row = _telemetry(engine, models, 2)[0]
    assert row.temperature is None and row.humidity is None


def test_overflowing_top_level_number_is_rejected(ingest):
    svc, engine, models = ingest
    svc.process_message_sync(_beacon(gw_snr="HUGE").replace('"HUGE"', "1e400"), network_id=1)
    assert _telemetry(engine, models) == []


@pytest.mark.parametrize("topic,expected", [
    ("lab/3/uplink", 3),
    ("lab/1/send_result", 1),
    ("lab/1/x/uplink", None),
    ("lab/uplink", None),
    ("lab/-1/uplink", None),
    ("other/1/uplink", None),
])
def test_topic_must_match_contract(topic, expected):
    from mqtt_service_sync import extract_network_id
    assert extract_network_id(topic) == expected


def test_oversized_mqtt_message_is_dropped(ingest, monkeypatch):
    svc, engine, models = ingest
    called = []
    monkeypatch.setattr(svc, "process_message_sync", lambda *a, **k: called.append(a))
    big = SimpleNamespace(topic="lab/1/uplink", payload=b"{" + b" " * svc.MAX_MQTT_PAYLOAD_BYTES + b"}")
    svc.on_message(None, None, big)
    ok = SimpleNamespace(topic="lab/1/uplink", payload=_beacon().encode())
    svc.on_message(None, None, ok)
    assert len(called) == 1


@pytest.mark.parametrize("payload", [
    {"type": "send_result", "msg_id": "1", "status": "sent"},
    {"type": "send_result", "msg_id": 1, "status": "pwned"},
    {"type": "send_result", "msg_id": 1, "status": "sent", "rtt_ms": -5},
])
def test_send_result_with_unexpected_fields_is_ignored(ingest, payload):
    svc, engine, models = ingest
    with Session(engine) as s:
        s.add(models.MessageLog(id=1, message_type="unicast", command="X", status="pending"))
        s.commit()
    svc.process_send_result(json.dumps(payload), network_id=1)
    with Session(engine) as s:
        assert s.get(models.MessageLog, 1).status == "pending"


def test_downlink_goes_to_the_node_network(monkeypatch):
    """Comando para nó da rede 2 sai em lora/mesh/2/command; sem rede, não sai."""
    import mqtt_service_sync as svc
    sent = []
    client = SimpleNamespace(is_connected=lambda: True,
                             publish=lambda topic, body: sent.append(topic) or SimpleNamespace(rc=0))
    monkeypatch.setattr(svc, "mqtt_client", client)
    assert svc.publish_to_downlink({"cmd": "send"}, 2) is True
    assert svc.publish_to_downlink({"cmd": "send"}, None) is False
    assert sent == ["lora/mesh/2/command"]


def test_watchdog_retry_uses_node_network(ingest, monkeypatch):
    """O reenvio do SIM_START pelo watchdog usa a rede do nó, não a rede 1."""
    from datetime import datetime, timedelta
    svc, engine, models = ingest
    topics = []
    monkeypatch.setattr(svc, "publish_to_downlink",
                        lambda payload, network_id: topics.append(network_id) or True)
    with Session(engine) as s:
        s.add(models.SimulationRun(node_id=5, anomaly="A3", duration_planned=300, telemetry_count=0,
                                   started_at=datetime.utcnow() - timedelta(seconds=80)))
        s.commit()
    svc._simulation_watchdog_tick(engine)
    assert topics == [2]
