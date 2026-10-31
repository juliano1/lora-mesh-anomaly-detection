"""Inicialização: migrações com falha fechada, MQTT com reconexão, /health e estado do detector."""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import create_engine


def test_service_migrations_also_stop_startup(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "STARTUP_MIGRATIONS", ())
    monkeypatch.setattr(main, "SERVICE_MIGRATIONS", (("add_ml_settings_table", "inexistente"),))
    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
    with pytest.raises(RuntimeError, match="add_ml_settings_table"):
        main.run_startup_migrations(engine)
    engine.dispose()


def test_mqtt_thread_no_longer_runs_migrations():
    import inspect
    import mqtt_service_sync
    assert "migrations." not in inspect.getsource(mqtt_service_sync.start_mqtt_client)


class _FakeClient:
    def __init__(self, *a, **k):
        self.calls = []

    def __getattr__(self, name):
        def record(*a, **k):
            self.calls.append((name, a, k))
        return record


def test_broker_down_at_startup_keeps_retrying(monkeypatch):
    import mqtt_service_sync as svc
    monkeypatch.setattr(svc, "service_state", {"mqtt_connected": False, "detector": "not_started"})
    monkeypatch.setattr(svc, "mqtt_client", None)
    clients = []
    monkeypatch.setattr(svc.mqtt, "Client", lambda *a, **k: clients.append(_FakeClient()) or clients[-1])
    monkeypatch.setattr(svc, "_load_ml_settings", MagicMock(side_effect=RuntimeError("banco")))
    monkeypatch.setattr(svc.asyncio, "run_coroutine_threadsafe", lambda coro, loop: coro.close())
    svc.start_mqtt_client(loop=None)
    names = [c[0] for c in clients[0].calls]
    assert "connect_async" in names and "connect" not in names
    assert ("loop_forever", (), {"retry_first_connection": True}) in clients[0].calls
    assert svc.service_state["detector"] == "failed"      # a ingestão segue; /health mostra


def test_connection_callbacks_update_state(monkeypatch):
    import mqtt_service_sync as svc
    monkeypatch.setattr(svc, "service_state", {"mqtt_connected": False, "detector": "running"})
    client = MagicMock()
    svc.on_connect(client, None, None, 0)
    assert svc.service_state["mqtt_connected"] is True
    client.subscribe.assert_any_call("lab/+/uplink")       # assinaturas refeitas a cada conexão
    svc.on_disconnect(client, None, None, 7)
    assert svc.service_state["mqtt_connected"] is False


@pytest.mark.parametrize("connected, detector, status", [
    (True, "running", "ok"), (False, "running", "degraded"), (True, "failed", "degraded"),
])
def test_health_reports_ingestion_and_detection(monkeypatch, connected, detector, status):
    import main
    import mqtt_service_sync as svc
    monkeypatch.setitem(svc.service_state, "mqtt_connected", connected)
    monkeypatch.setitem(svc.service_state, "detector", detector)
    body = TestClient(main.app).get("/health").json()
    assert body["status"] == status and body["mqtt_connected"] is connected
    assert body["detector"] == detector and set(body) == {"status", "mqtt_connected", "detector", "timestamp"}


def test_stats_describe_the_online_pipeline_not_a_legacy_model():
    from ai.anomaly_detector import AnomalyDetector
    d = object.__new__(AnomalyDetector)
    d._models_lock = threading.Lock()
    d._online_pipelines = {7: SimpleNamespace(status=lambda: {"primary": {"ready": True}},
                                              bundle_id="b")}
    d._online_primary, d._online_secondary, d._online_combo = "lstm_ae", None, "single"
    d._online_fusion_alpha = 0.5
    d.ml_running, d.ml_interval = False, 60
    d.stats = {"total_detected": 0}
    from ai.feature_engineering import FeatureExtractor
    d._feature_extractor = FeatureExtractor()
    stats = d.get_stats()
    assert stats["ml_status"] == "trained" and "model" not in stats
    assert stats["online"]["selection"]["primary"] == "lstm_ae"
    assert stats["online"]["networks"][7]["published"]["bundle_id"] == "b"
    assert d.get_stats(network_id=8)["online"]["networks"] == {}
