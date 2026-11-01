"""API de configuração ML (HTTP real via TestClient)."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine


@pytest.fixture
def client_and_engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)
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
        s.add(MLSettings(id=1))
        s.commit()

    main.app.dependency_overrides[get_session] = _get_session
    main.app.dependency_overrides[get_current_user] = lambda: "admin"
    main.app.dependency_overrides[get_current_admin] = lambda: type("U", (), {"id": 1, "is_admin": True})()
    token = jwt.encode({"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                       settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    tc = TestClient(main.app)
    tc.headers.update({"Authorization": f"Bearer {token}"})
    yield tc, eng
    main.app.dependency_overrides.clear()


def test_get_exposes_lstm_block_and_effective_config(client_and_engine):
    client, _ = client_and_engine
    body = client.get("/api/settings/ml").json()
    assert body["lstm"]["window_size"] == 12
    assert body["lstm"]["history_hours"] is None
    eff = body["effective_config"]
    assert eff["lstm_history_hours"] == 24.0 and eff["lstm_history_source"] == "ML_HISTORY_HOURS"
    assert eff["if_step_minutes"] == 7 and eff["if_min_train_samples"] == 10


def test_patch_lstm_updates_effective_config(client_and_engine):
    client, _ = client_and_engine
    r = client.patch("/api/settings/ml", json={"lstm": {"window_size": 6, "history_hours": 168}})
    assert r.status_code == 200, r.text
    eff = r.json()["effective_config"]
    assert eff["lstm_window_size"] == 6 and eff["lstm_history_hours"] == 168.0
    assert eff["lstm_history_source"] == "mlsettings.ml_lstm_history_hours"
    assert client.get("/api/settings/ml").json()["lstm"]["window_size"] == 6


def test_patch_rejects_unsupported_aggregation(client_and_engine):
    client, _ = client_and_engine
    r = client.patch("/api/settings/ml", json={"lstm": {"score_aggregation": "median"}})
    assert r.status_code == 422


def test_patch_rejects_persisted_dynamic_strategy_and_does_not_write(client_and_engine):
    client, eng = client_and_engine
    from models import MLSettings
    with Session(eng) as s:
        ml = s.get(MLSettings, 1)
        ml.ml_lstm_threshold_strategy = "dynamic"
        s.add(ml); s.commit()
    r = client.patch("/api/settings/ml", json={"lstm": {"window_size": 8}})
    assert r.status_code == 422 and "não implementada" in r.json()["detail"]
    with Session(eng) as s:
        assert s.get(MLSettings, 1).ml_lstm_window == 12


def test_patch_rejects_dynamic_strategy_request_and_preserves_db(client_and_engine):
    """Pedido sem suporte é recusado (422), não aceito e ignorado."""
    client, eng = client_and_engine
    from models import MLSettings
    r = client.patch("/api/settings/ml", json={"lstm": {"threshold_strategy": "dynamic"}})
    assert r.status_code == 422
    with Session(eng) as s:
        assert s.get(MLSettings, 1).ml_lstm_threshold_strategy == "percentile"


def test_patch_rejects_unknown_lstm_field(client_and_engine):
    client, _ = client_and_engine
    r = client.patch("/api/settings/ml", json={"lstm": {"dropout": 0.5}})
    assert r.status_code == 422


def test_patch_can_repair_legacy_dynamic_strategy(client_and_engine):
    client, eng = client_and_engine
    from models import MLSettings
    with Session(eng) as s:
        ml = s.get(MLSettings, 1)
        ml.ml_lstm_threshold_strategy = "dynamic"
        s.add(ml); s.commit()
    r = client.patch("/api/settings/ml", json={"lstm": {"threshold_strategy": "percentile"}})
    assert r.status_code == 200, r.text
    assert r.json()["effective_config"]["lstm_threshold_strategy"] == "percentile"
    assert r.json()["lstm"]["threshold_strategy"] == "percentile"


def test_lstm_is_the_default_and_combinations_are_experimental(client_and_engine):
    client, _ = client_and_engine
    sel = client.get("/api/settings/ml").json()["online_selection"]
    assert sel["primary"] == sel["default_primary"] == "lstm_ae"
    assert sel["combo"] == "single" and sel["experimental"] is False
    assert sel["experimental_combos"] == ["cascade", "fusion"]
    r = client.patch("/api/settings/ml", json={"online": {"primary": "isolation_forest",
                                                         "secondary": "lstm_ae", "combo": "fusion"}})
    assert r.status_code == 200, r.text
    assert r.json()["online_selection"]["experimental"] is True
