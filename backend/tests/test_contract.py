"""Validação de estado final + remoção do híbrido do contrato."""
import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool


@pytest.fixture
def client():
    eng = create_engine("sqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    import models  # noqa
    SQLModel.metadata.create_all(eng)
    import main
    from database import get_session
    from auth import get_current_user, get_current_admin

    def _get_session():
        with Session(eng) as s:
            yield s

    with Session(eng) as s:
        from models import User, MLSettings
        s.add(User(id=1, username="admin", password_hash="x", is_admin=True))
        s.add(MLSettings(id=1))
        s.commit()

    main.app.dependency_overrides[get_session] = _get_session
    main.app.dependency_overrides[get_current_user] = lambda: "admin"
    main.app.dependency_overrides[get_current_admin] = lambda: type("U", (), {"id": 1, "is_admin": True})()

    import jwt
    from config import settings
    token = jwt.encode({"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                       settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    tc = TestClient(main.app)
    tc.headers.update({"Authorization": f"Bearer {token}"})
    yield tc
    main.app.dependency_overrides.clear()


# ── Validação do estado final da seleção online ───────────────────────────
def test_patch_rejects_cascade_without_secondary(client):
    r = client.patch("/api/settings/ml", json={"online": {"primary": "isolation_forest",
                                                           "secondary": None, "combo": "cascade"}})
    assert r.status_code == 422


def test_patch_rejects_same_primary_and_secondary(client):
    r = client.patch("/api/settings/ml", json={"online": {"primary": "isolation_forest",
                                                           "secondary": "isolation_forest", "combo": "shadow"}})
    assert r.status_code == 422


def test_patch_rejects_single_with_secondary(client):
    r = client.patch("/api/settings/ml", json={"online": {"primary": "isolation_forest",
                                                           "secondary": "lstm_ae", "combo": "single"}})
    assert r.status_code == 422


def test_patch_validates_merged_state(client):
    client.patch("/api/settings/ml", json={"online": {"primary": "isolation_forest",
                                                       "secondary": None, "combo": "single"}})
    r = client.patch("/api/settings/ml", json={"online": {"combo": "cascade"}})
    assert r.status_code == 422


def test_patch_accepts_valid_cascade(client):
    r = client.patch("/api/settings/ml", json={"online": {"primary": "isolation_forest",
                                                          "secondary": "lstm_ae", "combo": "cascade"}})
    assert r.status_code == 200


def test_patch_honors_explicit_secondary_null(client):
    """secondary:null explícito é aplicado (não ignorado).

    Após um cascade válido, enviar só {secondary: null} deve LIMPAR o secundário;
    com o combo ainda em cascade, o estado mesclado fica inválido → 422 (prova que
    o null foi honrado; se fosse ignorado, o request passaria com 200)."""
    r0 = client.patch("/api/settings/ml", json={"online": {"primary": "isolation_forest",
                                                            "secondary": "lstm_ae", "combo": "cascade"}})
    assert r0.status_code == 200
    r1 = client.patch("/api/settings/ml", json={"online": {"secondary": None}})
    assert r1.status_code == 422


def test_patch_clear_secondary_and_single_ok(client):
    """secondary:null + combo:single juntos = estado válido (single sem secundário)."""
    client.patch("/api/settings/ml", json={"online": {"primary": "isolation_forest",
                                                       "secondary": "lstm_ae", "combo": "cascade"}})
    r = client.patch("/api/settings/ml", json={"online": {"secondary": None, "combo": "single"}})
    assert r.status_code == 200
    body = client.get("/api/settings/ml").json()
    assert body["online_selection"]["secondary"] is None
    assert body["online_selection"]["combo"] == "single"


# ── Contrato sem hybrid/active_detection ───────────────────────────────────
def test_settings_ml_has_no_hybrid_or_active_detection(client):
    body = client.get("/api/settings/ml").json()
    assert "hybrid" not in body
    assert "active_detection" not in body
    assert "online_selection" in body
    assert "available_algorithms" in body


# ── Endpoints híbridos ausentes ────────────────────────────────────────────
def test_hybrid_endpoints_removed(client):
    assert client.get("/api/ml/hybrid/status").status_code == 404
    assert client.post("/api/ml/hybrid/train?network_id=1").status_code == 404
