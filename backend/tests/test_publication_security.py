"""Regressões de segurança para o bootstrap e as rotas públicas."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
import jwt
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from config import Settings, settings, validate_security_settings


@pytest.mark.parametrize("secret", ["", "CHANGE-ME", "CHANGE_ME", "short"])
def test_startup_refuses_default_signing_key(secret):
    config = Settings(_env_file=None, JWT_SECRET_KEY=secret)
    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        validate_security_settings(config)


@pytest.mark.parametrize("algorithm", ["none", "RS256", "ES256"])
def test_startup_refuses_non_hmac_algorithm(algorithm):
    config = Settings(_env_file=None, JWT_SECRET_KEY="synthetic-test-key-with-at-least-32-bytes",  # pragma: allowlist secret
                      JWT_ALGORITHM=algorithm)
    with pytest.raises(ValueError, match="JWT_ALGORITHM"):
        validate_security_settings(config)


def test_explicit_configuration_is_accepted():
    config = Settings(_env_file=None, JWT_SECRET_KEY="synthetic-test-key-with-at-least-32-bytes",  # pragma: allowlist secret
                      ADMIN100_PASSWORD="", ADMIN200_PASSWORD="")
    validate_security_settings(config)


@pytest.mark.parametrize("payload", [
    {"sub": "operator"},
    {"exp": 1},
    {"sub": "", "exp": 9999999999},
    {"sub": "operator", "exp": 1},
])
def test_token_requires_subject_and_expiration(payload):
    from auth import decode_access_token
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(token)


@pytest.fixture
def client_and_session():
    import main
    from database import get_session
    from models import User
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(User(username="active", password_hash="unused", is_active=True))  # pragma: allowlist secret
        session.add(User(username="disabled", password_hash="unused", is_active=False, is_admin=True))  # pragma: allowlist secret
        session.commit()

    def override_session():
        with Session(engine) as session:
            yield session

    main.app.dependency_overrides[get_session] = override_session
    try:
        with Session(engine) as session:
            yield TestClient(main.app), session
    finally:
        main.app.dependency_overrides.clear()
        engine.dispose()


def _token(username):
    return jwt.encode({"sub": username, "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
                      settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


@pytest.mark.parametrize("username,status", [("active", 200), ("disabled", 401), ("absent", 401)])
def test_http_requires_existing_active_user(client_and_session, username, status):
    client, _ = client_and_session
    response = client.get("/nodes", headers={"Authorization": f"Bearer {_token(username)}"})
    assert response.status_code == status


def test_disabled_administrator_is_denied(client_and_session):
    from auth import get_current_user, require_admin
    _, session = client_and_session
    with pytest.raises(HTTPException) as exc:
        require_admin(session, "disabled")
    assert exc.value.status_code == 403
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=_token("disabled"))
    with pytest.raises(HTTPException) as exc:
        get_current_user(credentials, session)
    assert exc.value.status_code == 401


def test_login_rejects_password_larger_than_bcrypt_limit(client_and_session):
    client, _ = client_and_session
    response = client.post("/api/auth/login", json={"username": "active", "password": "é" * 37})
    assert response.status_code == 422


@pytest.mark.parametrize("method,path", [
    ("get", "/ws"),
    ("get", "/api/llm/analyses"),
    ("post", "/api/llm/test"),
    ("get", "/api/report/ax/options?network_id=1"),
    ("post", "/api/anomalies/compare"),
    ("post", "/api/report/feature-timeline/export"),
    ("get", "/api/battery-report?network_id=1"),
    ("post", "/api/send-message"),
])
def test_removed_components_have_no_routes(client_and_session, method, path):
    """Interface, LLM, Telegram, relatório AX/PDF e comparador não fazem parte da versão pública."""
    client, _ = client_and_session
    headers = {"Authorization": f"Bearer {_token('active')}"}
    response = getattr(client, method)(path, headers=headers)
    assert response.status_code == 404


def test_root_does_not_expose_configuration(client_and_session):
    client, _ = client_and_session
    body = client.get("/").json()
    assert "database" not in body


@pytest.mark.parametrize("module", [
    "ai.llm_worker", "ai.llm_prompt", "telegram_notifier", "connection_manager",
    "ai.ax_report", "ai.on_demand_detector",
])
def test_removed_modules_are_absent(module):
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.fixture
def login_client(client_and_session, monkeypatch):
    """Usuários com hash bcrypt real (custo baixo) e contador de falhas zerado."""
    import bcrypt
    from models import User
    from routers.auth import reset_login_failures
    client, session = client_and_session
    session.add(User(username="op", password_hash=bcrypt.hashpw(b"senha-sintetica",
                                                                bcrypt.gensalt(rounds=4)).decode()))
    session.add(User(username="adm", password_hash="unused", is_admin=True))  # pragma: allowlist secret
    session.commit()
    monkeypatch.setattr(settings, "LOGIN_MAX_FAILURES", 3)
    reset_login_failures()
    yield client
    reset_login_failures()


def _login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def test_login_succeeds_and_returns_token(login_client):
    response = _login(login_client, "op", "senha-sintetica")
    assert response.status_code == 200
    from auth import decode_access_token
    assert decode_access_token(response.json()["access_token"]) == "op"


def test_repeated_login_failures_are_throttled(login_client):
    assert [_login(login_client, "op", "errada").status_code for _ in range(3)] == [401] * 3
    blocked = _login(login_client, "op", "senha-sintetica")
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) > 0
    # O bloqueio é por usuário: outra conta do mesmo cliente segue com a resposta normal.
    assert _login(login_client, "outro", "errada").status_code == 401


def test_unknown_user_failures_also_count(login_client):
    for _ in range(3):
        assert _login(login_client, "fantasma", "x").status_code == 401
    assert _login(login_client, "FANTASMA", "x").status_code == 429


def test_successful_login_clears_failures(login_client):
    for _ in range(2):
        _login(login_client, "op", "errada")
    assert _login(login_client, "op", "senha-sintetica").status_code == 200
    for _ in range(2):
        assert _login(login_client, "op", "errada").status_code == 401


@pytest.mark.parametrize("method,path,body", [
    ("post", "/api/networks", {"name": "x", "gateways": [], "end_devices": []}),
    ("put", "/api/networks/999", {"name": "x", "gateways": [], "end_devices": []}),
    ("delete", "/api/networks/999", None),
    ("post", "/api/nodes/999/simulate", {"anomaly_type": "A1"}),
    ("delete", "/api/nodes/999/simulate", None),
])
def test_network_and_simulation_changes_require_admin(login_client, method, path, body):
    kwargs = {"json": body} if body is not None else {}
    as_user = login_client.request(method.upper(), path, headers={"Authorization": f"Bearer {_token('op')}"}, **kwargs)
    assert as_user.status_code == 403
    as_admin = login_client.request(method.upper(), path, headers={"Authorization": f"Bearer {_token('adm')}"}, **kwargs)
    assert as_admin.status_code in (404, 422), as_admin.text


def test_default_configuration_allows_no_browser_origin(client_and_session):
    client, _ = client_and_session
    response = client.get("/health", headers={"Origin": "https://example.org"})
    assert "access-control-allow-origin" not in response.headers
