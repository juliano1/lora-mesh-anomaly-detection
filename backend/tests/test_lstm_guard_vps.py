"""Perfil sem PyTorch: lstm_ae só é registrado quando ML_LSTM_ENABLED e torch disponível."""
from unittest.mock import patch

import pytest

from ai.online import registry as reg
import ai.online.pipeline as pl


def _ids_after_register(lstm_enabled: bool) -> set[str]:
    """Reexecuta _register_defaults com a flag desejada, isolando o registry global."""
    saved = dict(reg._registry)
    reg._registry.clear()
    try:
        with patch.object(pl.settings, "ML_LSTM_ENABLED", lstm_enabled):
            pl._register_defaults()
            return {spec.id for spec in reg.available()}
    finally:
        reg._registry.clear()
        reg._registry.update(saved)


def test_lstm_registered_when_enabled():
    pytest.importorskip("torch")
    ids = _ids_after_register(True)
    assert "lstm_ae" in ids  # torch presente no ambiente local de dev/test
    assert "isolation_forest" in ids


def test_lstm_not_registered_when_disabled():
    ids = _ids_after_register(False)
    assert "lstm_ae" not in ids
    assert ids == {"isolation_forest"}


def test_build_runners_fallback_when_primary_unavailable():
    """Primário não registrado → cai para isolation_forest e fica ready."""
    from ai.online.pipeline import OnlinePipeline

    class _ML:  # stub mínimo de ml_settings
        ml_if_contamination = 0.03
        ml_if_n_estimators = 100
        ml_lstm_window = 12

    pipe = OnlinePipeline(
        primary_id="lstm_ae",  # indisponível quando ML_LSTM_ENABLED=false
        secondary_id=None, combo="single", fusion_alpha=0.5,
    )
    with patch.object(pl.settings, "ML_LSTM_ENABLED", False):
        saved = dict(reg._registry)
        reg._registry.clear()
        pl._register_defaults()
        try:
            pipe.build_runners(_ML())
        finally:
            reg._registry.clear()
            reg._registry.update(saved)
    assert pipe.is_ready() is True
    assert pipe._primary_id == "isolation_forest"
    # O pacote levaria a identidade pedida (lstm_ae) com o IF dentro: congelar é recusado.
    with pytest.raises(ValueError, match="degradado"):
        pipe.export_bundle()


def test_put_ml_settings_rejects_unavailable_algorithm():
    """PATCH com primary=lstm_ae quando indisponível → 422 (não 500)."""
    from datetime import datetime, timedelta, timezone
    from fastapi.testclient import TestClient
    from sqlmodel import SQLModel, Session, create_engine
    from sqlalchemy.pool import StaticPool
    import jwt

    import main
    from database import get_session
    from auth import get_current_user, get_current_admin
    from config import settings as cfg

    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import models  # noqa
    SQLModel.metadata.create_all(eng)

    def _get_session():
        with Session(eng) as s:
            yield s

    # Seed: cria MLSettings e User admin para o endpoint funcionar
    from models import MLSettings, User
    with Session(eng) as s:
        s.add(User(id=1, username="admin", password_hash="x", is_admin=True))
        s.add(MLSettings(id=1, ml_online_primary="isolation_forest"))
        s.commit()

    main.app.dependency_overrides[get_session] = _get_session
    main.app.dependency_overrides[get_current_user] = lambda: "admin"
    main.app.dependency_overrides[get_current_admin] = lambda: type("U", (), {"id": 1, "is_admin": True})()

    token = jwt.encode(
        {"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        cfg.JWT_SECRET_KEY,
        algorithm=cfg.JWT_ALGORITHM,
    )

    with patch.object(pl.settings, "ML_LSTM_ENABLED", False):
        saved = dict(reg._registry)
        reg._registry.clear()
        pl._register_defaults()
        try:
            tc = TestClient(main.app)
            tc.headers.update({"Authorization": f"Bearer {token}"})
            resp = tc.patch("/api/settings/ml", json={"online": {"primary": "lstm_ae"}})
        finally:
            reg._registry.clear()
            reg._registry.update(saved)

    main.app.dependency_overrides.clear()

    assert resp.status_code == 422
    assert resp.status_code != 500


def test_build_runners_degrades_when_secondary_unavailable():
    """Secundário não registrado (ex.: lstm_ae sem PyTorch) → degrada p/ single, fica ready."""
    from ai.online.pipeline import OnlinePipeline

    class _ML:
        ml_if_contamination = 0.03
        ml_if_n_estimators = 100
        ml_lstm_window = 12

    pipe = OnlinePipeline(
        primary_id="isolation_forest",  # primário OK
        secondary_id="lstm_ae",         # secundário indisponível sem torch
        combo="cascade", fusion_alpha=0.5,
    )
    with patch.object(pl.settings, "ML_LSTM_ENABLED", False):
        saved = dict(reg._registry)
        reg._registry.clear()
        pl._register_defaults()
        try:
            pipe.build_runners(_ML())
        finally:
            reg._registry.clear()
            reg._registry.update(saved)
    assert pipe.is_ready() is True
    assert pipe._primary_id == "isolation_forest"
    assert pipe._secondary_id is None
    assert pipe._combo == "single"
