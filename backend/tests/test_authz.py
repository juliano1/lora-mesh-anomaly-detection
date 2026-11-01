"""Autorização de administrador."""
import pytest
from fastapi import HTTPException
from sqlmodel import SQLModel, Session, create_engine


@pytest.fixture
def session():
    eng = create_engine("sqlite:///:memory:")
    import models  # noqa
    SQLModel.metadata.create_all(eng)
    from models import User
    with Session(eng) as s:
        s.add(User(id=1, username="admin", password_hash="x", is_admin=True))
        s.add(User(id=2, username="op", password_hash="x", is_admin=False))
        s.commit()
        yield s


def test_resolve_user_returns_row(session):
    from auth import resolve_user
    u = resolve_user(session, "admin")
    assert u.is_admin is True


def test_require_admin_ok(session):
    from auth import require_admin
    require_admin(session, "admin")  # não levanta


def test_require_admin_denies_non_admin(session):
    from auth import require_admin
    with pytest.raises(HTTPException) as exc:
        require_admin(session, "op")
    assert exc.value.status_code == 403


def test_require_admin_denies_unknown_user(session):
    from auth import require_admin
    with pytest.raises(HTTPException) as exc:
        require_admin(session, "fantasma")
    assert exc.value.status_code == 403
