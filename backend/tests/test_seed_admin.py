"""seed_users garante is_admin=True em DB novo e é idempotente."""

import pytest
from sqlmodel import Session, SQLModel, create_engine, select


@pytest.fixture
def mem_engine(monkeypatch):
    import models  # noqa: F401  (registra metadata)

    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    import seed_users

    monkeypatch.setattr(seed_users, "engine", eng)
    monkeypatch.setattr(seed_users.settings, "ADMIN100_PASSWORD", "synthetic-password-for-tests-100")
    monkeypatch.setattr(seed_users.settings, "ADMIN200_PASSWORD", "synthetic-password-for-tests-200")
    return eng, seed_users


def test_seed_creates_admins_as_admin(mem_engine):
    """Fresh DB: ambos admin100 e admin200 são criados com is_admin=True."""
    eng, seed_users = mem_engine
    seed_users.seed_users()

    from models import User

    with Session(eng) as s:
        for uname in ("admin100", "admin200"):
            u = s.exec(select(User).where(User.username == uname)).first()
            assert u is not None, f"{uname} não foi criado"
            assert u.is_admin is True, f"{uname} deveria ser admin"


def test_seed_preserves_existing_non_admin(mem_engine):
    """Reiniciar o servidor não restaura privilégios removidos pelo operador."""
    eng, seed_users = mem_engine
    from models import User

    # Uma conta existente pode ter perdido o privilégio por decisão do operador.
    with Session(eng) as s:
        s.add(User(username="admin100", password_hash="x", is_admin=False))
        s.add(User(username="admin200", password_hash="y", is_admin=False))
        s.commit()

    seed_users.seed_users()

    with Session(eng) as s:
        for uname in ("admin100", "admin200"):
            u = s.exec(select(User).where(User.username == uname)).first()
            assert u is not None
            assert u.is_admin is False
            assert u.password_hash in {"x", "y"}


def test_seed_is_idempotent(mem_engine):
    """Rodar duas vezes não duplica usuários e mantém is_admin=True."""
    eng, seed_users = mem_engine
    seed_users.seed_users()
    seed_users.seed_users()  # re-run

    from models import User

    with Session(eng) as s:
        users = s.exec(select(User)).all()
        seeded = [u for u in users if u.username in ("admin100", "admin200")]
        assert len(seeded) == 2, "não deve duplicar usuários seed"
        assert all(u.is_admin for u in seeded)


def test_seed_without_passwords_creates_no_accounts(mem_engine, monkeypatch):
    eng, seed_users = mem_engine
    monkeypatch.setattr(seed_users.settings, "ADMIN100_PASSWORD", "")
    monkeypatch.setattr(seed_users.settings, "ADMIN200_PASSWORD", "")
    seed_users.seed_users()
    from models import User
    with Session(eng) as session:
        assert session.exec(select(User)).all() == []


@pytest.mark.parametrize("password", ["CHANGE_ME", "CHANGE-ME", "short", "é" * 37])
def test_seed_refuses_unsafe_password_before_creating_accounts(mem_engine, monkeypatch, password):
    eng, seed_users = mem_engine
    monkeypatch.setattr(seed_users.settings, "ADMIN200_PASSWORD", password)
    with pytest.raises(ValueError):
        seed_users.seed_users()
    from models import User
    with Session(eng) as session:
        assert session.exec(select(User)).all() == []
