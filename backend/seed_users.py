"""Cria contas iniciais somente quando suas senhas são configuradas."""

import logging

import bcrypt
from sqlmodel import Session, select

from config import settings, validate_bootstrap_password
from database import engine
from models import User

logger = logging.getLogger(__name__)


def seed_users():
    """Cria contas novas; preserva senha, estado e permissões de contas existentes."""
    users_to_seed = [
        ("admin100", settings.ADMIN100_PASSWORD),
        ("admin200", settings.ADMIN200_PASSWORD),
    ]
    for _, password in users_to_seed:
        validate_bootstrap_password(password)

    with Session(engine) as session:
        for username, password in users_to_seed:
            if not password:
                continue
            existing = session.exec(
                select(User).where(User.username == username)
            ).first()
            if not existing:
                hashed = bcrypt.hashpw(
                    password.encode("utf-8"),
                    bcrypt.gensalt(rounds=12),
                )
                user = User(
                    username=username,
                    password_hash=hashed.decode("utf-8"),
                    is_admin=True,
                )
                session.add(user)
                logger.info("Usuário '%s' criado.", username)
            else:
                logger.info("Usuário '%s' já existe, pulando.", username)
        session.commit()
