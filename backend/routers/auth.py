"""Endpoint de login com autenticação JWT."""

import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, status
import jwt
from sqlmodel import Session, select

from config import settings
from database import get_session
from models import User
from schemas.auth import LoginRequest, LoginResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Falhas recentes por (cliente, usuário). O backend roda com um único worker, então a
# contagem em memória cobre todas as requisições; reiniciar o processo zera os bloqueios.
_failures: dict[tuple[str, str], list[float]] = {}
_client_attempts: dict[str, list[float]] = {}
_failures_lock = threading.Lock()

# Limites por processo, aplicados antes do bcrypt. Ao atingir a capacidade, novas
# chaves aguardam a expiração; entradas ativas não são expulsas para liberar tentativas.
MAX_TRACKED_FAILURES = 4096
MAX_TRACKED_CLIENTS = 1024
MAX_ATTEMPTS_PER_CLIENT = 30
ATTEMPT_WINDOW_SECONDS = 60


@lru_cache(maxsize=1)
def _dummy_hash() -> bytes:
    """Hash descartável para que usuário inexistente custe o mesmo bcrypt que senha errada."""
    return bcrypt.hashpw(b"usuario-inexistente", bcrypt.gensalt(rounds=12))


def _failure_key(request: Request, username: str) -> tuple[str, str]:
    client = request.client.host if request.client else "desconhecido"
    return client, username.strip().lower()


def _recent_failures(key: tuple[str, str], now: float) -> list[float]:
    window_start = now - settings.LOGIN_LOCKOUT_SECONDS
    recent = [t for t in _failures.get(key, []) if t > window_start]
    if recent:
        _failures[key] = recent
    else:
        _failures.pop(key, None)
    return recent


def _prune_login_state(now: float) -> None:
    """Expira também usuários e clientes que não voltaram a fazer requisições.

    O chamador mantém _failures_lock durante a limpeza e a admissão da tentativa.
    """
    for key in list(_failures):
        _recent_failures(key, now)
    for client, attempts in list(_client_attempts.items()):
        recent = [t for t in attempts if t > now - ATTEMPT_WINDOW_SECONDS]
        if recent:
            _client_attempts[client] = recent
        else:
            del _client_attempts[client]


def _reject_login(retry_after: int) -> None:
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Muitas tentativas de login; tente novamente mais tarde",
        headers={"Retry-After": str(max(1, retry_after))},
    )


def reset_login_failures() -> None:
    """Limpa os bloqueios (usado pelos testes)."""
    with _failures_lock:
        _failures.clear()
        _client_attempts.clear()


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request, session: Session = Depends(get_session)):
    """Autentica usuário e retorna token JWT."""
    key = _failure_key(request, body.username)
    now = time.monotonic()
    with _failures_lock:
        _prune_login_state(now)
        client = key[0]
        attempts = _client_attempts.get(client, [])
        if len(attempts) >= MAX_ATTEMPTS_PER_CLIENT:
            _reject_login(int(attempts[0] + ATTEMPT_WINDOW_SECONDS - now))
        if ((client not in _client_attempts and len(_client_attempts) >= MAX_TRACKED_CLIENTS)
                or (key not in _failures and len(_failures) >= MAX_TRACKED_FAILURES)):
            _reject_login(ATTEMPT_WINDOW_SECONDS)
        recent = _recent_failures(key, now)
        if len(recent) >= settings.LOGIN_MAX_FAILURES:
            retry_after = max(1, int(recent[0] + settings.LOGIN_LOCKOUT_SECONDS - now))
            _reject_login(retry_after)
        _client_attempts.setdefault(client, []).append(now)

    user = session.exec(
        select(User).where(User.username == body.username, User.is_active == True)  # noqa: E712
    ).first()
    stored_hash = user.password_hash.encode("utf-8") if user else _dummy_hash()
    password_ok = bcrypt.checkpw(body.password.encode("utf-8"), stored_hash)

    if not user or not password_ok:
        with _failures_lock:
            # Outras requisições podem ter terminado o bcrypt enquanto esta aguardava.
            # Conferir novamente a capacidade impede ultrapassá-la por concorrência.
            if key in _failures or len(_failures) < MAX_TRACKED_FAILURES:
                failed_at = time.monotonic()
                failures = _recent_failures(key, failed_at)
                _failures[key] = (failures + [failed_at])[-settings.LOGIN_MAX_FAILURES:]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha inválidos",
        )

    with _failures_lock:
        _failures.pop(key, None)

    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    token = jwt.encode(
        {"sub": user.username, "exp": expire},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return LoginResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRATION_MINUTES * 60,
    )
