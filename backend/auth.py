"""Validação de tokens e autorização de usuários ativos."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from sqlmodel import Session, select

from config import settings
from database import get_session

security = HTTPBearer()


def decode_access_token(token: str) -> str:
    """Exige assinatura, expiração e identificação do usuário."""
    payload = jwt.decode(
        token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["exp", "sub"]},
    )
    username = payload.get("sub")
    if not isinstance(username, str) or not username.strip():
        raise jwt.InvalidTokenError("Identificação do usuário inválida")
    return username


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session),
) -> str:
    """Valida o token JWT e retorna o username."""
    try:
        username = decode_access_token(credentials.credentials)
        user = resolve_user(session, username)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
            )
        return username
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
        )


def resolve_user(session: "Session", username: str):
    from models import User
    return session.exec(select(User).where(User.username == username)).first()


def require_admin(session: "Session", username: str):
    """Levanta 403 se o usuário não existir ou não for admin (checagem no banco)."""
    user = resolve_user(session, username)
    if not user or not user.is_active or not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Requer privilégio de administrador")
    return user


def get_current_admin(
    username: str = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Dependência FastAPI: exige admin (verificado no banco, não só no JWT)."""
    return require_admin(session, username)
