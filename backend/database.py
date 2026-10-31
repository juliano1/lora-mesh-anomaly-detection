from sqlmodel import SQLModel, create_engine, Session
from config import settings


def _build_engine(url: str):
    """Cria engine com parâmetros corretos para SQLite ou PostgreSQL."""
    connect_args = {}
    if url.startswith("sqlite"):
        # check_same_thread=False é necessário para SQLite com FastAPI/Async
        connect_args["check_same_thread"] = False

    return create_engine(url, connect_args=connect_args)


engine = _build_engine(settings.DATABASE_URL)


def get_session():
    with Session(engine) as session:
        yield session
