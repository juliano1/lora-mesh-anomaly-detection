"""
Leitura da origem em snapshot único.

PostgreSQL: transação `REPEATABLE READ` somente leitura — todas as consultas
veem o mesmo snapshot (o padrão READ COMMITTED não garante isso).
SQLite: o driver pysqlite não emite BEGIN antes de SELECT; `Session.begin()`
sozinho não segura snapshot. O engine da exportação desliga o controle
transacional do driver e emite BEGIN explícito (padrão documentado do SQLAlchemy).
No journal padrão, essa transação de leitura bloqueia escritas enquanto dura.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlmodel import Session

from dataset_export.provenance import sha256_file


def build_source_engine(url: str):
    """Engine dedicado à leitura da origem da exportação."""
    parsed = make_url(url)
    if parsed.get_backend_name() not in ("sqlite", "postgresql"):
        raise ValueError("origem deve ser PostgreSQL ou SQLite")
    if parsed.get_backend_name() == "sqlite":
        if parsed.database and not parsed.database.startswith('file:') and parsed.database != ':memory:':
            if not Path(parsed.database).is_file():
                raise FileNotFoundError("arquivo SQLite de origem inexistente")
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _disable_driver_transactions(dbapi_connection, _record):
            dbapi_connection.isolation_level = None
            dbapi_connection.execute("PRAGMA query_only=ON")

        @event.listens_for(engine, "begin")
        def _explicit_begin(connection):
            connection.exec_driver_sql("BEGIN")

        return engine
    return create_engine(url)


@contextmanager
def snapshot_session(engine):
    """Sessão cujas consultas observam um único snapshot da origem."""
    with engine.connect() as connection:
        if engine.dialect.name == "postgresql":
            connection = connection.execution_options(
                isolation_level="REPEATABLE READ", postgresql_readonly=True
            )
        with connection.begin():
            with Session(bind=connection) as session:
                yield session


def describe_source(engine, sqlite_file: Optional[Path] = None) -> dict:
    """Identificação da fonte para o manifesto, sem URL nem credenciais."""
    dialect = engine.dialect.name
    if dialect == "sqlite":
        version = sqlite3.sqlite_version
    else:
        version = ".".join(str(p) for p in engine.dialect.server_version_info or ())
    info = {"dialect": dialect, "server_version": version, "file_sha256": None}
    if sqlite_file is not None and Path(sqlite_file).is_file():
        info["file_sha256"] = sha256_file(sqlite_file)
    return info
