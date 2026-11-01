"""Snapshot único nas leituras da exportação."""
import os

import pytest
from sqlalchemy import create_engine, text


def _seed(path):
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY)"))
        c.execute(text("INSERT INTO t VALUES (1)"))
    eng.dispose()


def test_sqlite_snapshot_is_stable_against_concurrent_write(tmp_path):
    """Regressão: sem BEGIN explícito, leituras do mesmo snapshot viam 1 e 2 linhas."""
    from dataset_export.snapshot import build_source_engine, snapshot_session
    db = tmp_path / "src.db"
    _seed(db)
    source = build_source_engine(f"sqlite:///{db}")
    writer = create_engine(f"sqlite:///{db}", connect_args={"timeout": 0.2})
    with snapshot_session(source) as s:
        first = s.execute(text("SELECT COUNT(*) FROM t")).scalar()
        try:
            with writer.begin() as c:
                c.execute(text("INSERT INTO t (id) VALUES (NULL)"))
        except Exception:
            pass  # journal padrão: a transação de leitura bloqueia a escrita
        second = s.execute(text("SELECT COUNT(*) FROM t")).scalar()
    assert first == second == 1
    source.dispose(); writer.dispose()


def test_plain_session_begin_would_not_hold_snapshot(tmp_path):
    """Controle: confirma que o problema existe sem o engine dedicado."""
    from sqlmodel import Session
    db = tmp_path / "src.db"
    _seed(db)
    plain = create_engine(f"sqlite:///{db}")
    writer = create_engine(f"sqlite:///{db}")
    with Session(plain) as s, s.begin():
        first = s.execute(text("SELECT COUNT(*) FROM t")).scalar()
        with writer.begin() as c:
            c.execute(text("INSERT INTO t (id) VALUES (NULL)"))
        second = s.execute(text("SELECT COUNT(*) FROM t")).scalar()
    assert (first, second) == (1, 2)
    plain.dispose(); writer.dispose()


def test_describe_sqlite_source(tmp_path):
    from dataset_export.snapshot import build_source_engine, describe_source
    db = tmp_path / "src.db"
    _seed(db)
    eng = build_source_engine(f"sqlite:///{db}")
    info = describe_source(eng, sqlite_file=db)
    assert info["dialect"] == "sqlite" and info["server_version"]
    assert len(info["file_sha256"]) == 64
    eng.dispose()


PG_URL = os.environ.get("TEST_POSTGRES_URL")


@pytest.mark.skipif(not PG_URL, reason="defina TEST_POSTGRES_URL para verificar no PostgreSQL")
def test_postgres_snapshot_is_stable_against_concurrent_write():
    from dataset_export.snapshot import build_source_engine, snapshot_session
    admin = create_engine(PG_URL)
    with admin.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS snapshot_test"))
        c.execute(text("CREATE TABLE snapshot_test (id SERIAL PRIMARY KEY)"))
        c.execute(text("INSERT INTO snapshot_test DEFAULT VALUES"))
    source = build_source_engine(PG_URL)
    with snapshot_session(source) as s:
        first = s.execute(text("SELECT COUNT(*) FROM snapshot_test")).scalar()
        with admin.begin() as c:
            c.execute(text("INSERT INTO snapshot_test DEFAULT VALUES"))
        second = s.execute(text("SELECT COUNT(*) FROM snapshot_test")).scalar()
    assert first == second == 1
    with admin.begin() as c:
        c.execute(text("DROP TABLE snapshot_test"))
    source.dispose(); admin.dispose()
