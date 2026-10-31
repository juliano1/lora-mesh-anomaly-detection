"""Migrações de inicialização em SQLite limpo, idempotentes e com falha fechada."""
import pytest
from sqlalchemy import inspect
from sqlmodel import SQLModel, create_engine


def test_startup_migrations_on_clean_sqlite_are_idempotent(tmp_path):
    import models  # noqa: F401
    from main import run_startup_migrations
    engine = create_engine(f"sqlite:///{tmp_path / 'limpo.db'}")
    SQLModel.metadata.create_all(engine)
    expected = {t: {c["name"] for c in inspect(engine).get_columns(t)} for t in inspect(engine).get_table_names()}
    run_startup_migrations(engine)
    run_startup_migrations(engine)
    after = {t: {c["name"] for c in inspect(engine).get_columns(t)} for t in inspect(engine).get_table_names()}
    # O ORM já cria o esquema completo; as migrações não removem nem duplicam colunas.
    assert all(expected[t] <= after[t] for t in expected)
    assert "is_admin" in after["user"]
    engine.dispose()


def test_failed_migration_stops_startup(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "STARTUP_MIGRATIONS", ("modulo_inexistente",))
    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
    with pytest.raises(RuntimeError, match="modulo_inexistente"):
        main.run_startup_migrations(engine)
    engine.dispose()


def test_legacy_database_missing_columns_is_upgraded(tmp_path):
    from sqlalchemy import text
    import models  # noqa: F401
    from main import run_startup_migrations
    engine = create_engine(f"sqlite:///{tmp_path / 'legado.db'}")
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE "user" DROP COLUMN is_admin'))
        conn.execute(text("ALTER TABLE telemetry DROP COLUMN data_origin"))
    run_startup_migrations(engine)
    columns = lambda t: {c["name"] for c in inspect(engine).get_columns(t)}
    assert "is_admin" in columns("user") and "data_origin" in columns("telemetry")
    engine.dispose()


@pytest.mark.skipif(not __import__("os").environ.get("SPEC064_PG_URL"),
                    reason="defina SPEC064_PG_URL (PostgreSQL descartável de teste)")
def test_startup_migrations_on_postgres_are_idempotent():
    import os
    import uuid
    from sqlalchemy import text
    import models  # noqa: F401
    from main import run_startup_migrations
    url = os.environ["SPEC064_PG_URL"]
    schema = f"mig_{uuid.uuid4().hex[:8]}"  # schema descartável; node↔simulation_run impede drop_all
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        SQLModel.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(text('ALTER TABLE "user" DROP COLUMN is_admin'))
        run_startup_migrations(engine)
        run_startup_migrations(engine)
        assert "is_admin" in {c["name"] for c in inspect(engine).get_columns("user", schema=schema)}
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
