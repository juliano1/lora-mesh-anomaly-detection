"""
Migração — adiciona campos de seleção online de algoritmos em mlsettings.

Novos campos:
    ml_online_primary    VARCHAR(30) DEFAULT 'isolation_forest'
    ml_online_secondary  VARCHAR(30) DEFAULT NULL
    ml_online_combo      VARCHAR(10) DEFAULT 'single'
    ml_online_fusion_alpha FLOAT DEFAULT 0.5

Idempotente — seguro para re-execução.
"""
from sqlalchemy import text


def _existing_columns(conn, dialect: str, table_name: str) -> set:
    if dialect == "postgresql":
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :table_name AND table_schema = current_schema()"
        ), {"table_name": table_name})
        return {r[0] for r in result.fetchall()}
    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
    return {r[1] for r in result.fetchall()}


def upgrade(engine) -> None:
    """Idempotente: adiciona colunas apenas se ainda não existirem."""
    dialect = engine.dialect.name
    with engine.connect() as conn:
        cols = _existing_columns(conn, dialect, "mlsettings")
        additions = [
            ("ml_online_primary",     "VARCHAR(30) DEFAULT 'isolation_forest'"),
            ("ml_online_secondary",   "VARCHAR(30) DEFAULT NULL"),
            ("ml_online_combo",       "VARCHAR(10) DEFAULT 'single'"),
            ("ml_online_fusion_alpha","FLOAT DEFAULT 0.5"),
        ]
        for col, col_def in additions:
            if col not in cols:
                if dialect == "postgresql":
                    conn.execute(text(
                        f"ALTER TABLE mlsettings ADD COLUMN IF NOT EXISTS {col} {col_def}"
                    ))
                else:
                    conn.execute(text(
                        f"ALTER TABLE mlsettings ADD COLUMN {col} {col_def}"
                    ))
        conn.commit()
