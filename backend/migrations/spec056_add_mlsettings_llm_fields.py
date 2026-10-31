"""
Legado (campos sem uso nesta versão): configuração do LLM em mlsettings.
Idempotente. Aditiva.
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
    dialect = engine.dialect.name
    additions = [
        ("llm_base_url",        "VARCHAR DEFAULT 'http://ollama:11434'"),
        ("llm_temperature",     "FLOAT DEFAULT 0.2"),
        ("llm_max_tokens",      "INTEGER DEFAULT 300"),
        ("llm_num_ctx",         "INTEGER DEFAULT 4096"),
        ("llm_timeout_seconds", "INTEGER DEFAULT 30"),
        ("llm_max_retries",     "INTEGER DEFAULT 2"),
    ]
    with engine.connect() as conn:
        cols = _existing_columns(conn, dialect, "mlsettings")
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
