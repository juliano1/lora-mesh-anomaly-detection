"""
Legado (tabela sem uso nesta versão): colunas de auditoria, snapshot e fila em
llm_analysis (aditivas).
A conversão NOT NULL->nullable dos campos de resultado é feita em
spec056_llm_analysis_nullable.py (deve rodar DEPOIS desta).
Idempotente.
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


ADDITIONS = [
    ("network_id",             "INTEGER"),
    ("status",                 "VARCHAR DEFAULT 'success'"),
    ("system_prompt",          "TEXT"),
    ("user_prompt",            "TEXT"),
    ("raw_response",           "TEXT"),
    ("parse_error",            "TEXT"),
    ("provider",               "VARCHAR"),
    ("mode",                   "VARCHAR"),
    ("temperature",            "FLOAT"),
    ("max_tokens",             "INTEGER"),
    ("num_ctx",                "INTEGER"),
    ("top_p",                  "FLOAT"),
    ("seed",                   "INTEGER"),
    ("llm_timeout_seconds",    "INTEGER"),
    ("llm_max_retries",        "INTEGER"),
    ("prompt_version",         "VARCHAR"),
    ("context_reference_time", "TIMESTAMP"),
    ("started_at",             "TIMESTAMP"),
    ("finished_at",            "TIMESTAMP"),
    ("latency_ms",             "INTEGER"),
    ("input_tokens",           "INTEGER"),
    ("output_tokens",          "INTEGER"),
    ("attempt_count",          "INTEGER DEFAULT 0"),
    ("lease_recovery_count",   "INTEGER DEFAULT 0"),
    ("telegram_status",        "VARCHAR"),
    ("telegram_sent_at",       "TIMESTAMP"),
    ("telegram_error",         "TEXT"),
    ("enqueued_at",            "TIMESTAMP"),
    ("claimed_at",             "TIMESTAMP"),
    ("worker_id",              "VARCHAR"),
    ("lease_expires_at",       "TIMESTAMP"),
]


def upgrade(engine) -> None:
    dialect = engine.dialect.name
    with engine.connect() as conn:
        cols = _existing_columns(conn, dialect, "llm_analysis")
        for col, col_def in ADDITIONS:
            if col not in cols:
                if dialect == "postgresql":
                    conn.execute(text(
                        f"ALTER TABLE llm_analysis ADD COLUMN IF NOT EXISTS {col} {col_def}"
                    ))
                else:
                    conn.execute(text(
                        f"ALTER TABLE llm_analysis ADD COLUMN {col} {col_def}"
                    ))
        conn.commit()
