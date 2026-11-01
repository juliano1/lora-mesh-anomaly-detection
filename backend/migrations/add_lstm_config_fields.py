"""
Migração — parâmetros LSTM explícitos em mlsettings.

Colunas anuláveis; NULL significa "usar o default efetivo documentado"
(ex.: histórico do LSTM = ML_HISTORY_HOURS = 24 h). Nenhum valor efetivo muda.
Idempotente — seguro para re-execução.
"""
from sqlalchemy import text

_ADDITIONS = [
    ("ml_lstm_history_hours", "FLOAT"),
    ("ml_lstm_hidden", "INTEGER"),
    ("ml_lstm_epochs", "INTEGER"),
    ("ml_lstm_patience", "INTEGER"),
    ("ml_lstm_score_aggregation", "VARCHAR(8)"),
    ("ml_lstm_threshold_percentile", "FLOAT"),
]


def _existing_columns(conn, dialect: str) -> set:
    if dialect == "postgresql":
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'mlsettings'"
        ))
        return {r[0] for r in result.fetchall()}
    result = conn.execute(text("PRAGMA table_info(mlsettings)"))
    return {r[1] for r in result.fetchall()}


def upgrade(engine) -> None:
    dialect = engine.dialect.name
    with engine.connect() as conn:
        cols = _existing_columns(conn, dialect)
        for col, col_type in _ADDITIONS:
            if col in cols:
                continue
            if dialect == "postgresql":
                conn.execute(text(f"ALTER TABLE mlsettings ADD COLUMN IF NOT EXISTS {col} {col_type} NULL"))
            else:
                conn.execute(text(f"ALTER TABLE mlsettings ADD COLUMN {col} {col_type}"))
        conn.commit()
