"""
Adiciona coluna anomaly_classification_mode à tabela mlsettings.
Idempotente — seguro para re-execução.

Valores: "contextual" (padrão, usa o contexto de ambiente da rede) | "absolute" (thresholds fixos clássicos)
"""
from sqlalchemy import inspect, text


def upgrade(engine):
    inspector = inspect(engine)
    existing = {c["name"] for c in inspector.get_columns("mlsettings")}
    if "anomaly_classification_mode" not in existing:
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(text(
                    "ALTER TABLE mlsettings ADD COLUMN IF NOT EXISTS anomaly_classification_mode VARCHAR(20) DEFAULT 'contextual'"
                ))
            else:
                conn.execute(text(
                    "ALTER TABLE mlsettings ADD COLUMN anomaly_classification_mode VARCHAR(20) DEFAULT 'contextual'"
                ))
