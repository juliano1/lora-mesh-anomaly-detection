"""
Adiciona coluna ml_detect_simulated à tabela mlsettings.
Idempotente — seguro para re-execução.
"""
from sqlalchemy import inspect, text


def upgrade(engine):
    inspector = inspect(engine)
    existing = {c["name"] for c in inspector.get_columns("mlsettings")}
    if "ml_detect_simulated" not in existing:
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(text(
                    "ALTER TABLE mlsettings ADD COLUMN IF NOT EXISTS ml_detect_simulated BOOLEAN DEFAULT TRUE"
                ))
            else:
                conn.execute(text(
                    "ALTER TABLE mlsettings ADD COLUMN ml_detect_simulated BOOLEAN DEFAULT TRUE"
                ))
