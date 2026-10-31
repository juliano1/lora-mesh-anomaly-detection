"""
Adiciona coluna classification_mode_used à tabela anomaly.
Idempotente — seguro para re-execução.

Registra qual modo de classificação estava ativo no momento da detecção,
permitindo análise A/B e auditoria retroativa.

Valores: "absolute" | "contextual" | "contextual_fallback"
"""
from sqlalchemy import inspect, text


def upgrade(engine):
    inspector = inspect(engine)
    existing = {c["name"] for c in inspector.get_columns("anomaly")}
    if "classification_mode_used" not in existing:
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(text(
                    "ALTER TABLE anomaly ADD COLUMN IF NOT EXISTS classification_mode_used VARCHAR(30)"
                ))
            else:
                conn.execute(text(
                    "ALTER TABLE anomaly ADD COLUMN classification_mode_used VARCHAR(30)"
                ))
