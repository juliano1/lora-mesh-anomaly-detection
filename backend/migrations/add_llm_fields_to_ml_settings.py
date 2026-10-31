"""
Legado (campos sem uso nesta versão): adiciona llm_enabled, llm_provider, llm_model
e telegram_enabled à tabela mlsettings.
"""

import logging
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

NEW_COLUMNS = {
    "llm_enabled": "BOOLEAN DEFAULT false",
    "llm_provider": "VARCHAR DEFAULT 'gemini'",
    "llm_model": "VARCHAR DEFAULT 'gemini-2.0-flash'",
    "telegram_enabled": "BOOLEAN DEFAULT false",
}


def migrate(engine):
    """Adiciona colunas LLM à tabela mlsettings (idempotente)."""
    inspector = inspect(engine)
    existing = {c["name"] for c in inspector.get_columns("mlsettings")}

    with engine.begin() as conn:
        for col_name, col_def in NEW_COLUMNS.items():
            if col_name not in existing:
                conn.execute(
                    text(f"ALTER TABLE mlsettings ADD COLUMN {col_name} {col_def}")
                )
                logger.info(f"Coluna '{col_name}' adicionada a mlsettings")
            else:
                logger.info(f"Coluna '{col_name}' já existe em mlsettings")
