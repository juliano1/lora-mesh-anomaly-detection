"""
Adiciona campos battery_type, battery_capacity_mah e power_source ao Node.
"""

import logging
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

NEW_COLUMNS = {
    "battery_type": "VARCHAR(20)",
    "battery_capacity_mah": "INTEGER",
    "power_source": "VARCHAR(20)",
}


def migrate(engine):
    """Adiciona colunas de bateria/alimentação à tabela node (idempotente)."""
    inspector = inspect(engine)
    existing = {c["name"] for c in inspector.get_columns("node")}

    with engine.begin() as conn:
        for col_name, col_def in NEW_COLUMNS.items():
            if col_name not in existing:
                conn.execute(
                    text(f"ALTER TABLE node ADD COLUMN {col_name} {col_def}")
                )
                logger.info(f"Coluna '{col_name}' adicionada a node")
            else:
                logger.info(f"Coluna '{col_name}' já existe em node")
