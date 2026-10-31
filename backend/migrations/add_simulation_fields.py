"""
Adiciona campos de simulação de anomalias ao Node.
sim_mode, sim_anomaly, sim_started_at, sim_ended_at
"""

import logging
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

NEW_COLUMNS = {
    "sim_mode": "BOOLEAN DEFAULT FALSE NOT NULL",
    "sim_anomaly": "VARCHAR(3)",
    "sim_started_at": "TIMESTAMP",
    "sim_ended_at": "TIMESTAMP",
}


def migrate(engine):
    """Adiciona colunas de simulação à tabela node (idempotente)."""
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
