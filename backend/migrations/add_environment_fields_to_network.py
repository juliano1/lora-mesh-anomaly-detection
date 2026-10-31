"""
Adiciona campos de contexto de ambiente à tabela network.

Campos básicos:
  environment_type, building_type, floors_count,
  predominant_obstacle_material, obstacle_density,
  typical_link_distance_class, availability_criticality

Campos avançados iniciais (opcionais na API):
  avg_install_height_m, avg_walls_per_link, node_mobility_profile

Compatibilidade retroativa: todas as colunas são nullable.
Redes existentes ficam com NULL e environment_context_complete=False.
"""

import logging
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

# (nome, DDL — compatível com PostgreSQL e SQLite)
NEW_COLUMNS = {
    "environment_type":              "VARCHAR(20)",
    "building_type":                 "VARCHAR(30)",
    "floors_count":                  "INTEGER",
    "predominant_obstacle_material": "VARCHAR(20)",
    "obstacle_density":              "VARCHAR(10)",
    "typical_link_distance_class":   "VARCHAR(10)",
    "availability_criticality":      "VARCHAR(10)",
    "avg_install_height_m":          "FLOAT",
    "avg_walls_per_link":            "INTEGER",
    "node_mobility_profile":         "VARCHAR(15)",
}


def migrate(engine):
    """Adiciona colunas de contexto de ambiente à tabela network (idempotente)."""
    inspector = inspect(engine)
    existing = {c["name"] for c in inspector.get_columns("network")}

    with engine.begin() as conn:
        for col_name, col_def in NEW_COLUMNS.items():
            if col_name not in existing:
                conn.execute(
                    text(f"ALTER TABLE network ADD COLUMN {col_name} {col_def}")
                )
                logger.info(f"Coluna '{col_name}' adicionada a network")
            else:
                logger.info(f"Coluna '{col_name}' já existe em network")
