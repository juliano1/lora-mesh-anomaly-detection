"""
Renomeia anomaly_type de IF1-IF7 para A1-A7 nas tabelas anomaly e llm_analysis.

Nomenclatura agnóstica ao algoritmo — os tipos descrevem a anomalia, não o detector.
Migração idempotente: executa UPDATE apenas se existirem registros com prefixo 'IF'.
"""

import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Mapeamento IF → A
RENAME_MAP = {
    "IF1": "A1",
    "IF2": "A2",
    "IF3": "A3",
    "IF4": "A4",
    "IF5": "A5",
    "IF6": "A6",
    "IF7": "A7",
}


def migrate(engine):
    """Renomeia anomaly_type IF1-IF7 → A1-A7 (idempotente)."""
    with engine.begin() as conn:
        for old, new in RENAME_MAP.items():
            # Tabela anomaly
            result = conn.execute(
                text("UPDATE anomaly SET anomaly_type = :new WHERE anomaly_type = :old"),
                {"old": old, "new": new},
            )
            if result.rowcount > 0:
                logger.info(f"anomaly.anomaly_type '{old}' → '{new}' ({result.rowcount} rows)")

            # Tabela llm_analysis (se existir) — SAVEPOINT protege contra
            # tabela inexistente sem invalidar a transação em PostgreSQL
            try:
                conn.execute(text("SAVEPOINT sp_llm"))
                result = conn.execute(
                    text("UPDATE llm_analysis SET anomaly_type = :new WHERE anomaly_type = :old"),
                    {"old": old, "new": new},
                )
                conn.execute(text("RELEASE SAVEPOINT sp_llm"))
                if result.rowcount > 0:
                    logger.info(f"llm_analysis.anomaly_type '{old}' → '{new}' ({result.rowcount} rows)")
            except Exception:
                conn.execute(text("ROLLBACK TO SAVEPOINT sp_llm"))

    logger.info("Migração anomaly_type IF→A concluída")
