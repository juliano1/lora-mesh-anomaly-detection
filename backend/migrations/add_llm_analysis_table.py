"""
Legado, não executado nesta versão: cria a tabela llm_analysis.

Em versões anteriores, cada registro guardava a decisão de um LLM sobre uma anomalia
detectada (então pelo Isolation Forest ou pelo LOF), como base para um ajuste fino do
LLM por LoRA (Low-Rank Adaptation, sem relação com o rádio LoRa).
"""

import logging
from sqlmodel import SQLModel, Session

logger = logging.getLogger(__name__)


def migrate(engine):
    """Cria tabela llm_analysis."""
    from models import LLMAnalysis

    # Cria a tabela (idempotente)
    SQLModel.metadata.create_all(engine, tables=[LLMAnalysis.__table__])
    logger.info("Tabela llm_analysis criada")
