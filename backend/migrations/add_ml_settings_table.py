"""
Cria a tabela mlsettings com registro singleton (id=1).

Parâmetros de detecção contínua persistidos no banco — prevalecem sobre config.py.
"""

import logging
from sqlmodel import SQLModel, Session

logger = logging.getLogger(__name__)


def migrate(engine):
    """Cria tabela mlsettings com valores padrão."""
    from models import MLSettings

    # Criar tabela (idempotente)
    SQLModel.metadata.create_all(engine, tables=[MLSettings.__table__])

    # Inserir registro singleton se não existir
    with Session(engine) as session:
        existing = session.get(MLSettings, 1)
        if not existing:
            session.add(MLSettings(id=1))
            session.commit()
            logger.info("Tabela ml_settings criada com valores padrão")
        else:
            logger.info("Tabela ml_settings já existe")
