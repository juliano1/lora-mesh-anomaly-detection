"""Inicia a API e o serviço MQTT (uvicorn).

Endereço e porta vêm de API_HOST/API_PORT (padrão 127.0.0.1:8000). Para recarregar
ao editar o código, defina UVICORN_RELOAD=true.
"""
import os

import uvicorn

from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
        workers=1,  # o loop de detecção e o cliente MQTT vivem no processo; não use mais de 1
        log_level="info",
    )
