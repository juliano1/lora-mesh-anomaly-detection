"""
Fixtures compartilhadas para os testes do backend.

Fornece:
    - engine_memory: SQLite em memória com schema completo
    - session: sessão SQLModel sobre o engine em memória
    - sample_features: vetor de features sintético para um nó
    - sample_sequence: tensor (W, 11) de features sintéticas
    - sample_dataset: tensor (N, W, 11) de sequências sintéticas
"""

import random
from datetime import datetime, timedelta
from typing import Generator

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de teste
# ──────────────────────────────────────────────────────────────────────────────

N_FEATURES = 11
WINDOW_SIZE = 12
N_SEQUENCES = 50        # mínimo para treino nos testes
RANDOM_SEED = 0


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures de banco
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def engine_memory():
    """SQLite em memória com schema completo do projeto."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    # Importar modelos para que SQLModel crie as tabelas
    from models import (  # noqa: F401
        Network, Node, Telemetry, Neighbor, Anomaly,
        MLSettings, MessageLog, LLMAnalysis, User,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine_memory) -> Generator[Session, None, None]:
    """Sessão SQLModel limpa (rollback após cada teste)."""
    with Session(engine_memory) as s:
        yield s


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures de dados sintéticos
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_features() -> dict:
    """Vetor de features sintético representando condição normal."""
    return {
        "rssi_mean": -75.0,
        "rssi_std": 2.5,
        "snr_mean": 8.0,
        "snr_std": 1.2,
        "pdr_mean": 8.5,
        "battery_delta": -0.5,
        "packet_loss_rate": 0.02,
        "rssi_snr_divergence": 0.5,
        "beacon_interval_std": 0.8,
        "route_change_rate": 0.05,
        "link_asymmetry": 3.0,
    }


@pytest.fixture
def sample_sequence() -> np.ndarray:
    """Sequência sintética (W, N_FEATURES) representando comportamento normal."""
    rng = np.random.default_rng(RANDOM_SEED)
    base = np.array(
        [-75.0, 2.5, 8.0, 1.2, 8.5, -0.5, 0.02, 0.5, 0.8, 0.05, 3.0],
        dtype=np.float32,
    )
    noise = rng.normal(0, 0.1, size=(WINDOW_SIZE, N_FEATURES)).astype(np.float32)
    return (base + noise).astype(np.float32)


@pytest.fixture
def sample_dataset() -> np.ndarray:
    """Dataset sintético (N, W, N_FEATURES) com dados normais para treino."""
    rng = np.random.default_rng(RANDOM_SEED)
    base = np.array(
        [-75.0, 2.5, 8.0, 1.2, 8.5, -0.5, 0.02, 0.5, 0.8, 0.05, 3.0],
        dtype=np.float32,
    )
    # (N_SEQUENCES, WINDOW_SIZE, N_FEATURES)
    data = base + rng.normal(0, 0.1, size=(N_SEQUENCES, WINDOW_SIZE, N_FEATURES))
    return data.astype(np.float32)


@pytest.fixture
def anomalous_sequence() -> np.ndarray:
    """Sequência sintética anômala (RSSI degradado, alta variância)."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    base = np.array(
        [-110.0, 15.0, -2.0, 8.0, 1.5, -25.0, 0.6, 12.0, 8.0, 0.9, 20.0],
        dtype=np.float32,
    )
    noise = rng.normal(0, 0.5, size=(WINDOW_SIZE, N_FEATURES)).astype(np.float32)
    return (base + noise).astype(np.float32)
