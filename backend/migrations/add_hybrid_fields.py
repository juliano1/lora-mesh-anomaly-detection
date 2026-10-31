"""
Migração — adiciona campos do detector híbrido IF + LSTM-AE.

Tabela mlsettings — campos adicionados:
    ml_use_hybrid       BOOLEAN  DEFAULT FALSE         — legado sem efeito
    ml_hybrid_mode      VARCHAR  DEFAULT 'if_only'     — legado sem efeito
    ml_lstm_window      INTEGER  DEFAULT 12            — W (sub-janelas por sequência)
    ml_lstm_alpha       FLOAT    DEFAULT 0.5           — legado sem efeito (peso da fusão em ml_online_fusion_alpha)
    ml_lstm_threshold_strategy VARCHAR DEFAULT 'percentile' — estratégia de limiar do LSTM-AE

Tabela anomaly — campo adicionado:
    detector_algorithm  VARCHAR(30) DEFAULT NULL
        Valores gravados por versões anteriores: if_only, lstm_only, if_shadow,
        lstm_shadow, hybrid_A, hybrid_B, hybrid_C, hybrid_D, if, if_only_fallback

Em bancos que têm ml_hybrid_mode como VARCHAR(1), amplia a coluna para VARCHAR(20).
PostgreSQL: ALTER TYPE via ALTER COLUMN. SQLite: tipagem dinâmica, sem ALTER.

Idempotente — seguro para re-execução.
"""

from sqlalchemy import text


def _existing_columns(conn, dialect: str, table_name: str) -> set:
    """Retorna set de nomes de colunas existentes (compatível com SQLite e PostgreSQL)."""
    if dialect == "postgresql":
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :table_name AND table_schema = current_schema()"
        ), {"table_name": table_name})
        return {r[0] for r in result.fetchall()}
    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
    return {r[1] for r in result.fetchall()}


def upgrade(engine):
    """Idempotente: adiciona colunas apenas se ainda não existirem."""
    dialect = engine.dialect.name  # 'sqlite' ou 'postgresql'

    with engine.connect() as conn:
        # ── mlsettings: campos do detector híbrido ─────────────────────────────
        ml_cols = _existing_columns(conn, dialect, "mlsettings")

        _ml_additions = [
            ("ml_use_hybrid",              "BOOLEAN DEFAULT FALSE"),
            ("ml_hybrid_mode",             "VARCHAR(20) DEFAULT 'if_only'"),
            ("ml_lstm_window",             "INTEGER DEFAULT 12"),
            ("ml_lstm_alpha",              "FLOAT DEFAULT 0.5"),
            ("ml_lstm_threshold_strategy", "VARCHAR(20) DEFAULT 'percentile'"),
        ]
        for col, col_def in _ml_additions:
            if col not in ml_cols:
                if dialect == "postgresql":
                    conn.execute(text(
                        f"ALTER TABLE mlsettings ADD COLUMN IF NOT EXISTS {col} {col_def}"
                    ))
                else:
                    conn.execute(text(
                        f"ALTER TABLE mlsettings ADD COLUMN {col} {col_def}"
                    ))

        # ── ml_hybrid_mode: ampliar de VARCHAR(1) para VARCHAR(20) ─────────────
        # SQLite não suporta ALTER COLUMN — tipagem dinâmica, o valor é armazenado
        # como TEXT independente da declaração, então não há truncamento.
        # PostgreSQL: alterar o tipo explicitamente.
        if dialect == "postgresql" and "ml_hybrid_mode" in ml_cols:
            conn.execute(text(
                "ALTER TABLE mlsettings "
                "ALTER COLUMN ml_hybrid_mode TYPE VARCHAR(20)"
            ))

        # ── anomaly: rastreabilidade do algoritmo detector ─────────────────────
        a_cols = _existing_columns(conn, dialect, "anomaly")
        if "detector_algorithm" not in a_cols:
            if dialect == "postgresql":
                conn.execute(text(
                    "ALTER TABLE anomaly "
                    "ADD COLUMN IF NOT EXISTS detector_algorithm VARCHAR(30) DEFAULT NULL"
                ))
            else:
                conn.execute(text(
                    "ALTER TABLE anomaly ADD COLUMN detector_algorithm VARCHAR(30) DEFAULT NULL"
                ))

        conn.commit()
