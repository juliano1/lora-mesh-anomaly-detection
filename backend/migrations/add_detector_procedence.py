"""
Migração — adiciona colunas estruturadas de procedência em anomaly + backfill.

Novas colunas em anomaly:
    detector_combo     VARCHAR(10)  DEFAULT NULL  — single|cascade|shadow|fusion
    detector_primary   VARCHAR(30)  DEFAULT NULL  — id do algoritmo que produziu a anomalia
    detector_secondary VARCHAR(30)  DEFAULT NULL  — id do parceiro; NULL quando single/shadow

Ajuste:
    detector_algorithm  max_length implícito sobe para 40 (SQLite: sem ALTER; PostgreSQL: ALTER TYPE).

Backfill: preenche as três colunas novas a partir do detector_algorithm legado segundo tabela
    do detector. hybrid_C/D e valores não mapeados ficam com colunas novas em NULL
    (catch-all) para preservar auditoria sem forçar classificação artificial.

Idempotente — seguro para re-execução.
"""
from sqlalchemy import text

_BACKFILL_MAP = {
    "if_only":     ("single",  "isolation_forest", None),
    "if":          ("single",  "isolation_forest", None),
    "if_only_fallback": ("single", "isolation_forest", None),
    "lstm_only":   ("single",  "lstm_ae",           None),
    "if_shadow":   ("shadow",  "isolation_forest",  None),
    "lstm_shadow": ("shadow",  "lstm_ae",           None),
    "hybrid_A":    ("cascade", "isolation_forest",  "lstm_ae"),
    "hybrid_B":    ("fusion",  "isolation_forest",  "lstm_ae"),
    # hybrid_C, hybrid_D, shadow (cru), etc. → catch-all NULL (não mapeados)
}


def _existing_columns(conn, dialect: str, table_name: str) -> set:
    if dialect == "postgresql":
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :table_name AND table_schema = current_schema()"
        ), {"table_name": table_name})
        return {r[0] for r in result.fetchall()}
    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
    return {r[1] for r in result.fetchall()}


def upgrade(engine) -> None:
    """Idempotente: adiciona colunas e faz backfill apenas se necessário."""
    dialect = engine.dialect.name
    with engine.connect() as conn:
        a_cols = _existing_columns(conn, dialect, "anomaly")

        # 1. Adicionar colunas estruturadas
        new_cols = [
            ("detector_combo",     "VARCHAR(10) DEFAULT NULL"),
            ("detector_primary",   "VARCHAR(30) DEFAULT NULL"),
            ("detector_secondary", "VARCHAR(30) DEFAULT NULL"),
        ]
        for col, col_def in new_cols:
            if col not in a_cols:
                if dialect == "postgresql":
                    conn.execute(text(
                        f"ALTER TABLE anomaly ADD COLUMN IF NOT EXISTS {col} {col_def}"
                    ))
                else:
                    conn.execute(text(
                        f"ALTER TABLE anomaly ADD COLUMN {col} {col_def}"
                    ))

        # 2. Ajuste de tamanho de detector_algorithm (PostgreSQL apenas; SQLite é dinâmico)
        if dialect == "postgresql" and "detector_algorithm" in a_cols:
            conn.execute(text(
                "ALTER TABLE anomaly ALTER COLUMN detector_algorithm TYPE VARCHAR(40)"
            ))

        conn.commit()

        # 3. Backfill: só preenche linhas onde as colunas novas ainda são NULL
        rows = conn.execute(text(
            "SELECT id, detector_algorithm FROM anomaly "
            "WHERE detector_combo IS NULL AND detector_primary IS NULL"
        )).fetchall()

        for row_id, algo in rows:
            mapping = _BACKFILL_MAP.get(algo)
            if mapping is None:
                continue
            combo, primary, secondary = mapping
            conn.execute(text(
                "UPDATE anomaly "
                "SET detector_combo = :combo, detector_primary = :primary, detector_secondary = :secondary "
                "WHERE id = :id"
            ), {"combo": combo, "primary": primary, "secondary": secondary, "id": row_id})

        conn.commit()
