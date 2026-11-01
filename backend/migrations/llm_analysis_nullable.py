"""
Legado (tabela sem uso nesta versão): torna verdict/confidence_llm/severity_llm/reasoning
NULLABLE em llm_analysis, para permitir registros pending/processing/error sem resultado.

PostgreSQL: ALTER COLUMN ... DROP NOT NULL.
SQLite: não suporta DROP NOT NULL -> recria a tabela preservando dados.

Idempotente:
- Postgres: DROP NOT NULL é no-op se já nullable.
- SQLite: detecta se 'verdict' ainda é NOT NULL via PRAGMA antes de recriar e retoma a
  cópia se uma execução anterior parou com a tabela antiga renomeada.
"""
import re

from sqlalchemy import text
from sqlalchemy.schema import DefaultClause

NULLABLE_COLS = ["verdict", "confidence_llm", "severity_llm", "reasoning"]


def _verdict_is_notnull_sqlite(conn) -> bool:
    rows = conn.execute(text("PRAGMA table_info(llm_analysis)")).fetchall()
    for r in rows:
        # PRAGMA: (cid, name, type, notnull, dflt_value, pk)
        if r[1] == "verdict":
            return bool(r[3])
    return False


def _columns_in_order(conn, table: str):
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return [r[1] for r in rows]


def _sqlite_server_defaults(conn, table: str) -> dict:
    """Mapa coluna -> expressão DEFAULT (dflt_value) existente na tabela SQLite."""
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    out = {}
    for r in rows:
        # PRAGMA: (cid, name, type, notnull, dflt_value, pk)
        if r[4] is not None:
            out[r[1]] = r[4]
    return out


OLD = "llm_analysis_backup"


def _tables(conn) -> set:
    return {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}


def _backup_table(tables: set[str]) -> str:
    """Encontra uma cópia interrompida, inclusive com o antigo sufixo numerado."""
    candidates = sorted(t for t in tables if t == OLD or re.fullmatch(
        r"llm_analysis_old_[a-z]+[0-9]{2,3}", t
    ))
    if len(candidates) > 1:
        raise RuntimeError("Mais de uma cópia de llm_analysis: resolva manualmente antes de migrar")
    return candidates[0] if candidates else OLD


def upgrade(engine) -> None:
    dialect = engine.dialect.name
    if dialect == "postgresql":
        with engine.connect() as conn:
            for col in NULLABLE_COLS:
                conn.execute(text(
                    f"ALTER TABLE llm_analysis ALTER COLUMN {col} DROP NOT NULL"
                ))
            conn.commit()
        return

    # SQLite: recriação de tabela
    with engine.connect() as conn:
        tables = _tables(conn)
        old_table = _backup_table(tables)
        if old_table in tables:
            # Execução anterior interrompida depois do RENAME: retoma a partir da antiga.
            existing_defaults = _sqlite_server_defaults(conn, old_table)
            if "llm_analysis" in tables:
                if conn.execute(text("SELECT COUNT(*) FROM llm_analysis")).scalar():
                    raise RuntimeError(f"{old_table} e llm_analysis têm dados: resolva manualmente")
                conn.execute(text("DROP TABLE llm_analysis"))
                conn.commit()
        else:
            if not _verdict_is_notnull_sqlite(conn):
                return  # já convertida — idempotente
            # Capturamos os DEFAULTs reais da tabela atual antes de renomeá-la, para
            # preservá-los na tabela recriada (ver explicação abaixo).
            existing_defaults = _sqlite_server_defaults(conn, "llm_analysis")
            # Os índices acompanham o RENAME e colidiriam com os da tabela nova.
            for (name,) in conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND tbl_name='llm_analysis' AND sql IS NOT NULL")).fetchall():
                conn.execute(text(f'DROP INDEX "{name}"'))
            # Sem o modo legado, o RENAME reescreveria as FKs de llm_human_review para a
            # tabela antiga, que é apagada no fim.
            conn.execute(text("PRAGMA legacy_alter_table=ON"))
            conn.execute(text(f"ALTER TABLE llm_analysis RENAME TO {old_table}"))
            conn.execute(text("PRAGMA legacy_alter_table=OFF"))
            conn.commit()

    # Recriar a tabela nova a partir do modelo (já com nullable correto nos 4 campos).
    #
    # Cuidado importante: o SQLModel usa defaults Python-side (default=/default_factory),
    # que NÃO viram cláusula DEFAULT no DDL. Recriar puramente a partir do modelo
    # perderia os DEFAULTs de nível de banco que a tabela real possui (ex.: telegram_sent,
    # status, attempt_count, lease_recovery_count criados pela migração aditiva). Isso
    # quebraria inserts feitos fora do ORM (jobs pending/error) que dependem desses
    # defaults. Por isso reaplicamos no modelo, apenas durante a criação desta tabela,
    # os DEFAULTs que existiam na tabela original, restaurando o estado do modelo depois.
    #
    # Caso especial: timestamp é NOT NULL via default_factory e não tem DEFAULT no banco;
    # garantimos um DEFAULT CURRENT_TIMESTAMP para que inserts sem timestamp não violem
    # o NOT NULL.
    from sqlalchemy import text as _sa_text
    from models import LLMAnalysis  # noqa: F401

    desired_defaults = dict(existing_defaults)
    desired_defaults.setdefault("timestamp", "CURRENT_TIMESTAMP")

    table = LLMAnalysis.__table__
    saved = {}
    for col_name, default_sql in desired_defaults.items():
        col = table.c.get(col_name)
        if col is None:
            continue
        saved[col_name] = col.server_default
        col.server_default = DefaultClause(_sa_text(str(default_sql)))
    try:
        table.create(bind=engine)
    finally:
        for col_name, prev in saved.items():
            table.c[col_name].server_default = prev

    with engine.connect() as conn:
        old_cols = _columns_in_order(conn, old_table)
        new_cols = _columns_in_order(conn, "llm_analysis")
        shared = [c for c in old_cols if c in new_cols]
        # Preserva o timestamp original quando existir; caso contrário (linha antiga
        # sem timestamp) usa CURRENT_TIMESTAMP para não violar o NOT NULL.
        select_cols = ", ".join(
            "COALESCE(timestamp, CURRENT_TIMESTAMP)" if c == "timestamp" else c
            for c in shared
        )
        shared_list = ", ".join(shared)
        conn.execute(text(
            f"INSERT INTO llm_analysis ({shared_list}) "
            f"SELECT {select_cols} FROM {old_table}"
        ))
        conn.execute(text(f"DROP TABLE {old_table}"))
        conn.commit()
