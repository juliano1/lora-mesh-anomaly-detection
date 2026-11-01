"""
Adiciona coluna is_admin em user + bootstrap do administrador inicial.

Idempotente — seguro para re-execução.
Bootstrap: só quando a coluna é criada, promove as contas de bootstrap (admin100 e
admin200) a is_admin=true. Depois disso, permissões alteradas no banco são preservadas.
"""
from sqlalchemy import text


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
    dialect = engine.dialect.name
    with engine.connect() as conn:
        cols = _existing_columns(conn, dialect, "user")
        if "is_admin" not in cols:
            if dialect == "postgresql":
                conn.execute(text(
                    'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE'
                ))
            else:
                conn.execute(text(
                    "ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0"
                ))
            table = '"user"' if dialect == "postgresql" else "user"
            conn.execute(
                text(f"UPDATE {table} SET is_admin = :admin WHERE username IN ('admin100', 'admin200')"),
                {"admin": True},
            )
        conn.commit()
