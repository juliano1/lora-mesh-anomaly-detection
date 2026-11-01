"""
Legado (tabelas sem uso nesta versão): backfill de llm_analysis + llm_human_review.

- network_id    <- node.network_id (via node_id)
- status        <- 'success' onde verdict não nulo e status nulo
- telegram_status <- 'sent' (telegram_sent=1) | 'legacy_unknown' (0/NULL)
- llm_human_review: cria review source='legacy' para human_verdict preexistente
  sem review correspondente.

Idempotente.
"""
from sqlalchemy import text


def upgrade(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text(
            "UPDATE llm_analysis "
            "SET network_id = ("
            "  SELECT n.network_id FROM node n WHERE n.node_id = llm_analysis.node_id"
            ") "
            "WHERE network_id IS NULL"
        ))
        conn.execute(text(
            "UPDATE llm_analysis SET status = 'success' "
            "WHERE status IS NULL AND verdict IS NOT NULL"
        ))
        # telegram_sent é BOOLEAN: no Postgres `= 1/0` falha (boolean = integer).
        # TRUE/FALSE são literais portáveis (SQLite mapeia p/ 1/0).
        conn.execute(text(
            "UPDATE llm_analysis SET telegram_status = 'sent' "
            "WHERE telegram_status IS NULL AND telegram_sent = TRUE"
        ))
        conn.execute(text(
            "UPDATE llm_analysis SET telegram_status = 'legacy_unknown' "
            "WHERE telegram_status IS NULL AND (telegram_sent = FALSE OR telegram_sent IS NULL)"
        ))
        conn.execute(text(
            "INSERT INTO llm_human_review "
            "(analysis_id, reviewer_user_id, human_verdict, notes, created_at, "
            " supersedes_review_id, source) "
            "SELECT a.id, NULL, a.human_verdict, a.human_notes, "
            "       COALESCE(a.timestamp, CURRENT_TIMESTAMP), NULL, 'legacy' "
            "FROM llm_analysis a "
            "WHERE a.human_verdict IS NOT NULL "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM llm_human_review r WHERE r.analysis_id = a.id"
            ")"
        ))
        conn.commit()
