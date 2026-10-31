"""
Adiciona data_origin a telemetry e neighbor, cria simulation_run,
adiciona sim_run_id a anomaly e current_sim_run_id a node.
Idempotente — seguro para re-execução.
"""
from sqlalchemy import text


def _cols(conn, table: str) -> set:
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :table_name AND table_schema = current_schema()"
        ), {"table_name": table})
        return {r[0] for r in result.fetchall()}
    result = conn.execute(text(f"PRAGMA table_info({table})"))
    return {r[1] for r in result.fetchall()}


def _tables(conn) -> set:
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        ))
        return {r[0] for r in result.fetchall()}
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    return {r[0] for r in result.fetchall()}


def upgrade(engine) -> None:
    with engine.connect() as conn:
        dialect = engine.dialect.name
        existing_tables = _tables(conn)

        # 1. data_origin em telemetry
        if "data_origin" not in _cols(conn, "telemetry"):
            conn.execute(text(
                "ALTER TABLE telemetry ADD COLUMN data_origin VARCHAR(9) NOT NULL DEFAULT 'real'"
            ))

        # 2. data_origin em neighbor
        if "data_origin" not in _cols(conn, "neighbor"):
            conn.execute(text(
                "ALTER TABLE neighbor ADD COLUMN data_origin VARCHAR(9) NOT NULL DEFAULT 'real'"
            ))

        # 3. Criar tabela simulation_run
        if "simulation_run" not in existing_tables:
            if dialect == "sqlite":
                conn.execute(text("""
                    CREATE TABLE simulation_run (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        node_id INTEGER NOT NULL,
                        anomaly VARCHAR(3) NOT NULL,
                        started_at TIMESTAMP NOT NULL,
                        ended_at TIMESTAMP,
                        duration_planned INTEGER NOT NULL,
                        ended_reason VARCHAR(20),
                        telemetry_count INTEGER NOT NULL DEFAULT 0,
                        anomaly_onset_at TIMESTAMP,
                        detected_at TIMESTAMP,
                        detector_primary VARCHAR(30),
                        detector_secondary VARCHAR(30),
                        detector_combo VARCHAR(10)
                    )
                """))
            else:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS simulation_run (
                        id SERIAL PRIMARY KEY,
                        node_id INTEGER NOT NULL,
                        anomaly VARCHAR(3) NOT NULL,
                        started_at TIMESTAMP NOT NULL,
                        ended_at TIMESTAMP,
                        duration_planned INTEGER NOT NULL,
                        ended_reason VARCHAR(20),
                        telemetry_count INTEGER NOT NULL DEFAULT 0,
                        anomaly_onset_at TIMESTAMP,
                        detected_at TIMESTAMP,
                        detector_primary VARCHAR(30),
                        detector_secondary VARCHAR(30),
                        detector_combo VARCHAR(10)
                    )
                """))

        # 3b. onset real da anomalia (1ª telemetria sintética),
        #     para latência do detector separada da latência total.
        if "simulation_run" in _tables(conn) and \
                "anomaly_onset_at" not in _cols(conn, "simulation_run"):
            conn.execute(text(
                "ALTER TABLE simulation_run ADD COLUMN anomaly_onset_at TIMESTAMP"
            ))

        # 4. sim_run_id em anomaly
        if "sim_run_id" not in _cols(conn, "anomaly"):
            conn.execute(text(
                "ALTER TABLE anomaly ADD COLUMN sim_run_id INTEGER"
            ))

        # 5. current_sim_run_id em node
        if "current_sim_run_id" not in _cols(conn, "node"):
            conn.execute(text(
                "ALTER TABLE node ADD COLUMN current_sim_run_id INTEGER"
            ))

        conn.commit()
