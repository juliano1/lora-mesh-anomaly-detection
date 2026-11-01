"""Testes das migrações de administrador, campos legados de LLM e backfill — idempotência,
colunas, nullable e backfill."""
import pytest
from sqlalchemy import create_engine, text


def _cols(conn, table: str) -> set:
    result = conn.execute(text(f"PRAGMA table_info({table})"))
    return {row[1] for row in result.fetchall()}


@pytest.fixture
def engine():
    """SQLite in-memory com as tabelas mínimas anteriores a essas migrações."""
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        conn.execute(text(
            "CREATE TABLE user ("
            "  id INTEGER PRIMARY KEY,"
            "  username VARCHAR(50),"
            "  password_hash VARCHAR(128),"
            "  is_active BOOLEAN DEFAULT 1,"
            "  created_at DATETIME"
            ")"
        ))
        conn.execute(text("INSERT INTO user (id, username) VALUES (1, 'admin100')"))
        conn.execute(text("INSERT INTO user (id, username) VALUES (2, 'operador')"))
        conn.commit()
    return eng


def test_add_user_is_admin_creates_column(engine):
    from migrations.add_user_is_admin import upgrade
    upgrade(engine)
    with engine.connect() as conn:
        assert "is_admin" in _cols(conn, "user")


def test_add_user_is_admin_bootstraps_admin_user(engine):
    from migrations.add_user_is_admin import upgrade
    upgrade(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT is_admin FROM user WHERE username='admin100'")
        ).fetchone()
        assert row[0] == 1


def test_add_user_is_admin_preserves_later_demotion(engine):
    from migrations.add_user_is_admin import upgrade
    upgrade(engine)
    with engine.connect() as conn:
        conn.execute(text("UPDATE user SET is_admin = 0 WHERE username='admin100'"))
        conn.commit()
    upgrade(engine)  # nova inicialização não devolve a permissão
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT is_admin FROM user WHERE username='admin100'")
        ).fetchone()
        assert row[0] == 0


def test_add_user_is_admin_is_idempotent(engine):
    from migrations.add_user_is_admin import upgrade
    upgrade(engine)
    upgrade(engine)  # segunda execução não pode lançar
    with engine.connect() as conn:
        assert "is_admin" in _cols(conn, "user")


@pytest.fixture
def mlsettings_engine():
    """SQLite in-memory com mlsettings anterior a essas migrações (só os campos LLM antigos)."""
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        conn.execute(text(
            "CREATE TABLE mlsettings ("
            "  id INTEGER PRIMARY KEY,"
            "  llm_enabled BOOLEAN DEFAULT 1,"
            "  llm_provider VARCHAR DEFAULT 'gemini',"
            "  llm_model VARCHAR DEFAULT 'gemini-2.0-flash'"
            ")"
        ))
        conn.execute(text("INSERT INTO mlsettings (id) VALUES (1)"))
        conn.commit()
    return eng


def test_add_mlsettings_llm_fields(mlsettings_engine):
    from migrations.add_mlsettings_llm_fields import upgrade
    upgrade(mlsettings_engine)
    with mlsettings_engine.connect() as conn:
        cols = _cols(conn, "mlsettings")
    for c in ("llm_base_url", "llm_temperature", "llm_max_tokens",
              "llm_num_ctx", "llm_timeout_seconds", "llm_max_retries"):
        assert c in cols


def test_add_mlsettings_llm_fields_idempotent(mlsettings_engine):
    from migrations.add_mlsettings_llm_fields import upgrade
    upgrade(mlsettings_engine)
    upgrade(mlsettings_engine)
    with mlsettings_engine.connect() as conn:
        assert "llm_base_url" in _cols(conn, "mlsettings")


@pytest.fixture
def llm_analysis_engine():
    """SQLite in-memory com llm_analysis no schema antigo (resultado NOT NULL)."""
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        conn.execute(text(
            "CREATE TABLE llm_analysis ("
            "  id INTEGER PRIMARY KEY,"
            "  anomaly_id INTEGER,"
            "  node_id INTEGER,"
            "  anomaly_type VARCHAR,"
            "  timestamp DATETIME,"
            "  verdict VARCHAR NOT NULL,"
            "  confidence_llm FLOAT NOT NULL,"
            "  confidence_ml FLOAT NOT NULL,"
            "  severity_llm VARCHAR NOT NULL,"
            "  reasoning VARCHAR NOT NULL,"
            "  action VARCHAR,"
            "  model_used VARCHAR NOT NULL,"
            "  features_json VARCHAR,"
            "  human_verdict VARCHAR,"
            "  human_notes VARCHAR,"
            "  telegram_sent BOOLEAN DEFAULT 0"
            ")"
        ))
        conn.commit()
    return eng


EXPECTED_NEW_LLM_ANALYSIS_COLS = [
    "network_id", "status", "system_prompt", "user_prompt", "raw_response",
    "parse_error", "provider", "mode", "temperature", "max_tokens", "num_ctx",
    "top_p", "seed", "llm_timeout_seconds", "llm_max_retries", "prompt_version",
    "context_reference_time", "started_at", "finished_at", "latency_ms",
    "input_tokens", "output_tokens", "attempt_count", "lease_recovery_count",
    "telegram_status", "telegram_sent_at", "telegram_error",
    "enqueued_at", "claimed_at", "worker_id", "lease_expires_at",
]


def test_add_llm_analysis_fields(llm_analysis_engine):
    from migrations.add_llm_analysis_fields import upgrade
    upgrade(llm_analysis_engine)
    with llm_analysis_engine.connect() as conn:
        cols = _cols(conn, "llm_analysis")
    for c in EXPECTED_NEW_LLM_ANALYSIS_COLS:
        assert c in cols, f"coluna {c} não criada"


def test_add_llm_analysis_fields_idempotent(llm_analysis_engine):
    from migrations.add_llm_analysis_fields import upgrade
    upgrade(llm_analysis_engine)
    upgrade(llm_analysis_engine)
    with llm_analysis_engine.connect() as conn:
        assert "status" in _cols(conn, "llm_analysis")


def test_llm_analysis_nullable_allows_pending_rows(llm_analysis_engine):
    from migrations.add_llm_analysis_fields import upgrade as add_fields
    from migrations.llm_analysis_nullable import upgrade as make_nullable
    add_fields(llm_analysis_engine)
    make_nullable(llm_analysis_engine)
    with llm_analysis_engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO llm_analysis "
            "(anomaly_id, node_id, anomaly_type, confidence_ml, model_used, status, verdict) "
            "VALUES (1, 1, 'A3', 0.9, 'qwen3:4b', 'pending', NULL)"
        ))
        conn.commit()
        row = conn.execute(text(
            "SELECT status, verdict FROM llm_analysis WHERE anomaly_id=1"
        )).fetchone()
    assert row[0] == "pending"
    assert row[1] is None


def test_llm_analysis_nullable_preserves_existing_rows(llm_analysis_engine):
    from migrations.add_llm_analysis_fields import upgrade as add_fields
    from migrations.llm_analysis_nullable import upgrade as make_nullable
    with llm_analysis_engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO llm_analysis "
            "(id, anomaly_id, node_id, anomaly_type, verdict, confidence_llm, "
            " confidence_ml, severity_llm, reasoning, model_used) "
            "VALUES (99, 5, 7, 'A1', 'REAL', 0.8, 0.7, 'high', 'porque sim', 'gemini-2.0-flash')"
        ))
        conn.commit()
    add_fields(llm_analysis_engine)
    make_nullable(llm_analysis_engine)
    with llm_analysis_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT verdict, reasoning FROM llm_analysis WHERE id=99"
        )).fetchone()
    assert row[0] == "REAL"
    assert row[1] == "porque sim"


def test_llm_analysis_nullable_idempotent(llm_analysis_engine):
    from migrations.add_llm_analysis_fields import upgrade as add_fields
    from migrations.llm_analysis_nullable import upgrade as make_nullable
    add_fields(llm_analysis_engine)
    make_nullable(llm_analysis_engine)
    make_nullable(llm_analysis_engine)  # não pode quebrar nem duplicar
    with llm_analysis_engine.connect() as conn:
        assert "verdict" in _cols(conn, "llm_analysis")


def test_llm_analysis_nullable_with_existing_index_and_fk(llm_analysis_engine):
    """Banco real tem índice ix_llm_analysis_* e llm_human_review apontando para a tabela:
    a recriação não pode falhar por índice duplicado nem deixar a FK numa tabela apagada."""
    from migrations.add_llm_analysis_fields import upgrade as add_fields
    from migrations.llm_analysis_nullable import upgrade as make_nullable
    with llm_analysis_engine.connect() as conn:
        conn.execute(text("CREATE INDEX ix_llm_analysis_anomaly_id ON llm_analysis(anomaly_id)"))
        conn.execute(text(
            "INSERT INTO llm_analysis (id, anomaly_id, node_id, anomaly_type, verdict, confidence_llm,"
            " confidence_ml, severity_llm, reasoning, model_used)"
            " VALUES (1, 1, 1, 'A1', 'REAL', 0.5, 0.5, 'low', 'x', 'm')"))
        conn.execute(text("CREATE TABLE llm_human_review (id INTEGER PRIMARY KEY,"
                          " analysis_id INTEGER REFERENCES llm_analysis(id))"))
        conn.commit()
    add_fields(llm_analysis_engine)
    make_nullable(llm_analysis_engine)
    make_nullable(llm_analysis_engine)  # segunda inicialização
    with llm_analysis_engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        review_sql = conn.execute(text(
            "SELECT sql FROM sqlite_master WHERE name='llm_human_review'")).scalar()
        notnull = {r[1]: r[3] for r in conn.execute(text("PRAGMA table_info(llm_analysis)"))}
        verdict = conn.execute(text("SELECT verdict FROM llm_analysis WHERE id=1")).scalar()
    assert "llm_analysis_backup" not in tables
    assert verdict == "REAL"
    assert notnull["verdict"] == 0
    assert "llm_analysis_backup" not in review_sql


@pytest.mark.parametrize("backup_name", ["llm_analysis_backup", "llm_analysis_old_legacy056"])
def test_llm_analysis_nullable_resumes_after_interrupted_rename(llm_analysis_engine, backup_name):
    """Se a execução anterior parou depois do RENAME, a seguinte recria e copia os dados."""
    from migrations.add_llm_analysis_fields import upgrade as add_fields
    from migrations.llm_analysis_nullable import upgrade as make_nullable
    add_fields(llm_analysis_engine)
    with llm_analysis_engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO llm_analysis (id, anomaly_id, node_id, anomaly_type, verdict, confidence_llm,"
            " confidence_ml, severity_llm, reasoning, model_used)"
            " VALUES (7, 1, 1, 'A1', 'REAL', 0.5, 0.5, 'low', 'x', 'm')"))
        conn.execute(text(f"ALTER TABLE llm_analysis RENAME TO {backup_name}"))
        conn.commit()
    make_nullable(llm_analysis_engine)
    with llm_analysis_engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        assert conn.execute(text("SELECT verdict FROM llm_analysis WHERE id=7")).scalar() == "REAL"
    assert backup_name not in tables


def test_new_tables_created_from_models():
    from sqlmodel import SQLModel, create_engine as sm_create_engine
    import models  # noqa: F401 — registra as classes no metadata
    eng = sm_create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    with eng.connect() as conn:
        attempt_cols = _cols(conn, "llm_inference_attempt")
        review_cols = _cols(conn, "llm_human_review")
    for c in ("id", "analysis_id", "attempt_number", "provider", "model",
              "model_digest", "started_at", "finished_at", "latency_ms",
              "status", "input_tokens", "output_tokens", "raw_response",
              "provider_metadata_json", "error"):
        assert c in attempt_cols, f"attempt: falta {c}"
    for c in ("id", "analysis_id", "reviewer_user_id", "human_verdict",
              "notes", "created_at", "supersedes_review_id", "source"):
        assert c in review_cols, f"review: falta {c}"


def _full_engine_with_legacy_rows():
    from sqlmodel import SQLModel, create_engine as sm_create_engine
    import models  # noqa
    eng = sm_create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    with eng.connect() as conn:
        conn.execute(text(
            "INSERT INTO node (node_id, network_id, name, status, last_seen, "
            " is_gateway, sim_mode) "
            "VALUES (7, 42, 'no7', 'online', CURRENT_TIMESTAMP, 0, 0)"
        ))
        conn.execute(text(
            "INSERT INTO llm_analysis "
            "(id, anomaly_id, node_id, anomaly_type, timestamp, verdict, confidence_llm, "
            " confidence_ml, severity_llm, reasoning, model_used, status, "
            " attempt_count, lease_recovery_count, "
            " human_verdict, human_notes, telegram_sent) "
            "VALUES (1, 10, 7, 'A1', CURRENT_TIMESTAMP, 'FALSO_POSITIVO', 0.9, 0.8, 'low', "
            " 'ruido', 'gemini-2.0-flash', 'success', 0, 0, "
            " 'REAL', 'na verdade era real', 1)"
        ))
        conn.execute(text(
            "INSERT INTO llm_analysis "
            "(id, anomaly_id, node_id, anomaly_type, timestamp, verdict, confidence_llm, "
            " confidence_ml, severity_llm, reasoning, model_used, status, "
            " attempt_count, lease_recovery_count, telegram_sent) "
            "VALUES (2, 11, 7, 'A2', CURRENT_TIMESTAMP, 'REAL', 0.7, 0.6, 'high', 'perda', "
            " 'gemini-2.0-flash', 'success', 0, 0, 0)"
        ))
        conn.commit()
    return eng


def test_backfill_network_id():
    eng = _full_engine_with_legacy_rows()
    from migrations.backfill_legacy import upgrade
    upgrade(eng)
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT network_id FROM llm_analysis WHERE node_id=7"
        )).fetchall()
    assert all(r[0] == 42 for r in rows)


def test_backfill_telegram_status():
    eng = _full_engine_with_legacy_rows()
    from migrations.backfill_legacy import upgrade
    upgrade(eng)
    with eng.connect() as conn:
        s1 = conn.execute(text("SELECT telegram_status FROM llm_analysis WHERE id=1")).fetchone()[0]
        s2 = conn.execute(text("SELECT telegram_status FROM llm_analysis WHERE id=2")).fetchone()[0]
    assert s1 == "sent"
    assert s2 == "legacy_unknown"


def test_backfill_legacy_human_review():
    eng = _full_engine_with_legacy_rows()
    from migrations.backfill_legacy import upgrade
    upgrade(eng)
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT analysis_id, human_verdict, source, reviewer_user_id "
            "FROM llm_human_review"
        )).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] == "REAL"
    assert rows[0][2] == "legacy"
    assert rows[0][3] is None


def test_backfill_idempotent():
    eng = _full_engine_with_legacy_rows()
    from migrations.backfill_legacy import upgrade
    upgrade(eng)
    upgrade(eng)  # segunda execução não pode duplicar reviews
    with eng.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM llm_human_review")).fetchone()[0]
    assert n == 1
