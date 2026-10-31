"""Testes das migrações da seleção online e da procedência do detector — idempotência e
criação de colunas."""
import pytest
from sqlalchemy import create_engine, text


def _existing_columns(conn, table: str) -> set:
    result = conn.execute(text(f"PRAGMA table_info({table})"))
    return {row[1] for row in result.fetchall()}


@pytest.fixture
def sqlite_engine():
    """Engine SQLite in-memory com tabelas mínimas para testar as migrações."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE mlsettings ("
            "  id INTEGER PRIMARY KEY,"
            "  ml_use_hybrid BOOLEAN DEFAULT 0,"
            "  ml_hybrid_mode VARCHAR(20) DEFAULT 'if_only',"
            "  ml_lstm_alpha FLOAT DEFAULT 0.5"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE anomaly ("
            "  id INTEGER PRIMARY KEY,"
            "  detector_algorithm VARCHAR(30)"
            ")"
        ))
        conn.commit()
    return engine


def test_add_online_selection_creates_columns(sqlite_engine):
    from migrations.add_online_selection import upgrade
    upgrade(sqlite_engine)
    with sqlite_engine.connect() as conn:
        cols = _existing_columns(conn, "mlsettings")
    assert "ml_online_primary" in cols
    assert "ml_online_secondary" in cols
    assert "ml_online_combo" in cols
    assert "ml_online_fusion_alpha" in cols


def test_add_online_selection_idempotent(sqlite_engine):
    from migrations.add_online_selection import upgrade
    upgrade(sqlite_engine)
    upgrade(sqlite_engine)  # segunda chamada não deve lançar exceção
    with sqlite_engine.connect() as conn:
        cols = _existing_columns(conn, "mlsettings")
    assert "ml_online_primary" in cols


@pytest.fixture
def sqlite_engine_with_anomalies():
    """Engine com tabela anomaly populada com valores legados."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE anomaly ("
            "  id INTEGER PRIMARY KEY,"
            "  detector_algorithm VARCHAR(30)"
            ")"
        ))
        rows = [
            ("if_only",), ("if",), ("lstm_only",),
            ("if_shadow",), ("lstm_shadow",),
            ("hybrid_A",), ("hybrid_B",),
            ("hybrid_C",), ("hybrid_D",),
            (None,),
        ]
        for (algo,) in rows:
            conn.execute(text("INSERT INTO anomaly (detector_algorithm) VALUES (:a)"), {"a": algo})
        conn.commit()
    return engine


def test_add_detector_procedence_creates_columns(sqlite_engine_with_anomalies):
    from migrations.add_detector_procedence import upgrade
    upgrade(sqlite_engine_with_anomalies)
    with sqlite_engine_with_anomalies.connect() as conn:
        cols = _existing_columns(conn, "anomaly")
    assert "detector_combo" in cols
    assert "detector_primary" in cols
    assert "detector_secondary" in cols


def test_backfill_maps_legacy_values(sqlite_engine_with_anomalies):
    from migrations.add_detector_procedence import upgrade
    upgrade(sqlite_engine_with_anomalies)
    with sqlite_engine_with_anomalies.connect() as conn:
        rows = conn.execute(text(
            "SELECT detector_algorithm, detector_combo, detector_primary, detector_secondary "
            "FROM anomaly"
        )).fetchall()
    by_algo = {r[0]: (r[1], r[2], r[3]) for r in rows}

    assert by_algo["if_only"] == ("single", "isolation_forest", None)
    assert by_algo["if"] == ("single", "isolation_forest", None)
    assert by_algo["lstm_only"] == ("single", "lstm_ae", None)
    assert by_algo["if_shadow"] == ("shadow", "isolation_forest", None)
    assert by_algo["lstm_shadow"] == ("shadow", "lstm_ae", None)
    assert by_algo["hybrid_A"] == ("cascade", "isolation_forest", "lstm_ae")
    assert by_algo["hybrid_B"] == ("fusion", "isolation_forest", "lstm_ae")
    # hybrid_C/D e NULL ficam com colunas novas em NULL (catch-all)
    assert by_algo["hybrid_C"] == (None, None, None)
    assert by_algo["hybrid_D"] == (None, None, None)
    assert by_algo[None] == (None, None, None)


def test_add_detector_procedence_idempotent(sqlite_engine_with_anomalies):
    from migrations.add_detector_procedence import upgrade
    upgrade(sqlite_engine_with_anomalies)
    upgrade(sqlite_engine_with_anomalies)  # não deve lançar
    with sqlite_engine_with_anomalies.connect() as conn:
        cols = _existing_columns(conn, "anomaly")
    assert "detector_combo" in cols
