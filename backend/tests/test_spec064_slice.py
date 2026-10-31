"""recorte, contexto e fecho referencial."""
from datetime import timedelta

import pytest
from sqlalchemy import text

from tests.conftest import EXPORT_T0, export_request


@pytest.fixture
def materialized(export_source, tmp_path):
    from dataset_export.slice import create_target, materialize_slice, finalize_sqlite
    from dataset_export.snapshot import build_source_engine, snapshot_session
    req = export_request()
    source = build_source_engine(f"sqlite:///{export_source}")
    target = create_target(tmp_path / "slice.db")
    with snapshot_session(source) as session:
        summary = materialize_slice(session, req, target, margin_minutes=0.0)
    violations = finalize_sqlite(target)
    yield req, summary, target, violations
    source.dispose(); target.dispose()


def _q(engine, sql):
    with engine.connect() as c:
        return c.execute(text(sql)).fetchall()


def test_nodes_networks_and_context(materialized):
    req, summary, target, violations = materialized
    assert summary.requested_node_ids == [1, 2, 4]
    assert summary.context_node_ids == [3]
    assert summary.network_ids == [1, 2]
    assert violations == []


def test_simulations_by_period_and_closure(materialized):
    _, summary, target, _ = materialized
    assert summary.simulation_run_ids == [10, 11, 12]   # 13 é distante; 10 entra pelo fecho


def test_time_bounds_and_first_window_context(materialized):
    req, summary, target, _ = materialized
    lo, hi = req.data_start(), req.end
    for table in ("telemetry", "neighbor"):
        mn, mx = _q(target, f"SELECT MIN(timestamp), MAX(timestamp) FROM {table}")[0]
        assert mn is not None
    rows = _q(target, "SELECT timestamp FROM telemetry")
    from datetime import datetime
    stamps = [datetime.fromisoformat(r[0]) for r in rows]
    assert min(stamps) > lo and max(stamps) <= hi
    assert any(t <= EXPORT_T0 for t in stamps)          # contexto anterior ao início
    assert {r[0] for r in _q(target, "SELECT DISTINCT node_id FROM telemetry")} == {1, 2}


def test_reverse_links_only_about_requested_nodes(materialized):
    _, summary, target, _ = materialized
    rows = _q(target, "SELECT DISTINCT neighbor_address FROM neighbor WHERE node_id = 3")
    assert rows == [("0x0001",)]
    assert summary.rows["neighbor"] == _q(target, "SELECT COUNT(*) FROM neighbor")[0][0]


def test_values_and_ids_preserved(materialized, export_source):
    from sqlalchemy import create_engine
    _, _, target, _ = materialized
    src = create_engine(f"sqlite:///{export_source}")
    got = _q(target, "SELECT * FROM telemetry ORDER BY id LIMIT 5")
    ids = ",".join(str(r[0]) for r in got)
    want = _q(src, f"SELECT * FROM telemetry WHERE id IN ({ids}) ORDER BY id")
    assert got == want
    src.dispose()


def test_unknown_node_and_empty_network_rejected(export_source, tmp_path):
    from dataset_export.slice import ExportError, create_target, materialize_slice
    from dataset_export.snapshot import build_source_engine, snapshot_session
    source = build_source_engine(f"sqlite:///{export_source}")
    for i, req in enumerate((export_request(node_ids=(1, 3)), export_request(network_id=99, node_ids=None))):
        target = create_target(tmp_path / f"s{i}.db")
        with snapshot_session(source) as session, pytest.raises(ExportError):
            materialize_slice(session, req, target, margin_minutes=0.0)
        target.dispose()
    source.dispose()


def test_target_must_not_exist(tmp_path):
    from dataset_export.slice import ExportError, create_target
    (tmp_path / "x.db").write_bytes(b"")
    with pytest.raises(ExportError):
        create_target(tmp_path / "x.db")


def test_foreign_key_check_detects_dangling_reference(tmp_path):
    from dataset_export.slice import create_target, finalize_sqlite, write_rows
    target = create_target(tmp_path / "f.db")
    write_rows(target, "telemetry", [{
        "id": 1, "node_id": 999, "timestamp": EXPORT_T0, "rssi": -80, "snr": 1.0,
        "packet_id": None, "parent_id": None, "uptime": None, "sequence": None,
        "battery_level": None, "temperature": None, "humidity": None,
        "payload_raw": "{}", "data_origin": "real"}])
    assert finalize_sqlite(target)          # violação apontada
    target.dispose()
