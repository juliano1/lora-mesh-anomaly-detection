"""PostgreSQL: snapshot e igualdade das linhas com o SQLite."""
import os
import uuid
import zipfile
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, make_url, text
from sqlmodel import SQLModel

from tests.conftest import export_request

BASE_URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="defina TEST_POSTGRES_URL (PostgreSQL de teste)")
# Schema próprio e descartável: os testes nunca tocam no schema padrão do banco apontado.
SCHEMA = f"dataset_test_{uuid.uuid4().hex[:12]}"
PG_URL = (make_url(BASE_URL).update_query_dict({"options": f"-csearch_path={SCHEMA}"})
          .render_as_string(hide_password=False) if BASE_URL else None)
SETTINGS = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=0.0,
                           model_fields_set=set())
NOW = datetime(2026, 10, 8, 12, 0)


@pytest.fixture(scope="module")
def pg_loaded(export_source):
    import models  # noqa: F401
    from dataset_export.slice import EXPORT_TABLES, table, write_rows
    admin = create_engine(BASE_URL)
    with admin.begin() as c:
        c.execute(text(f'CREATE SCHEMA "{SCHEMA}"'))
    pg = create_engine(PG_URL)
    tables = [table(n) for n in EXPORT_TABLES]
    SQLModel.metadata.create_all(pg, tables=tables)
    src = create_engine(f"sqlite:///{export_source}")

    def rows(name):
        with src.connect() as c:
            return [dict(r) for r in c.execute(table(name).select()).mappings()]

    nodes = rows("node")
    write_rows(pg, "network", rows("network"))
    write_rows(pg, "node", [{**n, "current_sim_run_id": None} for n in nodes])
    write_rows(pg, "simulation_run", rows("simulation_run"))
    write_rows(pg, "telemetry", rows("telemetry"))
    write_rows(pg, "neighbor", rows("neighbor"))
    with pg.begin() as c:
        for n in nodes:
            if n["current_sim_run_id"] is not None:
                c.execute(text("UPDATE node SET current_sim_run_id=:s WHERE node_id=:n"),
                          {"s": n["current_sim_run_id"], "n": n["node_id"]})
    src.dispose()
    yield pg
    pg.dispose()
    with admin.begin() as c:
        c.execute(text(f'DROP SCHEMA "{SCHEMA}" CASCADE'))
    admin.dispose()


def _csvs(zip_path, root):
    from dataset_export.package import RAW_FILES
    with zipfile.ZipFile(zip_path) as z:
        return {name: z.read(f"{root}/{name}") for name in RAW_FILES}


def test_postgres_and_sqlite_packages_have_identical_rows(pg_loaded, export_source, tmp_path):
    from dataset_export.package import export_dataset
    from dataset_export.snapshot import build_source_engine
    lite = build_source_engine(f"sqlite:///{export_source}")
    pgsrc = build_source_engine(PG_URL)
    m1 = export_dataset(lite, export_request(), tmp_path / "lite.zip", settings=SETTINGS, now=NOW)
    m2 = export_dataset(pgsrc, export_request(), tmp_path / "pg.zip", settings=SETTINGS, now=NOW)
    assert m2["source"]["dialect"] == "postgresql"
    assert _csvs(tmp_path / "lite.zip", m1["package_root"]) == _csvs(tmp_path / "pg.zip", m2["package_root"])
    lite.dispose(); pgsrc.dispose()


def test_postgres_snapshot_holds_during_export_read(pg_loaded):
    from dataset_export.snapshot import build_source_engine, snapshot_session
    source = build_source_engine(PG_URL)
    with snapshot_session(source) as s:
        before = s.execute(text("SELECT COUNT(*) FROM neighbor")).scalar()
        with pg_loaded.begin() as c:
            # id explícito: os IDs da fixture foram inseridos sem avançar a sequência
            c.execute(text("INSERT INTO neighbor (id, node_id, neighbor_address, rssi, snr, "
                           "is_parent, timestamp, data_origin) VALUES (999999, 1, '0x0002', -80, "
                           "5.0, false, '2026-09-01 00:30:00', 'real')"))
        after = s.execute(text("SELECT COUNT(*) FROM neighbor")).scalar()
    assert before == after
    source.dispose()
