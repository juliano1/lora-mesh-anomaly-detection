"""Pacotes features e full."""
import io
import json
import zipfile
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session

from tests.conftest import EXPORT_T0, export_request

SETTINGS = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE="2026-01-01T00:00:00Z",
                           ML_REFERENCE_MARGIN_MINUTES=0.0, model_fields_set=set())
NOW = datetime(2026, 10, 8, 12, 0)


def _export(source, tmp, name, **over):
    from dataset_export.package import export_dataset
    from dataset_export.snapshot import build_source_engine
    eng = build_source_engine(f"sqlite:///{source}")
    m = export_dataset(eng, export_request(**over), tmp / name, settings=SETTINGS, now=NOW)
    eng.dispose()
    return m


def _rows(zpath, root, name):
    from dataset_export.codec import read_csv
    with zipfile.ZipFile(zpath) as z, z.open(f"{root}/{name}") as raw:
        it = read_csv(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        header = next(it)
        return [dict(zip(header, r)) for r in it]


def test_features_kind_files_and_manifest(export_source, tmp_path):
    from dataset_export.package import verify_package
    m = _export(export_source, tmp_path, "f.zip", export_kind="features")
    verify_package(tmp_path / "f.zip")
    assert set(m["files"]) == {"features.csv", "coverage.csv", "schema.json", "README.md", "events_template.csv"}
    feats = m["features"]
    assert feats["rows"] == 3 * m["resolved"]["grid_instants"] and feats["complete"] is True
    assert sum(feats["status_counts"].values()) == feats["rows"]
    assert feats["feature_names"][0] == "rssi_mean" and len(feats["feature_names"]) == 11
    assert m["tables"] == {} and m["slice_rows"]["telemetry"] > 0


def test_full_kind_has_raw_and_features_and_imports(export_source, tmp_path):
    from dataset_export.importer import import_dataset
    m = _export(export_source, tmp_path, "full.zip", export_kind="full")
    assert {"telemetry.csv", "features.csv", "coverage.csv"} <= set(m["files"])
    import_dataset(tmp_path / "full.zip", tmp_path / "full.db")


def test_features_csv_equals_direct_extraction_on_source(export_source, tmp_path):
    """Features do recorte = extrator direto na origem."""
    from ai.feature_engineering import FeatureExtractor
    from ai.sequence_builder import FEATURE_NAMES
    from dataset_export.request import parse_utc
    m = _export(export_source, tmp_path, "c18.zip", export_kind="features")
    rows = _rows(tmp_path / "c18.zip", m["package_root"], "features.csv")
    src = create_engine(f"sqlite:///{export_source}")
    fe = FeatureExtractor()
    with Session(src) as s:
        for r in rows:
            prov = {}
            f = fe.extract_for_window(s, int(r["node_id"]), parse_utc(r["window_start_utc"]),
                                      parse_utc(r["window_end_utc"]), provenance=prov)
            if f is None:
                assert r["feature_status"] == "insufficient_data" and r["rssi_mean"] is None
            else:
                assert [float(r[k]) for k in FEATURE_NAMES] == [float(f[k]) for k in FEATURE_NAMES]
            assert int(r["neighbor_record_count"]) == prov["neighbor_record_count"]
            assert json.loads(r["source_node_ids"]) == sorted(prov["source_node_ids"] | {int(r["node_id"])})
    src.dispose()


def test_adjacent_exports_equal_single_export(export_source, tmp_path):
    """(T0, T0+1h] ∪ (T0+1h, T0+2h] = (T0, T0+2h]."""
    a = _export(export_source, tmp_path, "a.zip", export_kind="features", end=EXPORT_T0 + timedelta(hours=1))
    b = _export(export_source, tmp_path, "b.zip", export_kind="features", start=EXPORT_T0 + timedelta(hours=1))
    whole = _export(export_source, tmp_path, "w.zip", export_kind="features")
    key = lambda r: (int(r["node_id"]), r["window_end_utc"])
    parts = sorted(_rows(tmp_path / "a.zip", a["package_root"], "features.csv")
                   + _rows(tmp_path / "b.zip", b["package_root"], "features.csv"), key=key)
    single = sorted(_rows(tmp_path / "w.zip", whole["package_root"], "features.csv"), key=key)
    assert parts == single


def test_single_node_selection(export_source, tmp_path):
    """Só o nó selecionado; sem dados aparece como insuficiente."""
    m = _export(export_source, tmp_path, "n4.zip", export_kind="features", node_ids=(4,))
    rows = _rows(tmp_path / "n4.zip", m["package_root"], "features.csv")
    assert {r["node_id"] for r in rows} == {"4"}
    assert all(r["feature_status"] == "insufficient_data" for r in rows)
    cov = _rows(tmp_path / "n4.zip", m["package_root"], "coverage.csv")
    assert len(cov) == 1 and cov[0]["written_windows"] == cov[0]["expected_windows"]


def test_schema_documents_feature_units(export_source, tmp_path):
    m = _export(export_source, tmp_path, "s.zip", export_kind="features")
    with zipfile.ZipFile(tmp_path / "s.zip") as z:
        schema = json.loads(z.read(f"{m['package_root']}/schema.json"))
        readme = z.read(f"{m['package_root']}/README.md").decode()
    cols = {c["name"]: c for c in schema["files"]["features.csv"]["columns"]}
    assert cols["packet_loss_rate"]["unit"].startswith("fração")
    assert "não é PDR físico" in cols["pdr_mean"]["description"]
    assert cols["feature_status"]["enum"] == ["valid", "insufficient_data", "invalid_numeric", "error"]
    assert "coverage.csv" in schema["files"]
    assert "reference_eligible_by_policy" in readme and "fit_predict" in readme
