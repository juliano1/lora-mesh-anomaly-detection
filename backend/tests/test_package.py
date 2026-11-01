"""Pacote bruto verificável."""
import csv
import io
import json
import zipfile
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from tests.conftest import EXPORT_T0, export_request

SETTINGS = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=0.0,
                           model_fields_set=set())
NOW = datetime(2026, 10, 8, 12, 0)


@pytest.fixture(scope="module")
def package(export_source, tmp_path_factory):
    from dataset_export.package import export_dataset
    from dataset_export.snapshot import build_source_engine
    out = tmp_path_factory.mktemp("pkg") / "pacote.zip"
    engine = build_source_engine(f"sqlite:///{export_source}")
    manifest = export_dataset(engine, export_request(), out, settings=SETTINGS,
                              sqlite_file=export_source, now=NOW)
    engine.dispose()
    return out, manifest


def _read(zpath, root, name):
    from dataset_export.codec import read_csv
    with zipfile.ZipFile(zpath) as z, z.open(f"{root}/{name}") as raw:
        return list(read_csv(io.TextIOWrapper(raw, encoding="utf-8", newline="")))


def test_expected_files_and_no_features(package):
    out, m = package
    root = m["package_root"]
    with zipfile.ZipFile(out) as z:
        names = {n.split("/", 1)[1] for n in z.namelist()}
    assert names == {"manifest.json", "schema.json", "README.md", "events_template.csv",
                     "telemetry.csv", "neighbors.csv", "nodes.csv", "networks.csv",
                     "simulation_runs.csv"}
    assert m["export_kind"] == "raw" and m["labels_status"] == "not_provided"
    assert root.startswith("dataset_lab-teste_20260901T000000Z_20260901T020000Z_")


def test_manifest_counts_hashes_and_validation(package):
    from dataset_export.package import verify_package
    out, m = package
    verified, root = verify_package(out)
    assert verified["dataset_id"] == m["dataset_id"] and root == m["package_root"]
    for name, info in m["files"].items():
        if name.endswith(".csv") and name != "events_template.csv":
            assert len(_read(out, root, name)) - 1 == info["rows"]
    assert m["validation"] == {"foreign_key_violations": 0, "counts_match": True}
    assert m["tables"]["simulation_run"]["rows"] == 3
    assert m["nodes"] == {"requested": [1, 2, 4], "context": [3]}
    assert m["resolved"]["interval_convention"] == "(start, end]"
    assert m["interpretation"]["source_id"] == "lab-teste"
    assert m["source"]["dialect"] == "sqlite" and len(m["source"]["file_sha256"]) == 64
    assert set(m["timings_seconds"]) == {"read_and_materialize", "finalize", "write_csv", "package"}
    assert "commit" in m["code"] and m["source_hashes"]


def test_one_byte_change_is_detected(package, tmp_path):
    from dataset_export.package import PackageError, verify_package
    out, m = package
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(bad, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith("neighbors.csv"):
                data = data[:-2] + bytes([data[-2] ^ 1]) + data[-1:]
            dst.writestr(item, data)
    with pytest.raises(PackageError, match="neighbors.csv"):
        verify_package(bad)


def test_roles_and_ordering(package):
    out, m = package
    root = m["package_root"]
    nodes = _read(out, root, "nodes.csv")
    header = nodes[0]
    role = {int(r[header.index("node_id")]): r[header.index("record_role")] for r in nodes[1:]}
    assert role == {1: "requested", 2: "requested", 3: "context", 4: "requested"}
    sims = _read(out, root, "simulation_runs.csv")
    h = sims[0]
    srole = {int(r[h.index("id")]): r[h.index("record_role")] for r in sims[1:]}
    assert srole == {10: "context", 11: "requested", 12: "context"}
    tel = _read(out, root, "telemetry.csv")
    h = tel[0]
    keys = [(int(r[h.index("node_id")]), r[h.index("timestamp")], int(r[h.index("id")])) for r in tel[1:]]
    assert keys == sorted(keys)
    roles = {r[h.index("record_role")] for r in tel[1:]}
    assert roles == {"requested", "context"}
    assert all(r[h.index("source_id")] == "lab-teste" for r in tel[1:])


def test_difficult_text_and_nulls_roundtrip(package):
    out, m = package
    root = m["package_root"]
    nodes = _read(out, root, "nodes.csv")
    h = nodes[0]
    names = {r[h.index("name")] for r in nodes[1:]}
    assert 'Nó, Ação "1"' in names
    tel = _read(out, root, "telemetry.csv")
    h = tel[0]
    assert tel[1][h.index("payload_raw")] == '{"msg": "a,b", "q": "x\\"y"}\nsegunda linha'
    temps = {r[h.index("temperature")] for r in tel[1:]}
    assert None in temps and "21.5" in temps
    assert all(r[h.index("timestamp")].endswith(".123456Z") for r in tel[1:])


def test_schema_and_events_template(package):
    out, m = package
    root = m["package_root"]
    with zipfile.ZipFile(out) as z:
        schema = json.loads(z.read(f"{root}/schema.json"))
        readme = z.read(f"{root}/README.md").decode()
    cols = {c["name"]: c for c in schema["files"]["telemetry.csv"]["columns"]}
    assert cols["snr"]["type"] == "float" and cols["temperature"]["nullable"] is True
    assert cols["snr"]["unit"] == "dB"
    assert cols["record_role"]["type"] == "string" and "requested" in cols["record_role"]["description"]
    assert schema["conventions"]["null"]
    assert _read(out, root, "events_template.csv") == [[
        "event_id", "campaign_id", "node_id", "event_type", "origin", "start_at_utc",
        "end_at_utc", "observer", "evidence", "notes"]]
    assert "import_dataset.py" in readme


def test_out_path_must_not_exist_and_failure_leaves_nothing(export_source, tmp_path, monkeypatch):
    import dataset_export.package as pkg
    from dataset_export.snapshot import build_source_engine
    engine = build_source_engine(f"sqlite:///{export_source}")
    exists = tmp_path / "x.zip"
    exists.write_bytes(b"")
    with pytest.raises(pkg.ExportError):
        pkg.export_dataset(engine, export_request(), exists, settings=SETTINGS, now=NOW)
    created = []
    real_mkdtemp = pkg.tempfile.mkdtemp
    monkeypatch.setattr(pkg.tempfile, "mkdtemp",
                        lambda **kw: created.append(real_mkdtemp(**kw)) or created[-1])
    monkeypatch.setattr(pkg, "build_schema", lambda: (_ for _ in ()).throw(RuntimeError("falha")))
    out = tmp_path / "y.zip"
    with pytest.raises(RuntimeError):
        pkg.export_dataset(engine, export_request(), out, settings=SETTINGS, now=NOW)
    assert not out.exists() and not list(tmp_path.glob("y.zip*"))
    from pathlib import Path
    assert created and not Path(created[0]).exists()
    engine.dispose()


def test_invalid_request_rejected_before_reading(export_source, tmp_path):
    from dataset_export.package import export_dataset
    from dataset_export.snapshot import build_source_engine
    engine = build_source_engine(f"sqlite:///{export_source}")
    with pytest.raises(ValueError, match="futuro"):
        export_dataset(engine, export_request(end=NOW + timedelta(hours=1)), tmp_path / "z.zip",
                       settings=SETTINGS, now=NOW)
    engine.dispose()
