"""Importação do pacote bruto: reconstrução em outra máquina."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlmodel import Session

from tests.conftest import export_request

SETTINGS = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE="2026-05-05T03:00:00Z",
                           ML_REFERENCE_MARGIN_MINUTES=15.0, model_fields_set=set())
NOW = datetime(2026, 10, 8, 12, 0)


@pytest.fixture(scope="module")
def imported(export_source, tmp_path_factory):
    from dataset_export.importer import import_dataset
    from dataset_export.package import export_dataset
    from dataset_export.snapshot import build_source_engine
    d = tmp_path_factory.mktemp("imp")
    engine = build_source_engine(f"sqlite:///{export_source}")
    manifest = export_dataset(engine, export_request(), d / "p.zip", settings=SETTINGS, now=NOW)
    engine.dispose()
    info = import_dataset(d / "p.zip", d / "dados.db")
    return d / "dados.db", info, manifest


def test_rows_identical_to_source(imported, export_source):
    db, info, manifest = imported
    src, dst = create_engine(f"sqlite:///{export_source}"), create_engine(f"sqlite:///{db}")
    for table, key in (("telemetry", "id"), ("neighbor", "id"), ("node", "node_id"),
                       ("network", "id"), ("simulation_run", "id")):
        with dst.connect() as c:
            got = c.execute(text(f"SELECT * FROM {table} ORDER BY {key}")).fetchall()
        ids = ",".join(str(r[0]) for r in got)
        with src.connect() as c:
            want = c.execute(text(f"SELECT * FROM {table} WHERE {key} IN ({ids}) ORDER BY {key}")).fetchall()
        assert got == want, table
        assert len(got) == manifest["tables"][table]["rows"]
    with dst.connect() as c:
        assert c.execute(text("PRAGMA foreign_key_check")).fetchall() == []
    src.dispose(); dst.dispose()


def test_sidecar_keeps_manifest_interpretation(imported):
    db, info, manifest = imported
    sidecar = json.loads(db.with_suffix(".dataset.json").read_text())
    assert sidecar["interpretation"] == manifest["interpretation"]
    assert sidecar["interpretation"]["reference_margin_minutes"] == 15.0
    assert sidecar["source_manifest"]["dataset_id"] == manifest["dataset_id"]
    assert info["interpretation"] == manifest["interpretation"]


@pytest.mark.parametrize("version,accepted", [("dataset-v1", True), ("dataset-v2", False)])
def test_old_package_prefix_preserves_only_compatible_format(imported, tmp_path, version, accepted):
    import hashlib
    import zipfile
    from dataset_export.importer import import_dataset
    from dataset_export.package import PackageError
    db, _, manifest = imported
    root = manifest["package_root"]
    with zipfile.ZipFile(db.parent / "p.zip") as source:
        files = {name: source.read(name) for name in source.namelist()}
    legacy_version = "legacy064-" + version
    schema = json.loads(files[f"{root}/schema.json"])
    schema["format"] = legacy_version
    files[f"{root}/schema.json"] = json.dumps(schema).encode()
    manifest = json.loads(files[f"{root}/manifest.json"])
    manifest["format"] = legacy_version
    manifest["files"]["schema.json"]["bytes"] = len(files[f"{root}/schema.json"])
    manifest["files"]["schema.json"]["sha256"] = hashlib.sha256(files[f"{root}/schema.json"]).hexdigest()
    files[f"{root}/manifest.json"] = json.dumps(manifest).encode()
    package = tmp_path / "old.zip"
    with zipfile.ZipFile(package, "w") as target:
        for name, data in files.items():
            target.writestr(name, data)
    if accepted:
        info = import_dataset(package, tmp_path / "importado.db")
        assert info["source_manifest"]["format"] == legacy_version
        assert info["format"] == "import-v1"
    else:
        with pytest.raises(PackageError, match="formato"):
            import_dataset(package, tmp_path / "incompativel.db")


def test_receiving_machine_settings_do_not_change_interpretation(imported):
    from dataset_export.importer import load_interpretation
    from dataset_export.interpretation import InterpretationConflict, resolve_for_computation
    db, _, _ = imported
    params = load_interpretation(db)
    other = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=0.0,
                            DATASET_SOURCE_ID="outra-maquina", model_fields_set=set())
    eff, applied = resolve_for_computation(params, other)
    assert eff.reference_margin_minutes == 15.0 and eff.source_id == "lab-teste" and applied == {}
    explicit = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=0.0,
                               model_fields_set={"ML_REFERENCE_MARGIN_MINUTES"})
    with pytest.raises(InterpretationConflict):
        resolve_for_computation(params, explicit)


def test_features_and_reference_policy_match_source(imported, export_source):
    """O recorte contém todo o contexto que o extrator e a política consultam."""
    from ai.feature_engineering import FeatureExtractor
    from ai.reference_policy import ReferencePolicy
    db, _, manifest = imported
    req = export_request()
    src, dst = create_engine(f"sqlite:///{export_source}"), create_engine(f"sqlite:///{db}")
    fe = FeatureExtractor()
    with Session(src) as s_src, Session(dst) as s_dst:
        p_src = ReferencePolicy.from_session(s_src, 15.0)
        p_dst = ReferencePolicy.from_session(s_dst, 15.0)
        for t in req.grid():
            start = t - timedelta(seconds=req.window_seconds)
            for node in (1, 2, 4):
                prov_a, prov_b = {}, {}
                a = fe.extract_for_window(s_src, node, start, t, provenance=prov_a)
                b = fe.extract_for_window(s_dst, node, start, t, provenance=prov_b)
                assert a == b and prov_a == prov_b, (node, t)
                assert (p_src.node_contaminated([node], start, t)
                        == p_dst.node_contaminated([node], start, t))
    src.dispose(); dst.dispose()


def test_tampered_or_existing_target_rejected(imported, tmp_path, export_source):
    from dataset_export.importer import import_dataset
    from dataset_export.package import PackageError
    db, _, _ = imported
    with pytest.raises(PackageError):
        import_dataset(tmp_path / "nao_existe.zip", tmp_path / "x.db")
    with pytest.raises(FileExistsError):
        import_dataset(db.parent / "p.zip", db)
    assert not (tmp_path / "x.db").exists()
