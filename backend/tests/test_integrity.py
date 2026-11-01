"""Casos de integridade da exportação (nulos no CSV, snapshot, limites e arquivos
obrigatórios), incluindo Python 3.11/3.12."""
import csv
import io
import json
import zipfile
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from tests.conftest import export_request


@pytest.mark.parametrize("values", [
    [None], [""], [None, "", None], ["NA", "NULL", "nan", "0"],
    ['a,"b"\r\nc', '\x00', " ", None, ""], ['"', ',', '\r', '\n'],
])
def test_csv_null_roundtrip_on_both_writers(values, monkeypatch):
    from dataset_export.codec import csv_writer, read_csv
    expected = None
    for fallback in (False, True):
        if fallback:
            monkeypatch.delattr(csv, "QUOTE_NOTNULL", raising=False)
        output = io.StringIO(newline="")
        csv_writer(output).writerow(values)
        encoded = output.getvalue()
        assert list(read_csv(io.StringIO(encoded, newline=""))) == [values]
        if expected is not None:
            assert encoded == expected
        expected = encoded


def test_snapshot_with_successful_concurrent_writer_in_wal(tmp_path):
    from dataset_export.snapshot import build_source_engine, snapshot_session
    path = tmp_path / "wal.db"
    writer = create_engine(f"sqlite:///{path}")
    with writer.begin() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        conn.exec_driver_sql("CREATE TABLE t (value INTEGER)")
        conn.exec_driver_sql("INSERT INTO t VALUES (1)")
    source = build_source_engine(f"sqlite:///{path}")
    with snapshot_session(source) as session:
        assert session.exec(text("SELECT value FROM t")).scalar_one() == 1
        with writer.begin() as conn:
            conn.exec_driver_sql("UPDATE t SET value=2")
        assert session.exec(text("SELECT value FROM t")).scalar_one() == 1
        with pytest.raises(OperationalError):
            session.exec(text("UPDATE t SET value=3"))
    with writer.connect() as conn:
        assert conn.exec_driver_sql("SELECT value FROM t").scalar_one() == 2
    source.dispose()
    writer.dispose()


def test_raw_never_runs_extractor_and_manifest_matches_return(export_source, tmp_path, monkeypatch):
    from ai.feature_engineering import FeatureExtractor
    from dataset_export.package import export_dataset, verify_package
    from dataset_export.snapshot import build_source_engine
    def forbidden(*args, **kwargs):
        raise AssertionError("raw não pode executar o extrator")
    monkeypatch.setattr(FeatureExtractor, "extract_for_window", forbidden)
    source = build_source_engine(f"sqlite:///{export_source}")
    settings = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=0.)
    path = tmp_path / "raw.zip"
    manifest = export_dataset(source, export_request(), path, settings=settings,
                              now=datetime(2026, 10, 8))
    source.dispose()
    original_read = zipfile.ZipFile.read
    def no_csv_read(self, name, *args, **kwargs):
        assert not str(name).endswith('.csv'), "verificação deve ler CSV em streaming"
        return original_read(self, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'read', no_csv_read)
    checked, _ = verify_package(path)
    assert checked == manifest


def test_omitted_required_file_is_rejected_even_if_manifest_omits_it(export_source, tmp_path):
    from dataset_export.package import export_dataset, PackageError, verify_package
    from dataset_export.snapshot import build_source_engine
    source = build_source_engine(f"sqlite:///{export_source}")
    path = tmp_path / "raw.zip"
    export_dataset(source, export_request(), path, settings=SimpleNamespace(
        DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=0.), now=datetime(2026, 10, 8))
    source.dispose()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(path) as original, zipfile.ZipFile(bad, 'w') as dest:
        for name in original.namelist():
            if name.endswith('/neighbors.csv'):
                continue
            content = original.read(name)
            if name.endswith('/manifest.json'):
                manifest = json.loads(content)
                manifest['files'].pop('neighbors.csv')
                content = json.dumps(manifest).encode()
            dest.writestr(name, content)
    with pytest.raises(PackageError, match='obrigatórios'):
        verify_package(bad)


def test_cli_does_not_ignore_zero_step(export_source, tmp_path, capsys):
    from tests.test_cli import _load
    export = _load('export_dataset')
    result = export.main(['--sqlite', str(export_source), '--source-id', 'lab',
        '--network-id', '1', '--start', '2026-09-01T00:00:00Z',
        '--end', '2026-09-01T01:00:00Z', '--step-seconds', '0', '--out', str(tmp_path/'bad.zip')])
    assert result == 2 and 'step_seconds' in capsys.readouterr().err
    assert not (tmp_path/'bad.zip').exists()


def test_invalid_margin_is_rejected_before_source_read():
    from dataset_export.interpretation import InterpretationParams
    with pytest.raises(ValueError, match='reference_margin'):
        InterpretationParams.for_export(export_request(), SimpleNamespace(
            DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=float('nan')))


def test_package_size_limits_are_checked_before_decompression(export_source, tmp_path, monkeypatch):
    """Bomba de ZIP: o tamanho declarado é conferido antes de descomprimir."""
    import dataset_export.package as package
    from dataset_export.snapshot import build_source_engine
    source = build_source_engine(f"sqlite:///{export_source}")
    path = tmp_path / "raw.zip"
    package.export_dataset(source, export_request(), path, settings=SimpleNamespace(
        DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=0.), now=datetime(2026, 10, 8))
    source.dispose()
    with zipfile.ZipFile(path) as zf:
        total = sum(info.file_size for info in zf.infolist())
    monkeypatch.setattr(package, "MAX_PACKAGE_UNCOMPRESSED_BYTES", total - 1)
    with pytest.raises(package.PackageError, match="limite de tamanho"):
        package.verify_package(path)
    monkeypatch.setattr(package, "MAX_PACKAGE_UNCOMPRESSED_BYTES", total)
    monkeypatch.setattr(package, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(package.PackageError, match="manifesto acima"):
        package.verify_package(path)
