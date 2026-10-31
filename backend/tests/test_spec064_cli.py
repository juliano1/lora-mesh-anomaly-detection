"""CLIs de exportação e importação."""
import importlib.util
import json

import pytest
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_help():
    for name in ("export_dataset", "import_dataset"):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py"), "--help"],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr


def test_export_then_import(export_source, tmp_path, capsys):
    exp, imp = _load("export_dataset"), _load("import_dataset")
    out = tmp_path / "p.zip"
    code = exp.main(["--sqlite", str(export_source), "--source-id", "lab-teste",
                     "--network-id", "1", "--nodes", "1,2,4",
                     "--start", "2026-09-01T00:00:00Z", "--end", "2026-09-01T02:00:00Z",
                     "--window-seconds", "900", "--step-seconds", "420", "--out", str(out)])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["export_kind"] == "raw" and summary["tables"]["node"]["rows"] == 4
    assert imp.main(["--package", str(out), "--out", str(tmp_path / "d.db")]) == 0
    assert json.loads(capsys.readouterr().out)["database"] == "d.db"


def test_errors_exit_2(export_source, tmp_path, capsys):
    exp = _load("export_dataset")
    code = exp.main(["--sqlite", str(export_source), "--source-id", "x", "--network-id", "1",
                     "--start", "2026-09-01T00:00:00", "--end", "2026-09-01T02:00:00Z",
                     "--out", str(tmp_path / "e.zip")])
    assert code == 2 and "fuso" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        exp.main(["--sqlite", str(export_source), "--network-id", "1", "--kind", "xyz",
                  "--start", "2026-09-01T00:00:00Z", "--end", "2026-09-01T02:00:00Z",
                  "--out", str(tmp_path / "f.zip")])
    assert exc.value.code == 2


def test_export_features_kind(export_source, tmp_path, capsys):
    exp = _load("export_dataset")
    assert exp.main(["--sqlite", str(export_source), "--source-id", "lab-teste", "--network-id", "1",
                     "--kind", "features", "--start", "2026-09-01T00:00:00Z",
                     "--end", "2026-09-01T02:00:00Z", "--window-seconds", "900",
                     "--step-seconds", "420", "--out", str(tmp_path / "f.zip")]) == 0
    assert json.loads(capsys.readouterr().out)["export_kind"] == "features"

