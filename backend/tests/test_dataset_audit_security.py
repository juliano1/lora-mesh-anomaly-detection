"""Auditar um dataset não executa os exemplos Python fornecidos pelo pacote."""
import importlib.util
import sys
from pathlib import Path


def _auditor():
    path = Path(__file__).resolve().parents[2] / "scripts/audit_dataset_export.py"
    spec = importlib.util.spec_from_file_location("dataset_audit_security", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verified_readme_cannot_execute_code(tmp_path):
    module = _auditor()
    marker = tmp_path / "executed"
    readme = f"```python\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n```\n"
    audit = module.Audit()
    module._readme_blocks(audit, readme, verified=True)
    assert not marker.exists()
    assert audit.checks[-1]["status"] == "pass"
    assert audit.checks[-1]["observed"]["executed"] is False


def test_invalid_readme_python_is_reported():
    module = _auditor()
    audit = module.Audit()
    module._readme_blocks(audit, "```python\nif invalid syntax\n```\n", verified=True)
    assert audit.checks[-1]["status"] == "fail"
