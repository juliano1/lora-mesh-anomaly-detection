"""Regras do scanner de publicação (scripts/check_publication.py)."""
import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_publication.py"


@pytest.fixture(scope="module")
def scanner():
    spec = importlib.util.spec_from_file_location("check_publication", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rules(scanner, text, path="docs/exemplo.md"):
    return sorted(f["rule"] for f in scanner.scan_text(path, text))


# Amostras montadas por partes: o próprio arquivo de teste não contém os padrões e
# não aparece como achado quando o scanner examina a árvore.
AT, COLON = "@", ":"
URL_WITH_PASSWORD = "postgresql" + COLON + "//user" + COLON + "segredo" + AT + "db/x"  # pragma: allowlist secret


@pytest.mark.parametrize("text,rule", [
    ("DATABASE_URL=" + URL_WITH_PASSWORD, "url_with_credentials"),
    ("broker em " + ".".join(["192", "168", "1", "100"]), "ip_address"),
    ("contato: alguem" + AT + "exemplo.com.br", "email_address"),
    ("mac " + "-".join(["01", "23", "45", "67", "89", "ab"]), "mac_address"),
    ("caminho /" + "home/fulano/projeto", "personal_path"),
    ("caminho C:" + "\\Users\\fulano\\projeto", "personal_path"),
    ('MQTT_PASS = "valor-real"', "literal_secret_candidate"),
])
def test_flags_identifiers_and_credentials(scanner, text, rule):
    assert rule in _rules(scanner, text)


@pytest.mark.parametrize("path,text", [
    (".env", "JWT_SECRET_KEY=" + "valor-sem-aspas"),                 # .env sem aspas
    ("compose.yml", "  DB_PASSWORD: " + "valor-sem-aspas"),           # YAML sem aspas
    ("app/config.py", 'SECRET_KEY = "' + 'valor-real"'),              # nome genérico
    ("firmware/x.h", '#define WIFI_PASS "' + 'valor-real"'),
])
def test_flags_secret_assignments_beyond_fixed_names(scanner, path, text):
    assert "literal_secret_candidate" in _rules(scanner, text, path)


@pytest.mark.parametrize("path,text", [
    (".env", "JWT_SECRET_KEY="),                                      # vazio
    (".env", "MQTT_PASSWORD=${MQTT_PASSWORD}"),                       # referência
    ("app/config.py", "JWT_SECRET_KEY = os.environ['X']"),            # sem aspas em código
    ("schemas.py", 'token_type: str = "bearer"'),
])
def test_ignores_empty_referenced_or_code_assignments(scanner, path, text):
    assert "literal_secret_candidate" not in _rules(scanner, text, path)


@pytest.mark.parametrize("text", [
    "DATABASE_URL=postgresql://<usuario>:<senha>@localhost:5432/lora_mesh",
    "API em 127.0.0.1, bind 0.0.0.0, exemplo 192.0.2.10",
    "autor: alguem@example.com",
    "versão 10.0.0.300 não é IPv4",
])
def test_ignores_placeholders_and_documentation_values(scanner, text):
    assert _rules(scanner, text) == []


def test_findings_never_include_the_matched_value(scanner):
    findings = scanner.scan_text("x.txt", URL_WITH_PASSWORD)
    assert all("segredo" not in str(value) for f in findings for value in f.values())


def test_inventory_keeps_rows_of_removed_files(scanner, tmp_path, monkeypatch):
    inventory = tmp_path / "arquivos.csv"
    with inventory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arquivo", "estado", "destino"])
        writer.writeheader()
        writer.writerow({"arquivo": "retirado.py", "estado": "x", "destino": "retirar da distribuição"})
    monkeypatch.setattr(scanner, "current_tree", lambda _inv: ([{"arquivo": "mantido.py", "estado": "ok"}], []))
    monkeypatch.setattr("sys.argv", ["check_publication.py", "--inventory", str(inventory)])
    assert scanner.main() == 0
    with inventory.open(encoding="utf-8", newline="") as stream:
        rows = {row["arquivo"]: row for row in csv.DictReader(stream)}
    assert rows["retirado.py"]["destino"] == "retirar da distribuição"
    assert rows["retirado.py"]["estado"] == "ausente da árvore"
    assert "mantido.py" in rows


def _git(repo, *args):
    import subprocess
    subprocess.run(["git", "-c", "user.name=teste", "-c", "user.email=teste@invalid", *args],
                   cwd=repo, check=True, capture_output=True)


def test_history_of_repository_without_commits_is_empty(scanner, tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    monkeypatch.setattr(scanner, "ROOT", tmp_path)
    counts, findings = scanner.history()
    assert counts == {"text_blobs": 0, "binary_blobs_skipped": 0, "large_blobs_skipped": 0}
    assert findings == []


def test_history_finds_credentials_in_committed_blobs(scanner, tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    (tmp_path / "config.txt").write_text("DATABASE_URL=" + URL_WITH_PASSWORD + "\n", encoding="utf-8")
    _git(tmp_path, "add", "config.txt")
    _git(tmp_path, "commit", "-q", "-m", "c1")
    (tmp_path / "config.txt").write_text("limpo\n", encoding="utf-8")
    _git(tmp_path, "commit", "-q", "-am", "c2")
    monkeypatch.setattr(scanner, "ROOT", tmp_path)
    counts, findings = scanner.history()
    assert counts["text_blobs"] == 2
    assert [f["rule"] for f in findings] == ["url_with_credentials"]
