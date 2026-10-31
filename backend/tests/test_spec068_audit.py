"""Auditoria de reconciliação do pacote contra a origem."""
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import export_request

SETTINGS = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE="2026-01-01T00:00:00Z",
                           ML_REFERENCE_MARGIN_MINUTES=0.0, model_fields_set=set())
NOW = datetime(2026, 10, 21, 12, 0)
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _git_resolves(rev: str) -> bool:
    proc = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
                          cwd=SCRIPTS.parent, capture_output=True, text=True)
    return proc.returncode == 0


# O auditor compara o pacote com o HEAD do repositório: sem commit não há referência
# para certificar o código, e o auditor falha de propósito. Esses casos ficam pulados
# (não aprovados) até existir histórico.
requires_head = pytest.mark.skipif(not _git_resolves("HEAD"),
                                   reason="repositório sem commit: auditor não tem HEAD para comparar")
requires_parent_commit = pytest.mark.skipif(not _git_resolves("HEAD~1"),
                                            reason="caso --expect-commit exige ao menos dois commits")


def _export(source, out, **over):
    from dataset_export.package import export_dataset
    from dataset_export.snapshot import build_source_engine
    eng = build_source_engine(f"sqlite:///{source}")
    m = export_dataset(eng, export_request(**over), out, settings=SETTINGS, now=NOW)
    eng.dispose()
    return m


def _audit(args):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import audit_dataset_export as mod
    finally:
        sys.path.pop(0)
    return mod.main([str(a) for a in args])


@pytest.fixture(scope="module")
def full_package(export_source, tmp_path_factory):
    out = tmp_path_factory.mktemp("audit") / "full.zip"
    _export(export_source, out, export_kind="full")
    return out


@requires_head
def test_audit_passes_on_consistent_package(export_source, full_package, tmp_path):
    report = tmp_path / "VALIDACAO.json"
    rc = _audit(["--package", full_package, "--sqlite", export_source, "--out", report])
    data = json.loads(report.read_text())
    failed = [c for c in data["checks"] if c["status"] == "fail"]
    assert rc == 0 and not failed, failed
    ids = {c["id"] for c in data["checks"]}
    assert {"ids:telemetry", "values:neighbor", "role:simulation_run", "features:values",
            "features:grid", "coverage:sums", "schema:complete", "manifest:hashes",
            "readme:python_blocks", "origin:cases"} <= ids


def _rewrite_csv(src_zip, dst_zip, name, transform):
    """Reescreve um CSV do pacote com `transform(rows: list[dict]) -> list[dict]` (valores como texto,
    None = nulo) usando o escritor/leitor do projeto, e corrige hash/tamanho/linhas no manifesto."""
    import hashlib
    import io
    from dataset_export.codec import csv_writer, read_csv
    with zipfile.ZipFile(src_zip) as z:
        root = z.namelist()[0].split("/")[0]
        items = {n: z.read(n) for n in z.namelist()}
    it = read_csv(io.StringIO(items[f"{root}/{name}"].decode(), newline=""))
    header = next(it)
    rows = transform([dict(zip(header, r)) for r in it])
    buf = io.StringIO()
    w = csv_writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow([r[c] for c in header])
    data = buf.getvalue().encode()
    items[f"{root}/{name}"] = data
    manifest = json.loads(items[f"{root}/manifest.json"])
    manifest["files"][name].update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=len(rows))
    items[f"{root}/manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(dst_zip, "w") as z:
        for n, b in items.items():
            z.writestr(n, b)


def _checks(report):
    return {c["id"]: c for c in json.loads(report.read_text())["checks"]}


def test_audit_detects_tampered_value(export_source, full_package, tmp_path):
    def minus_one(rows):
        rows[0]["rssi"] = str(int(rows[0]["rssi"]) - 1)
        return rows
    tampered = tmp_path / "tampered.zip"
    _rewrite_csv(full_package, tampered, "neighbors.csv", minus_one)
    report = tmp_path / "V.json"
    rc = _audit(["--package", tampered, "--sqlite", export_source, "--out", report])
    assert rc == 1 and _checks(report)["values:neighbor"]["status"] == "fail"


def test_audit_detects_duplicated_window_replacing_missing_one(export_source, full_package, tmp_path):
    """Mesma contagem de linhas, uma janela duplicada e outra ausente → features:keys falha."""
    def dup(rows):
        return [rows[0]] + rows[:-1]           # repete a 1ª, descarta a última
    tampered = tmp_path / "dup.zip"
    _rewrite_csv(full_package, tampered, "features.csv", dup)
    report = tmp_path / "V.json"
    rc = _audit(["--package", tampered, "--sqlite", export_source, "--out", report])
    checks = _checks(report)
    assert rc == 1 and checks["features:grid"]["status"] == "pass"
    assert checks["features:keys"]["status"] == "fail"
    assert checks["features:keys"]["observed"]["duplicated"] and checks["features:keys"]["observed"]["missing"]


def test_audit_detects_wrong_unavailable_features(export_source, full_package, tmp_path):
    def wrong(rows):
        for r in rows:
            if r["feature_status"] == "valid":
                r["unavailable_features"] = json.dumps({"rssi_mean": "no_eligible_link"})
        return rows
    tampered = tmp_path / "unav.zip"
    _rewrite_csv(full_package, tampered, "features.csv", wrong)
    report = tmp_path / "V.json"
    rc = _audit(["--package", tampered, "--sqlite", export_source, "--out", report])
    assert rc == 1 and _checks(report)["features:defaults_match"]["status"] == "fail"


@requires_head
def test_audit_checks_manifest_against_expected_request(export_source, full_package, tmp_path):
    report = tmp_path / "V.json"
    rc = _audit(["--package", full_package, "--sqlite", export_source, "--out", report,
                 "--expect-source-id", "lab-teste", "--expect-network-id", "1",
                 "--expect-start", "2026-09-01T00:00:00Z", "--expect-end", "2026-09-01T02:00:00Z",
                 "--expect-window", "900", "--expect-step", "420", "--expect-kind", "full", "--expect-nodes", "1,2,4"])
    checks = {c["id"]: c for c in json.loads(report.read_text())["checks"]}
    assert rc == 0 and checks["request:matches_expected"]["status"] == "pass"
    rc = _audit(["--package", full_package, "--sqlite", export_source, "--out", report, "--expect-window", "600"])
    checks = {c["id"]: c for c in json.loads(report.read_text())["checks"]}
    assert rc == 1 and checks["request:matches_expected"]["status"] == "fail"


@requires_head
def test_compare_lists_variable_metadata(export_source, full_package, tmp_path):
    other = tmp_path / "full2.zip"
    _export(export_source, other, export_kind="full")
    report = tmp_path / "C.json"
    rc = _audit(["--package", full_package, "--sqlite", export_source, "--out", tmp_path / "V.json",
                 "--compare", other, "--compare-out", report])
    data = json.loads(report.read_text())
    assert rc == 0
    assert data["csv_equal"] == {n: True for n in data["csv_equal"]} and "telemetry.csv" in data["csv_equal"]
    assert data["scientific_params_equal"] is True
    assert set(data["variable_metadata"]) >= {"dataset_id", "package_root"}


# ─── Adulterações, casos de origem e documentação do pacote ───────────────────────
import hashlib  # noqa: E402
import io  # noqa: E402
import shutil  # noqa: E402
from datetime import timedelta  # noqa: E402

from tests.conftest import EXPORT_T0  # noqa: E402

NO_IMPORT = "--no-import-check"   # reimportação só nos testes que a exercitam (custo de subprocesso)
RAW_TABLES = {"telemetry.csv": "telemetry", "neighbors.csv": "neighbor", "nodes.csv": "node",
              "networks.csv": "network", "simulation_runs.csv": "simulation_run"}


def _git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=SCRIPTS.parent, capture_output=True,
                          text=True).stdout.strip()


def _hashes_at(commit):
    """SHA-256 dos fontes de provenance.RELEVANT_SOURCES como estão no commit."""
    from dataset_export.provenance import RELEVANT_SOURCES
    hashes = {}
    for rel in RELEVANT_SOURCES:
        proc = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=SCRIPTS.parent, capture_output=True)
        if proc.returncode == 0:
            hashes[rel] = hashlib.sha256(proc.stdout).hexdigest()
    return hashes


@pytest.fixture(scope="module", autouse=True)
def _package_code_at_head():
    """Os pacotes destes testes são gerados aqui, numa árvore que pode estar em edição: a revisão e
    os hashes dos fontes gravados pelo gerador são fixados em HEAD/limpo. O lado do auditor (git
    rev-parse HEAD, git show HEAD:<fonte>) continua real; os testes de manifest:code trocam este valor."""
    head = _git_head()
    hashes = _hashes_at(head)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("dataset_export.package.code_revision", lambda: {"commit": head, "dirty": False})
        mp.setattr("dataset_export.package.source_hashes", lambda: dict(hashes))
        yield head


def _export_with(source, out, settings=SETTINGS, **over):
    from dataset_export.package import export_dataset
    from dataset_export.snapshot import build_source_engine
    eng = build_source_engine(f"sqlite:///{source}")
    try:
        return export_dataset(eng, export_request(**over), out, settings=settings, now=NOW)
    finally:
        eng.dispose()


def _settings(tracked="2026-01-01T00:00:00Z", margin=0.0):
    return SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE=tracked, ML_REFERENCE_MARGIN_MINUTES=margin,
                           model_fields_set=set())


def _repack(src_zip, dst_zip, *edits, fix_manifest=True, root=None):
    """Aplica edits(items: nome → bytes) e regrava o ZIP; com fix_manifest, sha256/bytes/linhas e
    tables do manifesto são recalculados, para que só a adulteração pretendida fique visível."""
    from dataset_export.codec import read_csv
    with zipfile.ZipFile(src_zip) as z:
        old_root = z.namelist()[0].split("/")[0]
        items = {n.split("/", 1)[1]: z.read(n) for n in z.namelist()}
    for edit in edits:
        edit(items)
    if fix_manifest:
        manifest = json.loads(items["manifest.json"])
        for name, info in manifest["files"].items():
            data = items[name]
            info.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            if name.endswith(".csv"):
                info["rows"] = sum(1 for _ in read_csv(io.StringIO(data.decode(), newline=""))) - 1
                if name in RAW_TABLES:
                    manifest["tables"][RAW_TABLES[name]]["rows"] = info["rows"]
        items["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
    with zipfile.ZipFile(dst_zip, "w") as z:
        for name, data in items.items():
            z.writestr(f"{root or old_root}/{name}", data)


def _csv_edit(name, transform):
    from dataset_export.codec import csv_writer, read_csv

    def edit(items):
        it = read_csv(io.StringIO(items[name].decode(), newline=""))
        header = next(it)
        rows = transform([dict(zip(header, r)) for r in it])
        buf = io.StringIO()
        writer = csv_writer(buf)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[c] for c in header])
        items[name] = buf.getvalue().encode()
    return edit


def _json_edit(name, fn):
    def edit(items):
        data = json.loads(items[name])
        fn(data)
        items[name] = json.dumps(data, indent=2, ensure_ascii=False).encode()
    return edit


def _run(package, source, tmp_path, *extra):
    report = tmp_path / "V.json"
    report.unlink(missing_ok=True)
    rc = _audit(["--package", package, "--sqlite", source, "--out", report, *extra])
    return rc, (_checks(report) if report.exists() else {})


def _tampered(full_package, tmp_path, *edits, **kw):
    out = tmp_path / "t.zip"
    out.unlink(missing_ok=True)
    _repack(full_package, out, *edits, **kw)
    return out


def _features_rows(package):
    from dataset_export.codec import read_csv
    with zipfile.ZipFile(package) as z:
        root = z.namelist()[0].split("/")[0]
        it = read_csv(io.StringIO(z.read(f"{root}/features.csv").decode(), newline=""))
        header = next(it)
        return [dict(zip(header, r)) for r in it]


def _swap_first_two(rows):
    rows[0], rows[1] = rows[1], rows[0]
    return rows


def _rename_first(rows):
    rows[0]["node_name"] += " "
    return rows


def _add_event(rows):
    return [{"event_id": "EV-1", "campaign_id": None, "node_id": None, "event_type": "x", "origin": "y",
             "start_at_utc": "2026-09-01T00:00:00Z", "end_at_utc": None, "observer": "z",
             "evidence": None, "notes": None}]


def _set(index, column, fn):
    def transform(rows):
        rows[index][column] = fn(rows[index][column])
        return rows
    return transform


def _null_first(column):
    return _set(0, column, lambda _v: None)


def _strip_z(rows):
    rows[3]["timestamp"] = rows[3]["timestamp"].rstrip("Z")
    return rows


def _all(column, value):
    def transform(rows):
        for r in rows:
            r[column] = value
        return rows
    return transform


def _underscore_sequence(rows):
    """sequence '12' → '1_2' (int() aceita; o texto do codec não)."""
    r = next(r for r in rows if len(r["sequence"]) >= 2)
    r["sequence"] = r["sequence"][0] + "_" + r["sequence"][1:]
    return rows


def _start_plus3(rows):
    """Mesmo instante de window_start_utc escrito em hora local +03:00."""
    from dataset_export.request import parse_utc
    r = rows[3]
    r["window_start_utc"] = (parse_utc(r["window_start_utc"]) + timedelta(hours=3)).isoformat() + "+03:00"
    return rows


@pytest.mark.parametrize("name,transform,check_id", [
    # ordem, identidade, cabeçalho, nulos e valores
    ("telemetry.csv", _swap_first_two, "order:telemetry"),
    ("features.csv", _rename_first, "features:identity"),
    ("events_template.csv", _add_event, "events_template:header"),
    ("coverage.csv", _null_first("first_instant_utc"), "coverage:identity"),
    ("features.csv", _null_first("defaulted_features"), "features:defaults_match"),
    ("features.csv", _null_first("last_observation_utc"), "features:counts"),
    ("neighbors.csv", _strip_z, "values:neighbor"),
    # instante com offset (mesmo instante) não é o formato do pacote
    ("features.csv", _set(3, "window_start_utc", lambda v: v.replace("Z", "+00:00")), "features:window"),
    ("features.csv", _start_plus3, "features:window"),
    # política de extração por linha
    ("features.csv", _all("extraction_policy", "real_only_v1"), "features:extraction_policy"),
    # ordem de features.csv e coverage.csv
    ("features.csv", _swap_first_two, "order:features"),
    ("coverage.csv", _swap_first_two, "order:features"),
    # mesmo número, outro texto
    ("neighbors.csv", _set(0, "rssi", lambda v: " " + v), "values:neighbor"),
    ("telemetry.csv", _set(5, "uptime", lambda v: "+" + v), "values:telemetry"),
    ("telemetry.csv", _underscore_sequence, "values:telemetry"),
    ("neighbors.csv", _set(0, "snr", lambda v: "%e" % float(v)), "values:neighbor"),
    ("telemetry.csv", _set(0, "timestamp", lambda v: v.replace("Z", "+00:00")), "values:telemetry"),
])
def test_audit_added_checks_detect_tampering(export_source, full_package, tmp_path, name, transform, check_id):
    """Cada adulteração falha a verificação correspondente (valor ilegível vira falha de
    auditoria, não erro de execução)."""
    rc, checks = _run(_tampered(full_package, tmp_path, _csv_edit(name, transform)), export_source, tmp_path,
                      NO_IMPORT)
    assert rc == 1 and checks[check_id]["status"] == "fail", checks.get(check_id)


@pytest.fixture(scope="module")
def baseline(export_source, full_package, tmp_path_factory):
    """Uma auditoria completa (com reimportação) do pacote sem adulteração, para os testes que só a leem."""
    return _run(full_package, export_source, tmp_path_factory.mktemp("baseline"))


@requires_head
def test_audit_reports_compared_items(baseline):
    """Nenhuma verificação de valores aprova sem objeto: contagens de itens comparados > 0."""
    rc, checks = baseline
    assert rc == 0
    compared = checks["audit:compared"]["observed"]
    assert all(compared[k] > 0 for k in ("values:telemetry", "values:neighbor", "values:simulation_run",
                                          "features:valid_windows", "features:values", "readme:python_blocks"))
    cases = checks["origin:spec_cases"]["observed"]
    assert (cases["mixed"]["windows"] and cases["reverse_synthetic"]["windows"]
            and cases["reverse_simulation"]["windows"])
    for cid in ("manifest:code", "manifest:features", "manifest:resolved", "zip:name",
                "import:foreign_key_check", "order:features", "features:extraction_policy",
                "schema:model_input_columns"):
        assert checks[cid]["status"] == "pass", (cid, checks[cid])


# Legado: intervalo Node.sim_* que espelha uma SimulationRun não é "fallback sem SimulationRun"
@requires_head
def test_legacy_case_ignores_interval_mirroring_a_simulation_run(export_source, tmp_path):
    from sqlmodel import Session, create_engine
    from models import Node
    src = tmp_path / "origem.db"
    shutil.copy(export_source, src)
    eng = create_engine(f"sqlite:///{src}")
    with Session(eng) as s:
        mirrored, legacy = s.get(Node, 1), s.get(Node, 2)
        # nó 1: Node.sim_* = SimulationRun 11 (intervalo espelhado); nó 2: legado sem SimulationRun
        mirrored.sim_started_at = EXPORT_T0 + timedelta(minutes=30)
        mirrored.sim_ended_at = EXPORT_T0 + timedelta(hours=1)
        legacy.sim_started_at = EXPORT_T0 + timedelta(minutes=90)
        legacy.sim_ended_at = EXPORT_T0 + timedelta(minutes=100)
        s.add(mirrored)
        s.add(legacy)
        s.commit()
    eng.dispose()
    pkg = tmp_path / "p.zip"
    _export_with(src, pkg, export_kind="full")
    rc, checks = _run(pkg, src, tmp_path, NO_IMPORT)
    assert rc == 0, [c for c in checks.values() if c["status"] == "fail"]
    rows = _features_rows(pkg)
    only_mirrored = [r for r in rows if "legacy_node:1" in r["simulation_run_ids"]
                     and "legacy_node:2" not in r["simulation_run_ids"]]
    true_legacy = [r for r in rows if "legacy_node:2" in r["simulation_run_ids"]]
    assert only_mirrored and true_legacy
    assert checks["origin:spec_cases"]["observed"]["legacy"]["windows"] == len(true_legacy)


# model_input_columns ausente falha (não some)
def test_missing_model_input_columns_fails(export_source, full_package, tmp_path):
    edit = _json_edit("schema.json", lambda d: d["files"]["features.csv"].pop("model_input_columns"))
    rc, checks = _run(_tampered(full_package, tmp_path, edit), export_source, tmp_path, NO_IMPORT)
    assert rc == 1 and checks["schema:model_input_columns"]["status"] == "fail"


def _column(d, file_name, name):
    return next(c for c in d["files"][file_name]["columns"] if c["name"] == name)


# Schema coluna a coluna
@pytest.mark.parametrize("label,mutate", [
    ("sem type", lambda d: _column(d, "neighbors.csv", "snr").pop("type")),
    ("sem nullable", lambda d: _column(d, "telemetry.csv", "id").pop("nullable")),
    ("tipo desconhecido", lambda d: _column(d, "telemetry.csv", "rssi").update(type="intger")),
    ("integer → string", lambda d: _column(d, "neighbors.csv", "rssi").update(type="string")),
    ("nullable invertido", lambda d: _column(d, "telemetry.csv", "id").update(nullable=True)),
    ("nullable não booleano", lambda d: _column(d, "telemetry.csv", "id").update(nullable="no")),
    ("descrição só com espaços", lambda d: _column(d, "coverage.csv", "node_name").update(description="   ")),
    ("enum reduzido", lambda d: _column(d, "features.csv", "feature_status").update(enum=["valid"])),
    ("enum alterado", lambda d: _column(d, "features.csv", "origin_status").update(
        enum=["real", "synthetic", "mixed", "unknown", "no_data", "other"])),
    ("features integer → string", lambda d: _column(d, "features.csv", "node_id").update(type="string")),
])
def test_schema_column_checks(export_source, full_package, tmp_path, label, mutate):
    rc, checks = _run(_tampered(full_package, tmp_path, _json_edit("schema.json", mutate)), export_source,
                      tmp_path, NO_IMPORT)
    assert rc == 1 and checks["schema:complete"]["status"] == "fail", label


# Manifesto conferido campo a campo
@pytest.mark.parametrize("mutate,check_id", [
    (lambda d: d["features"]["status_counts"].update(valid=d["features"]["status_counts"]["valid"] + 5),
     "manifest:features"),
    (lambda d: d["features"].update(extraction_policy="real_only_v1"), "manifest:features"),
    (lambda d: d["features"].update(rows=d["features"]["rows"] + 1), "manifest:features"),
    (lambda d: d.update(networks=[1]), "manifest:networks"),
    (lambda d: d["resolved"].update(data_start="2026-08-31T23:59:00Z"), "manifest:resolved"),
    (lambda d: d["resolved"].update(grid_instants=999), "manifest:resolved"),
    (lambda d: d["slice_rows"].update(telemetry=d["slice_rows"]["telemetry"] + 1), "manifest:slice_rows"),
    (lambda d: d.update(labels_status="verified"), "manifest:labels_status"),
    (lambda d: d.update(limitations=["x"]), "manifest:limitations"),
    (lambda d: d.update(limitations=d["limitations"] + [""]), "manifest:limitations"),
])
def test_manifest_fields_checked(export_source, full_package, tmp_path, mutate, check_id):
    rc, checks = _run(_tampered(full_package, tmp_path, _json_edit("manifest.json", mutate)), export_source,
                      tmp_path, NO_IMPORT)
    assert rc == 1 and checks[check_id]["status"] == "fail"


# Grade recalculada pelo auditor, não por ExportRequest
def test_generator_grid_missing_one_instant_fails(export_source, tmp_path, monkeypatch):
    from dataset_export import request as rq
    grid, bounds = rq.ExportRequest.grid, rq.ExportRequest.grid_bounds

    def drop_first(self):
        return grid(self)[1:]

    def drop_first_bounds(self):
        n, first, last = bounds(self)
        return (n - 1, first + timedelta(seconds=self.step_seconds), last) if n > 1 else (0, None, None)

    monkeypatch.setattr(rq.ExportRequest, "grid", drop_first)
    monkeypatch.setattr(rq.ExportRequest, "grid_bounds", drop_first_bounds)
    pkg = tmp_path / "p.zip"
    _export_with(export_source, pkg, export_kind="full")
    rc, checks = _run(pkg, export_source, tmp_path, NO_IMPORT)   # gerador ainda alterado durante a auditoria
    assert rc == 1
    assert checks["features:grid"]["status"] == "fail" and checks["features:keys"]["status"] == "fail"
    assert checks["manifest:resolved"]["status"] == "fail"


# Interpretação e commit do código contra o esperado
@requires_head
def test_expect_interpretation_flags(export_source, full_package, tmp_path):
    ok = ["--expect-tracked-since", "2026-01-01T00:00:00+00:00", "--expect-margin", "0",
          "--expect-grid-origin", "1970-01-01T00:00:00Z"]
    rc, checks = _run(full_package, export_source, tmp_path, NO_IMPORT, *ok)
    assert rc == 0 and checks["request:matches_expected"]["status"] == "pass"
    for flags in (["--expect-tracked-since", "none"], ["--expect-margin", "30"],
                  ["--expect-grid-origin", "2026-01-01T00:00:00Z"]):
        rc, checks = _run(full_package, export_source, tmp_path, NO_IMPORT, *flags)
        assert rc == 1 and checks["request:matches_expected"]["status"] == "fail", flags


@requires_head
def test_expect_tracked_since_none_on_untracked_package(export_source, tmp_path):
    pkg = tmp_path / "p.zip"
    _export_with(export_source, pkg, settings=_settings(tracked=None), export_kind="features")
    rc, checks = _run(pkg, export_source, tmp_path, "--expect-tracked-since", "none")
    assert rc == 0 and checks["request:matches_expected"]["status"] == "pass"
    rc, checks = _run(pkg, export_source, tmp_path, "--expect-tracked-since", "2026-01-01T00:00:00Z")
    assert rc == 1 and checks["request:matches_expected"]["status"] == "fail"


@requires_head
@pytest.mark.parametrize("code,flags,expected_rc", [
    ({"commit": "0" * 40, "dirty": False}, [], 1),
    ("head-dirty", [], 1),
    ("head-dirty", ["--allow-dirty"], 0),
    ({"commit": "0" * 40, "dirty": False}, ["--allow-dirty"], 1),
])
def test_manifest_code_against_auditor_head(export_source, tmp_path, monkeypatch, code, flags, expected_rc):
    value = {"commit": _git_head(), "dirty": True} if code == "head-dirty" else code
    monkeypatch.setattr("dataset_export.package.code_revision", lambda: value)
    pkg = tmp_path / "p.zip"
    _export_with(export_source, pkg, export_kind="features")
    rc, checks = _run(pkg, export_source, tmp_path, *flags)
    assert rc == expected_rc
    assert checks["manifest:code"]["status"] == ("pass" if expected_rc == 0 else "fail")


# Margem: só janela excluída por simulation_overlap
@requires_head
def test_margin_case_requires_simulation_overlap_exclusion(export_source, tmp_path):
    import sqlite3
    src = tmp_path / "origem.db"
    shutil.copy(export_source, src)
    with sqlite3.connect(src) as c:
        c.execute("DELETE FROM neighbor WHERE node_id = 2")    # janelas do nó 2 sem features
    pkg = tmp_path / "p.zip"
    _export_with(src, pkg, settings=_settings(margin=45.0), export_kind="features")
    rc, checks = _run(pkg, src, tmp_path)
    assert rc == 0, [c for c in checks.values() if c["status"] == "fail"]
    rows = _features_rows(pkg)
    margin_invalid = [r for r in rows if r["node_id"] == "2" and r["simulation_overlap"] == "1"
                      and r["synthetic_record_count"] == "0"
                      and r["reference_exclusion_reason"] == "features_not_valid"]
    assert margin_invalid   # pré-condição: janelas inválidas que só a margem liga à simulação 11
    assert checks["origin:spec_cases"]["observed"]["margin_only"]["windows"] == 0


# Pacote adulterado gera relatório e rc=1, não erro de execução
def test_hash_mismatch_writes_report_and_fails(export_source, full_package, tmp_path):
    edit = _csv_edit("neighbors.csv", _set(0, "rssi", lambda v: "-1"))
    rc, checks = _run(_tampered(full_package, tmp_path, edit, fix_manifest=False), export_source, tmp_path,
                      NO_IMPORT)
    assert rc == 1
    assert checks["package:verify"]["status"] == "fail" and checks["manifest:hashes"]["status"] == "fail"


def test_dropped_column_writes_report_and_fails(export_source, full_package, tmp_path):
    def drop_hops(items):
        from dataset_export.codec import csv_writer, read_csv
        it = read_csv(io.StringIO(items["neighbors.csv"].decode(), newline=""))
        header = next(it)
        keep = [i for i, h in enumerate(header) if h != "hops"]
        buf = io.StringIO()
        writer = csv_writer(buf)
        writer.writerow([header[i] for i in keep])
        for r in it:
            writer.writerow([r[i] for i in keep])
        items["neighbors.csv"] = buf.getvalue().encode()
        schema = json.loads(items["schema.json"])
        cols = schema["files"]["neighbors.csv"]["columns"]
        schema["files"]["neighbors.csv"]["columns"] = [c for c in cols if c["name"] != "hops"]
        items["schema.json"] = json.dumps(schema).encode()
    rc, checks = _run(_tampered(full_package, tmp_path, drop_hops), export_source, tmp_path, NO_IMPORT)
    assert rc == 1 and checks["columns:neighbor"]["status"] == "fail"


# pandas × codec na leitura de nulos (evidência, não falha)
@requires_head
def test_pandas_nulls_evidence(export_source, full_package, tmp_path, baseline):
    rc, checks = baseline
    nulls = checks["pandas:nulls"]
    assert rc == 0 and nulls["status"] == "info"
    assert nulls["observed"]["telemetry.csv"]["temperature"]["codec_null"] > 0
    edit = _csv_edit("nodes.csv", _set(0, "device_model", lambda _v: ""))   # texto vazio ≠ nulo
    _, checks = _run(_tampered(full_package, tmp_path, edit), export_source, tmp_path, NO_IMPORT)
    assert checks["pandas:nulls"]["observed"]["nodes.csv"]["device_model"]["empty_text_as_nan"] == 1


# Reimportação e PRAGMA foreign_key_check
@requires_head
def test_import_foreign_key_check(export_source, full_package, tmp_path):
    edit = _csv_edit("networks.csv", lambda rows: [r for r in rows if r["record_role"] == "requested"])
    rc, checks = _run(_tampered(full_package, tmp_path, edit), export_source, tmp_path)
    assert rc == 1 and checks["import:foreign_key_check"]["status"] == "fail"
    rc, checks = _run(full_package, export_source, tmp_path, NO_IMPORT)
    assert rc == 0 and checks["import:foreign_key_check"]["status"] == "info"


# Nome da raiz e do arquivo
def test_zip_root_name_pattern(export_source, full_package, tmp_path):
    def rename(d):
        d.update(package_root="dataset_outro", dataset_id="dataset_outro")
    rc, checks = _run(_tampered(full_package, tmp_path, _json_edit("manifest.json", rename), root="dataset_outro"),
                      export_source, tmp_path, NO_IMPORT)
    assert rc == 1 and checks["zip:name"]["status"] == "fail"
    assert checks["zip:filename"]["status"] == "info"


@requires_head
def test_zip_filename_api_name_has_kind_prefix(export_source, full_package, tmp_path):
    """O nome padronizado do pacote é `<opção>_<dataset_id>.zip` (bruto_, feature_por_janela_, completo_)."""
    import shutil
    with zipfile.ZipFile(full_package) as z:
        root = z.namelist()[0].split("/", 1)[0]
    baixado = tmp_path / f"completo_{root}.zip"
    shutil.copy(full_package, baixado)
    rc, checks = _run(baixado, export_source, tmp_path, NO_IMPORT)
    assert rc == 0 and checks["zip:filename"]["status"] == "info"
    assert checks["zip:filename"]["observed"]["api_name"] == f"completo_{root}.zip"
    assert checks["zip:filename"]["observed"]["matches_api_name"] is True


# ─── README, campanha, commit e falhas por seção ──────────────────────────────────
def _readme_of(package):
    with zipfile.ZipFile(package) as z:
        return z.read(next(n for n in z.namelist() if n.endswith("/README.md")))


def _put(name, data):
    def edit(items):
        items[name] = data
    return edit


# README de outro pacote: da opção features, e de outro full (outro dataset_id e período)
def test_readme_of_features_package_fails(export_source, full_package, tmp_path):
    features = tmp_path / "features.zip"
    _export_with(export_source, features, export_kind="features")
    pkg = _tampered(full_package, tmp_path, _put("README.md", _readme_of(features)))
    rc, checks = _run(pkg, export_source, tmp_path, NO_IMPORT)
    assert rc == 1
    observed = checks["readme:identity"]["observed"]
    assert checks["readme:identity"]["status"] == "fail"
    assert {"title", "option"} <= set(observed["missing"])
    assert {"telemetry.csv", "neighbors.csv", "nodes.csv"} <= set(observed["csv_not_cited"])


def test_readme_of_other_full_package_fails(export_source, full_package, tmp_path):
    other = tmp_path / "other.zip"
    _export_with(export_source, other, export_kind="full", start=EXPORT_T0 + timedelta(minutes=30),
                 end=EXPORT_T0 + timedelta(hours=1, minutes=30))
    pkg = _tampered(full_package, tmp_path, _put("README.md", _readme_of(other)))
    rc, checks = _run(pkg, export_source, tmp_path, NO_IMPORT)
    assert rc == 1
    assert checks["readme:identity"]["status"] == "fail"
    assert set(checks["readme:identity"]["observed"]["missing"]) == {"title", "network_period"}
    assert checks["readme:no_absent_files"]["status"] == "pass"   # mesma opção: só a identidade difere


def test_readme_citing_csv_outside_the_option_fails(export_source, full_package, tmp_path):
    """Pacote features com telemetry.csv a mais no ZIP e no manifesto, citado no README: o CSV é do
    pacote, mas não da opção features."""
    pkg = tmp_path / "features.zip"
    _export_with(export_source, pkg, export_kind="features")
    with zipfile.ZipFile(full_package) as z:
        telemetry = z.read(next(n for n in z.namelist() if n.endswith("/telemetry.csv")))

    def add_telemetry(items):
        items["telemetry.csv"] = telemetry
        items["README.md"] += b"\nVer `telemetry.csv`.\n"
        manifest = json.loads(items["manifest.json"])
        manifest["files"]["telemetry.csv"] = {}
        manifest["tables"]["telemetry"] = {}
        items["manifest.json"] = json.dumps(manifest).encode()
    rc, checks = _run(_tampered(pkg, tmp_path, add_telemetry), export_source, tmp_path)
    assert rc == 1 and checks["readme:no_absent_files"]["observed"] == ["telemetry.csv"]
    assert checks["readme:identity"]["status"] == "pass"


# Campanha esperada
@requires_head
def test_expect_campaign_id(export_source, full_package, tmp_path):
    rc, checks = _run(full_package, export_source, tmp_path, NO_IMPORT, "--expect-campaign-id", "none")
    assert rc == 0 and checks["request:matches_expected"]["status"] == "pass"
    rc, checks = _run(full_package, export_source, tmp_path, NO_IMPORT, "--expect-campaign-id", "C1")
    assert rc == 1 and checks["request:matches_expected"]["status"] == "fail"
    pkg = _tampered(full_package, tmp_path, _json_edit("manifest.json", lambda d: d.update(campaign_id="C9")))
    rc, checks = _run(pkg, export_source, tmp_path, NO_IMPORT)            # sem a flag: como antes
    assert rc == 0 and checks["request:matches_expected"]["status"] == "info"
    rc, checks = _run(pkg, export_source, tmp_path, NO_IMPORT, "--expect-campaign-id", "none")
    assert rc == 1 and checks["request:matches_expected"]["observed"] == {"campaign_id": "C9"}


# source_hashes recalculados no commit esperado; source.file_sha256 = --sqlite
@requires_head
@pytest.mark.parametrize("mutate", [
    lambda d: d["source_hashes"].update({"backend/models.py": "0" * 64}),
    lambda d: d["source_hashes"].pop("backend/ai/feature_engineering.py"),
    lambda d: d.update(source_hashes={}),
])
def test_source_hashes_recomputed(export_source, full_package, tmp_path, mutate):
    rc, checks = _run(_tampered(full_package, tmp_path, _json_edit("manifest.json", mutate)), export_source,
                      tmp_path, NO_IMPORT)
    assert rc == 1 and checks["manifest:source_hashes"]["status"] == "fail"


@requires_head
def test_source_hashes_info_when_dirty(export_source, tmp_path, monkeypatch):
    monkeypatch.setattr("dataset_export.package.code_revision", lambda: {"commit": _git_head(), "dirty": True})
    pkg = tmp_path / "p.zip"
    _export_with(export_source, pkg, export_kind="features")
    rc, checks = _run(pkg, export_source, tmp_path, "--allow-dirty")
    assert rc == 0 and checks["manifest:source_hashes"]["status"] == "info"


@requires_head
def test_source_file_hash(export_source, full_package, tmp_path, baseline):
    assert baseline[1]["manifest:source_file"]["status"] == "info"     # exportação sem --sqlite: nulo
    digest = hashlib.sha256(Path(export_source).read_bytes()).hexdigest()
    for value, status in ((digest, "pass"), ("0" * 64, "fail")):
        edit = _json_edit("manifest.json", lambda d, v=value: d["source"].update(file_sha256=v))
        rc, checks = _run(_tampered(full_package, tmp_path, edit), export_source, tmp_path, NO_IMPORT)
        assert checks["manifest:source_file"]["status"] == status and rc == (0 if status == "pass" else 1)


# Commit esperado explícito (SHA curto resolvido pelo git)
@requires_parent_commit
def test_expect_commit(export_source, tmp_path, monkeypatch):
    previous = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=SCRIPTS.parent, capture_output=True,
                              text=True).stdout.strip()
    monkeypatch.setattr("dataset_export.package.code_revision", lambda: {"commit": previous, "dirty": False})
    monkeypatch.setattr("dataset_export.package.source_hashes", lambda: _hashes_at(previous))
    pkg = tmp_path / "p.zip"
    _export_with(export_source, pkg, export_kind="features")
    rc, checks = _run(pkg, export_source, tmp_path)
    assert rc == 1 and checks["manifest:code"]["status"] == "fail"                  # padrão: HEAD
    rc, checks = _run(pkg, export_source, tmp_path, "--expect-commit", previous[:9])
    assert rc == 0, [c for c in checks.values() if c["status"] == "fail"]
    assert checks["manifest:code"]["expected"]["commit"] == previous
    assert checks["manifest:source_hashes"]["status"] == "pass"
    rc, checks = _run(pkg, export_source, tmp_path, "--expect-commit", "f" * 12)   # não resolve: uso
    assert rc == 2 and checks == {}


# Exceção numa seção: falha com tipo, mensagem e fim do traceback
def test_section_error_reports_traceback(export_source, full_package, tmp_path):
    pkg = _tampered(full_package, tmp_path, _json_edit("manifest.json", lambda d: d["nodes"].pop("context")))
    rc, checks = _run(pkg, export_source, tmp_path, NO_IMPORT)
    error = checks["manifest:error"]
    assert rc == 1 and error["status"] == "fail" and error["description"] == "seção manifest interrompida"
    assert error["observed"]["type"] == "KeyError" and error["observed"]["message"] == "'context'"
    assert any("audit_manifest" in line for line in error["observed"]["traceback"])


# Pacote de comparação adulterado: COMPARE.json com ok=false e rc=1
def test_compare_with_tampered_package(export_source, full_package, tmp_path):
    edit = _csv_edit("neighbors.csv", _set(0, "rssi", lambda v: "-1"))
    bad = _tampered(full_package, tmp_path, edit, fix_manifest=False)
    out = tmp_path / "COMPARE.json"
    rc, _ = _run(full_package, export_source, tmp_path, NO_IMPORT, "--compare", bad, "--compare-out", out)
    compare = json.loads(out.read_text())
    assert rc == 1 and compare["ok"] is False and "verify_package" in compare["error"]


# Pacote que falhou em package:verify não executa os blocos do README
def test_readme_blocks_skipped_when_verify_fails(export_source, full_package, tmp_path):
    def plant(items):
        items["README.md"] += b"\n```python\nopen('PLANTADO', 'w').write('x')\n```\n"
    pkg = _tampered(full_package, tmp_path, plant, fix_manifest=False)
    rc, checks = _run(pkg, export_source, tmp_path, NO_IMPORT)
    blocks = checks["readme:python_blocks"]
    assert rc == 1 and checks["package:verify"]["status"] == "fail"
    assert blocks["status"] == "fail" and blocks["observed"]["skipped"] == "package:verify falhou"


# Origem com espaço, '#', '?' e acento no caminho (URI file: montada por Path.as_uri)
@requires_head
def test_source_path_with_special_characters(export_source, full_package, tmp_path):
    folder = tmp_path / "pasta com espaço #1 ?x"
    folder.mkdir()
    src = folder / "orígem.db"
    shutil.copy(export_source, src)
    rc, checks = _run(full_package, src, tmp_path, NO_IMPORT)
    assert rc == 0 and checks["ids:telemetry"]["status"] == "pass"
