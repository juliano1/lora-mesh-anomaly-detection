"""Documentação embutida no pacote: README por opção e schema completo."""
import json
import zipfile
from datetime import datetime
from types import SimpleNamespace

import pytest

from tests.conftest import export_request

SETTINGS = SimpleNamespace(DATA_ORIGIN_TRACKED_SINCE=None, ML_REFERENCE_MARGIN_MINUTES=0.0,
                           model_fields_set=set())
NOW = datetime(2026, 10, 21, 12, 0)


@pytest.fixture(scope="module")
def packages(export_source, tmp_path_factory):
    from dataset_export.package import export_dataset
    from dataset_export.snapshot import build_source_engine
    out = {}
    for kind in ("raw", "features", "full"):
        path = tmp_path_factory.mktemp("docs") / f"{kind}.zip"
        eng = build_source_engine(f"sqlite:///{export_source}")
        m = export_dataset(eng, export_request(export_kind=kind), path, settings=SETTINGS, now=NOW)
        eng.dispose()
        out[kind] = (path, m)
    return out


def _read(packages, kind, name):
    path, m = packages[kind]
    with zipfile.ZipFile(path) as z:
        return z.read(f"{m['package_root']}/{name}").decode()


def test_readme_raw_does_not_describe_features(packages):
    readme = _read(packages, "raw", "README.md")
    assert "telemetry.csv" in readme and "import_dataset.py" in readme
    assert "features.csv" not in readme and "fit_predict" not in readme


def test_readme_features_does_not_describe_raw_tables(packages):
    readme = _read(packages, "features", "README.md")
    assert "features.csv" in readme and "coverage.csv" in readme
    assert "telemetry.csv" not in readme and "import_dataset.py" not in readme


def test_readme_full_describes_both(packages):
    readme = _read(packages, "full", "README.md")
    assert "telemetry.csv" in readme and "features.csv" in readme and "import_dataset.py" in readme


@pytest.mark.parametrize("kind", ["features", "full"])
def test_readme_explains_valid_vs_defaults(packages, kind):
    """D4 (features v4): valid = 11 features observadas; a não observada deixa a janela fora."""
    readme = _read(packages, kind, "README.md")
    assert "unavailable_features" in readme and "defaulted_features" in readme
    assert "11 features observadas e finitas" in readme and "insufficient_data" in readme


@pytest.mark.parametrize("kind", ["raw", "features", "full"])
def test_readme_points_to_schema_not_to_missing_manual(packages, kind):
    """O manual de dados não acompanha o pacote: as descrições vêm do schema.json."""
    readme = _read(packages, kind, "README.md")
    schema = _read(packages, kind, "schema.json")
    assert "manual_dados" not in readme and "manual_dados" not in schema
    assert "`schema.json`" in readme


@pytest.mark.parametrize("kind", ["raw", "features", "full"])
def test_schema_covers_every_csv_with_descriptions(packages, kind):
    """Todo CSV do pacote no schema, toda coluna com descrição não vazia."""
    path, m = packages[kind]
    schema = json.loads(_read(packages, kind, "schema.json"))
    csvs = [n for n in m["files"] if n.endswith(".csv")]
    assert set(csvs) <= set(schema["files"]), set(csvs) - set(schema["files"])
    with zipfile.ZipFile(path) as z:
        for name in csvs:
            header = z.read(f"{m['package_root']}/{name}").decode().splitlines()[0]
            header = [h.strip('"') for h in header.split(",")]
            cols = {c["name"]: c for c in schema["files"][name]["columns"]}
            assert list(cols) == header, name
            empty = [n for n, c in cols.items() if not c.get("description")]
            assert not empty, (name, empty)
            assert all("type" in c and "nullable" in c for c in cols.values()), name


def test_schema_feature_status_valid_text(packages):
    schema = json.loads(_read(packages, "features", "schema.json"))
    cols = {c["name"]: c for c in schema["files"]["features.csv"]["columns"]}
    assert "unavailable_features" in cols["feature_status"]["description"]


def _col_desc(packages, kind, file_name, column):
    schema = json.loads(_read(packages, kind, "schema.json"))
    cols = {c["name"]: c for c in schema["files"][file_name]["columns"]}
    return cols[column]["description"]


def test_full_readme_files_list_precedes_sections(packages):
    """'## Arquivos' lista todos os arquivos do pacote antes das seções."""
    readme = _read(packages, "full", "README.md")
    files_section = readme.split("## Arquivos", 1)[1].split("\n## ", 1)[0]
    for name in ("telemetry.csv", "features.csv", "coverage.csv", "events_template.csv"):
        assert name in files_section, name
    assert readme.index("coverage.csv") < readme.index("## Leitura")


def test_neighbors_timestamp_is_backend_reception(packages):
    assert "recepção no backend" in _col_desc(packages, "raw", "neighbors.csv", "timestamp")


def test_telemetry_battery_level_documents_markers(packages):
    text = _col_desc(packages, "raw", "telemetry.csv", "battery_level")
    assert "−1" in text and "−2" in text


def test_coverage_windows_with_defaults_counts_unobserved_windows(packages):
    # Features v4: janela com feature não observada nunca é válida; a contagem cobre todas.
    assert "nenhuma delas é válida" in _col_desc(packages, "features", "coverage.csv", "windows_with_defaults")


def test_simulation_runs_telemetry_count_is_synthetic(packages):
    assert "sintéticas" in _col_desc(packages, "raw", "simulation_runs.csv", "telemetry_count")


def test_raw_descriptions_have_no_internal_prefixes_or_english(packages):
    schema = json.loads(_read(packages, "full", "schema.json"))
    banned = ("FEAT-", "SPEC-004", "SPEC-005", "True if", "UTC timestamp", "Simulated")
    for name in ("telemetry.csv", "neighbors.csv", "nodes.csv", "networks.csv", "simulation_runs.csv"):
        for col in schema["files"][name]["columns"]:
            assert not any(b in col["description"] for b in banned), (name, col["name"])


def test_raw_column_text_keys_are_real_columns():
    """Guarda contra chaves obsoletas: todo (arquivo, coluna) de RAW_COLUMN_TEXT existe."""
    from dataset_export.package import RAW_COLUMN_TEXT, build_schema
    files = build_schema()["files"]
    stale = [k for k in RAW_COLUMN_TEXT
             if k[0] not in files or k[1] not in {c["name"] for c in files[k[0]]["columns"]}]
    assert not stale, stale


# Manual de dados do README: textos conferidos contra o firmware e o pandas 2.3.
def _python_blocks(readme):
    import re
    return re.findall(r"```python\n(.*?)```", readme, flags=re.S)


@pytest.mark.parametrize("kind", ["raw", "features", "full"])
def test_readme_examples_parse_dates_with_iso8601(packages, kind):
    """pandas 2.3: parse_dates sem date_format deixa como texto uma coluna com instantes sem fração."""
    lines = [ln for block in _python_blocks(_read(packages, kind, "README.md"))
             for ln in block.splitlines() if "parse_dates" in ln]
    assert lines, kind
    assert all('date_format="ISO8601"' in ln for ln in lines), lines


def test_gw_rssi_text_is_gateway_hello_not_last_hop(packages):
    """gw_rssi/gw_snr vêm da tabela de rotas do gateway (último Hello direto), não do pacote recebido."""
    for column in ("rssi", "payload_raw"):
        text = _col_desc(packages, "raw", "telemetry.csv", column)
        assert "diretamente" in text and "gw_rssi" in text, column
        assert "último salto" not in text and "valor do pacote" not in text, column


def test_parent_texts_do_not_claim_next_hop(packages):
    """O parent é o primeiro vizinho direto da tabela de rotas, não o próximo salto até o gateway."""
    for file_name, column in (("neighbors.csv", "is_parent"), ("telemetry.csv", "parent_id")):
        text = _col_desc(packages, "raw", file_name, column)
        assert "não necessariamente o próximo salto" in text, (file_name, column)
        assert "encaminhou" not in text and "rota ativa" not in text, (file_name, column)
    quality = _col_desc(packages, "raw", "neighbors.csv", "link_quality")
    assert "heurístico" in quality and "RSSI" in quality and "PDR físico" not in quality


@pytest.mark.parametrize("kind", ["raw", "full"])
def test_record_role_text_does_not_put_every_reverse_link_in_context(packages, kind):
    """Enlaces reversos relatados por nós pedidos dentro do período são `requested`."""
    schema = json.loads(_read(packages, kind, "schema.json"))
    for name in ("telemetry.csv", "neighbors.csv", "nodes.csv", "networks.csv", "simulation_runs.csv"):
        cols = {c["name"]: c for c in schema["files"][name]["columns"]}
        text = cols["record_role"]["description"]
        assert "relatado por nó não pedido" in text and "enlaces reversos que eles relatam" in text, name
    files_section = _read(packages, kind, "README.md").split("## Arquivos", 1)[1].split("\n## ", 1)[0]
    assert "relatados por nós não" in files_section and "`requested`" in files_section
