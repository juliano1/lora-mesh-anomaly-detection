"""Detector 001 — leitura, contrato e proveniência da entrada (dados sintéticos)."""
import json

import pandas as pd
import pytest

import detector_001_isolation_forest_sklearn as det
from dados_sinteticos_001 import COLUMNS, FEATURES, make_rows, sha256, write_csv, write_package


def load(tmp_path, rows, **kwargs):
    return det.load_dataset(write_package(tmp_path / "p", rows, **kwargs))


def test_le_datas_utc_com_fracao_de_segundo_variavel(tmp_path):
    rows = make_rows(nodes=(1,), n=3)
    rows[1].update(window_start_utc="2026-01-01T00:59:00.5Z", window_end_utc="2026-01-01T01:14:00.5Z")
    rows[2].update(window_start_utc="2026-01-01T01:06:00.123456Z",
                   window_end_utc="2026-01-01T01:21:00.123456+00:00")
    fim = load(tmp_path, rows).data["window_end_utc"]
    assert isinstance(fim.dtype, pd.DatetimeTZDtype) and str(fim.dt.tz) == "UTC"
    assert fim.iloc[1] == pd.Timestamp("2026-01-01T01:14:00.5Z")
    assert fim.iloc[2] == pd.Timestamp("2026-01-01T01:21:00.123456Z")


def test_le_json_e_booleanos_como_o_exportador(tmp_path):
    rows = make_rows(nodes=(1,), n=2)
    rows[1].update(defaulted_features='["link_asymmetry"]',
                   unavailable_features='{"link_asymmetry": "no_reverse_link"}',
                   simulation_overlap="1", reference_eligible_by_policy="0")
    data = load(tmp_path, rows).data
    assert data["defaulted_features"].tolist() == [[], ["link_asymmetry"]]
    assert data["unavailable_features"].tolist() == [{}, {"link_asymmetry": "no_reverse_link"}]
    assert data["simulation_overlap"].tolist() == [0, 1]
    assert data["reference_eligible_by_policy"].tolist() == [1, 0]


@pytest.mark.parametrize("valor", ["true", "False", "2", "1.0", " 1"])
def test_rejeita_booleano_fora_do_contrato(tmp_path, valor):
    rows = make_rows(nodes=(1,), n=2)
    rows[1]["simulation_overlap"] = valor
    with pytest.raises(det.InputError, match="simulation_overlap"):
        load(tmp_path, rows)


@pytest.mark.parametrize("fim", ["2026-01-01T00:14:00", "2026-01-01T00:14:00+03:00", "01/01/2026 00:14"])
def test_rejeita_data_sem_utc_explicito(tmp_path, fim):
    rows = make_rows(nodes=(1,), n=2)
    rows[1]["window_end_utc"] = fim
    with pytest.raises(det.InputError, match="window_end_utc"):
        load(tmp_path, rows)


@pytest.mark.parametrize("coluna, valor", [
    ("unavailable_features", "{link_asymmetry"),   # JSON malformado
    ("defaulted_features", "{}"),                  # objeto no lugar de lista
    ("unavailable_features", "[]"),                # lista no lugar de objeto
])
def test_rejeita_json_malformado_ou_de_tipo_errado(tmp_path, coluna, valor):
    rows = make_rows(nodes=(1,), n=2)
    rows[1][coluna] = valor
    with pytest.raises(det.InputError, match=coluna):
        load(tmp_path, rows)


def test_rejeita_data_inexistente_no_calendario(tmp_path):
    rows = make_rows(nodes=(1,), n=2)
    rows[1].update(window_start_utc="2026-02-29T23:45:00Z", window_end_utc="2026-02-30T00:00:00Z")
    with pytest.raises(det.InputError, match="window_start_utc"):
        load(tmp_path, rows)


def test_rejeita_node_id_fora_do_intervalo_inteiro(tmp_path):
    rows = make_rows(nodes=(1,), n=2)
    rows[1]["node_id"] = "99999999999999999999"
    with pytest.raises(det.InputError, match="node_id"):
        load(tmp_path, rows)


def test_rejeita_linha_com_numero_de_campos_diferente_do_cabecalho(tmp_path):
    pasta = tmp_path / "isolado"
    pasta.mkdir()
    write_csv(pasta / "features.csv", make_rows(nodes=(1,), n=3))
    linhas = (pasta / "features.csv").read_bytes().split(b"\r\n")
    linhas[2] = b",".join(linhas[2].split(b",")[:-5])        # 2ª linha de dados sem os 5 últimos campos
    (pasta / "features.csv").write_bytes(b"\r\n".join(linhas))
    with pytest.raises(det.InputError, match="campos"):
        det.load_dataset(pasta / "features.csv")


def test_rejeita_feature_nao_numerica(tmp_path):
    rows = make_rows(nodes=(1,), n=2)
    rows[0]["rssi_mean"] = "-70,5"   # vírgula decimal, como sai de uma planilha
    with pytest.raises(det.InputError, match="rssi_mean"):
        load(tmp_path, rows)


def test_rejeita_coluna_obrigatoria_ausente(tmp_path):
    colunas = [c for c in COLUMNS if c != "origin_status"]
    with pytest.raises(det.InputError, match="origin_status") as exc:
        load(tmp_path, make_rows(nodes=(1,), n=2), columns=colunas)
    assert "Features por janela" in str(exc.value)


def test_rejeita_schema_com_contrato_de_features_diferente(tmp_path):
    with pytest.raises(det.InputError, match="model_input_columns"):
        load(tmp_path, make_rows(nodes=(1,), n=2), model_inputs=FEATURES[::-1])


def test_rejeita_features_csv_alterado_depois_do_manifesto(tmp_path):
    pasta = write_package(tmp_path / "p", make_rows(nodes=(1,), n=2))
    csv_path = pasta / "features.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace('"valid"', '"error"', 1),
                        encoding="utf-8", newline="")
    with pytest.raises(det.InputError, match="SHA-256"):
        det.load_dataset(pasta)


def test_rejeita_schema_alterado_depois_do_manifesto(tmp_path):
    pasta = write_package(tmp_path / "p", make_rows(nodes=(1,), n=2))
    schema = json.loads((pasta / "schema.json").read_text(encoding="utf-8"))
    schema["files"]["features.csv"]["description"] = "editado"
    (pasta / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(det.InputError, match="schema.json"):
        det.load_dataset(pasta)


def test_rejeita_manifesto_ilegivel(tmp_path):
    pasta = write_package(tmp_path / "p", make_rows(nodes=(1,), n=2))
    (pasta / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(det.InputError, match="manifest.json"):
        det.load_dataset(pasta)


def test_rejeita_manifesto_sem_hash_do_features_csv(tmp_path):
    pasta = write_package(tmp_path / "p", make_rows(nodes=(1,), n=2))
    manifest = json.loads((pasta / "manifest.json").read_text(encoding="utf-8"))
    del manifest["files"]["features.csv"]
    (pasta / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(det.InputError, match="manifest.json"):
        det.load_dataset(pasta)


def test_rejeita_chaves_repetidas(tmp_path):
    rows = make_rows(nodes=(1,), n=3)
    rows.append(dict(rows[0]))
    with pytest.raises(det.InputError, match="repetid"):
        load(tmp_path, rows)


def test_rejeita_duracao_diferente_de_window_seconds(tmp_path):
    rows = make_rows(nodes=(1,), n=3)
    rows[1]["window_start_utc"] = "2026-01-01T00:00:00Z"   # fim 00:14: 14 min, não 15
    with pytest.raises(det.InputError, match="window_seconds"):
        load(tmp_path, rows)


def test_rejeita_mais_de_uma_fonte(tmp_path):
    rows = make_rows(nodes=(1,), n=3)
    rows[1]["source_id"] = "outra-fonte"
    with pytest.raises(det.InputError, match="source_id"):
        load(tmp_path, rows)


def test_rejeita_mais_de_uma_configuracao_de_passo(tmp_path):
    rows = make_rows(nodes=(1,), n=3)
    rows[1]["step_seconds"] = "300"
    with pytest.raises(det.InputError, match="step_seconds"):
        load(tmp_path, rows)


def test_pacote_completo_registra_identidade_hashes_e_versoes(tmp_path):
    pasta = write_package(tmp_path / "p", make_rows(nodes=(1,), n=2), dataset_id="dataset_teste_1")
    ds = det.load_dataset(pasta)
    assert ds.dataset_id == "dataset_teste_1"
    assert ds.sha256 == sha256(pasta / "features.csv")
    assert ds.provenance["manifest"]["sha256"] == sha256(pasta / "manifest.json")
    assert ds.provenance["schema"]["sha256"] == sha256(pasta / "schema.json")
    assert ds.provenance["versoes"]["extractor_revision"] == "sha256:teste"
    assert ds.provenance["situacao"].startswith("completa")


def test_csv_isolado_recebe_id_derivado_e_proveniencia_limitada(tmp_path):
    pasta = tmp_path / "isolado"
    pasta.mkdir()
    write_csv(pasta / "features.csv", make_rows(nodes=(1,), n=2))
    ds = det.load_dataset(pasta / "features.csv")
    assert ds.dataset_id == f"csv_{sha256(pasta / 'features.csv')[:12]}"
    assert ds.provenance["situacao"] == "proveniência limitada: manifesto/schema ausentes"
    assert ds.provenance["manifest"] is None and ds.provenance["schema"] is None
    assert ds.provenance["versoes"]["extractor_revision"] == "sha256:teste"   # lida do próprio CSV
    assert set(ds.provenance["versoes_fonte"].values()) == {"features.csv"}


def test_fonte_de_cada_versao_e_registrada_campo_a_campo(tmp_path):
    pasta = write_package(tmp_path / "p", make_rows(nodes=(1,), n=2))
    manifest = json.loads((pasta / "manifest.json").read_text(encoding="utf-8"))
    del manifest["features"]["extractor_revision"]
    (pasta / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    prov = det.load_dataset(pasta).provenance
    assert prov["versoes"]["extractor_revision"] == "sha256:teste"
    assert prov["versoes_fonte"]["extractor_revision"] == "features.csv"
    assert prov["versoes_fonte"]["feature_schema_version"] == "manifest.json"


def test_pasta_externa_do_zip_indica_a_pasta_interna(tmp_path):
    externa = tmp_path / "entrada"
    write_package(externa / "dataset_x", make_rows(nodes=(1,), n=2))
    with pytest.raises(det.InputError, match="dataset_x"):
        det.load_dataset(externa)


def test_pacote_bruto_explica_a_exportacao_necessaria(tmp_path):
    pasta = tmp_path / "bruto"
    pasta.mkdir()
    (pasta / "telemetry.csv").write_text('"id"\r\n', encoding="utf-8")
    (pasta / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(det.InputError, match="Features por janela"):
        det.load_dataset(pasta)


@pytest.mark.parametrize("nome", ["telemetry.csv", "neighbors.csv", "coverage.csv"])
def test_outros_csvs_do_pacote_explicam_a_exportacao_necessaria(tmp_path, nome):
    arquivo = tmp_path / nome
    arquivo.write_text('"id"\r\n', encoding="utf-8")
    with pytest.raises(det.InputError, match="features.csv"):
        det.load_dataset(arquivo)
