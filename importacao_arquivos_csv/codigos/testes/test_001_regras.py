"""Detector 001 — corte, partições, seleção da referência e modelo (dados sintéticos)."""
import numpy as np
import pandas as pd
import pytest

import detector_001_isolation_forest_sklearn as det
from dados_sinteticos_001 import (CORTE, invalidate, make_rows, position, read_outputs, run_detector,
                                  write_package)


def run(tmp_path, rows, *args):
    code, folders = run_detector(tmp_path, write_package(tmp_path / "p", rows), "--corte-treino", CORTE, *args)
    assert code == 0 and len(folders) == 1
    return read_outputs(folders[0])


def test_particoes_nas_fronteiras_do_corte_e_da_folga():
    corte = pd.Timestamp("2026-01-01T04:00:00Z")
    fins = pd.Series(pd.to_datetime(["2026-01-01T03:59:59Z", "2026-01-01T04:00:00Z", "2026-01-01T04:00:01Z",
                                     "2026-01-01T04:15:00Z", "2026-01-01T04:15:01Z"], utc=True))
    assert det.assign_partitions(fins, corte, 900).tolist() == [
        "treino_candidato", "treino_candidato", "folga", "folga", "teste"]


def test_corte_automatico_quantil_dos_fins_validos_arredondado_ao_minuto(tmp_path):
    rows = make_rows(nodes=(1, 2), n=20)
    for row in rows[-8:]:
        invalidate(row)
    data = det.load_dataset(write_package(tmp_path / "p", rows)).data
    validos = data.loc[data["feature_status"] == "valid", "window_end_utc"]
    esperado = validos.quantile(0.7).floor("min")
    assert det.automatic_cut(data, 0.7) == esperado
    assert esperado != data["window_end_utc"].quantile(0.7).floor("min")   # inválidas não contam


def test_corte_explicito_exige_fuso_e_converte_para_utc():
    assert det.parse_utc_instant("2026-01-01T01:00:00-03:00", "--corte-treino") == pd.Timestamp(CORTE)
    with pytest.raises(det.InputError, match="fuso"):
        det.parse_utc_instant("2026-01-01T04:00:00", "--corte-treino")


def test_decisao_usa_comparacao_estrita_com_o_limiar():
    assert det.decide(np.array([0.4, 0.5, 0.6]), 0.5).tolist() == [False, False, True]


def test_pipeline_ajustado_so_no_treino_e_score_maior_para_ponto_isolado():
    rng = np.random.default_rng(0)
    treino = rng.normal(size=(300, 11))
    teste = np.vstack([np.zeros(11), np.full(11, 8.0)])
    r = det.fit_and_score(treino, teste, n_estimators=50, seed=1, percentile=99)
    assert r["test_scores"][1] > r["test_scores"][0]                      # maior = mais anômalo
    np.testing.assert_allclose(r["pipeline"][0].center_, np.median(treino, axis=0))
    np.testing.assert_allclose(r["train_scores"], -r["pipeline"].score_samples(treino))
    assert r["threshold"] == pytest.approx(np.percentile(r["train_scores"], 99))
    assert r["is_outlier"].tolist() == (r["test_scores"] > r["threshold"]).tolist()
    outro = det.fit_and_score(treino, teste * 3 + 100, n_estimators=50, seed=1, percentile=99)
    assert outro["threshold"] == r["threshold"]                           # o teste não mexe no ajuste
    np.testing.assert_array_equal(outro["train_scores"], r["train_scores"])


def test_scores_nao_dependem_da_ordem_das_linhas_no_arquivo(tmp_path):
    rows = make_rows(n=150)                # treino com 363 linhas: o Isolation Forest subamostra 256
    shuffled = [rows[i] for i in np.random.default_rng(1).permutation(len(rows))]
    scores = []
    for nome, linhas in (("ordem", rows), ("embaralhado", shuffled)):
        code, folders = run_detector(tmp_path / nome, write_package(tmp_path / nome / "p", linhas),
                                     "--corte-treino", "2026-01-01T14:07:00Z")
        assert code == 0
        p = read_outputs(folders[0])["predicoes"]
        scores.append(p.set_index(["node_id", "window_end_utc"])["anomaly_score"].sort_index())
    pd.testing.assert_series_equal(scores[0], scores[1])


def test_linhas_invalidas_e_com_default_ficam_nas_predicoes_sem_score(tmp_path):
    rows = make_rows()
    invalidate(rows[position(0, 5)])                                       # no treino
    rows[position(1, 40)].update(defaulted_features='["link_asymmetry"]', link_asymmetry="0.0",
                                 unavailable_features='{"link_asymmetry": "no_reverse_link"}')
    rows[position(2, 45)]["rssi_std"] = "nan"                              # no teste
    p = run(tmp_path, rows)["predicoes"]
    assert len(p) == len(rows)
    assert p["numero_linha_original"].tolist() == [str(i) for i in range(1, len(rows) + 1)]
    assert p["rssi_mean"].fillna("").tolist() == [r["rssi_mean"] or "" for r in rows]   # texto original
    casos = {position(0, 5): ("excluido_treino", "treino:estado_insufficient_data"),
             position(1, 40): ("excluido_teste", "teste:default_declarado"),
             position(2, 45): ("excluido_teste", "teste:feature_nao_finita")}
    for i, (estado, motivo) in casos.items():
        linha = p.iloc[i]
        assert linha["estado_analise"] == estado
        assert motivo in linha["motivos_exclusao"].split(";")
        assert pd.isna(linha["anomaly_score"]) and pd.isna(linha["is_outlier"])


def test_simulacao_fica_fora_do_treino_e_e_analisada_no_teste(tmp_path):
    rows = make_rows()
    for k in (10, 50):
        rows[position(0, k)].update(simulation_overlap="1", origin_status="mixed",
                                    simulation_run_ids='[7, "legacy_node:1"]',
                                    reference_eligible_by_policy="0", reference_exclusion_reason="synthetic_rows")
    saidas = run(tmp_path, rows)
    p = saidas["predicoes"]
    treino = p.iloc[position(0, 10)]
    assert treino["estado_analise"] == "excluido_treino"
    assert {"treino:simulacao_sobreposta", "treino:nao_elegivel_politica",
            "treino:origem_mixed"} <= set(treino["motivos_exclusao"].split(";"))
    teste = p.iloc[position(0, 50)]
    assert teste["estado_analise"] == "teste_analisado"
    assert not pd.isna(teste["anomaly_score"]) and teste["is_outlier"] in {"0", "1"}
    sim = saidas["execucao"]["simulacoes"]
    assert sim["teste_analisado"] == 1
    assert sim["teste_analisado_execucoes"] == ["7"]                 # id de simulation_runs.csv
    assert sim["teste_analisado_marcadores_legados"] == ["legacy_node:1"]
    assert sim["treino_candidato"] == 1
    assert "1 execução" in saidas["relatorio_md"]


def test_trechos_contiguos_de_janelas_na_grade():
    fins = pd.Series(pd.to_datetime(["2026-01-01T00:07:00Z", "2026-01-01T00:14:00Z", "2026-01-01T00:35:00Z",
                                     "2026-01-01T00:21:00Z"], utc=True))
    assert det.count_runs(fins, 420) == 2                 # 00:07–00:21 e 00:35


def test_origem_unknown_entra_no_treino_com_ressalva_e_synthetic_nao(tmp_path):
    rows = make_rows()                                      # unknown, como no pacote de referência
    rows[position(1, 3)]["origin_status"] = "synthetic"
    rows[position(1, 4)]["origin_status"] = "real"
    saidas = run(tmp_path, rows)
    p = saidas["predicoes"]
    assert p.iloc[position(0, 3)]["estado_analise"] == "treino_efetivo"
    assert p.iloc[position(1, 4)]["estado_analise"] == "treino_efetivo"
    assert "treino:origem_synthetic" in p.iloc[position(1, 3)]["motivos_exclusao"].split(";")
    assert any("unknown" in r for r in saidas["execucao"]["ressalvas"])


def test_treino_com_menos_de_24_horas_gera_ressalva_de_ciclo_diario(tmp_path):
    ressalvas = run(tmp_path, make_rows())["execucao"]["ressalvas"]   # treino de 23:52 a 04:00
    assert any("cobre 4,1 h" in r and "ciclo diário" in r for r in ressalvas)


def test_corte_unico_para_todos_os_nos_sem_sobreposicao(tmp_path):
    saidas = run(tmp_path, make_rows())
    p, ex = saidas["predicoes"], saidas["execucao"]
    corte = pd.Timestamp(CORTE)
    fim = pd.to_datetime(p["window_end_utc"], utc=True)
    inicio = pd.to_datetime(p["window_start_utc"], utc=True)
    treino, teste = p["particao_temporal"] == "treino_candidato", p["particao_temporal"] == "teste"
    assert (fim[treino] <= corte).all() and (inicio[teste] > corte).all()
    assert p[treino].groupby("node_id")["window_end_utc"].max().nunique() == 1
    assert (p["particao_temporal"] == "folga").sum() == 6                 # 2 janelas por nó
    assert ex["corte"] == {**ex["corte"], "instante_utc": CORTE, "modo": "explicito"}
    assert ex["folga"]["segundos"] == 900
    assert ex["verificacao_sobreposicao"]["sem_sobreposicao"] is True


def test_inicio_treino_exclui_janelas_que_comecam_antes(tmp_path):
    p = run(tmp_path, make_rows(), "--inicio-treino", "2026-01-01T01:00:00Z")["predicoes"]
    inicio = pd.to_datetime(p["window_start_utc"], utc=True)
    antes = (p["particao_temporal"] == "treino_candidato") & (inicio < pd.Timestamp("2026-01-01T01:00:00Z"))
    assert antes.any()
    assert (p.loc[antes, "estado_analise"] == "excluido_treino").all()
    assert p.loc[antes, "motivos_exclusao"].str.contains("treino:antes_inicio_referencia", regex=False).all()
    assert (p.loc[~antes & (p["particao_temporal"] == "treino_candidato"), "estado_analise"] == "treino_efetivo").all()
