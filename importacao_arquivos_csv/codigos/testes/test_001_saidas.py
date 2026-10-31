"""Detector 001 — arquivos da execução, relatório e estados de encerramento (dados sintéticos)."""
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import detector_001_isolation_forest_sklearn as det
from dados_sinteticos_001 import (CORTE, FEATURES, center, invalidate, make_rows, position, read_outputs,
                                  run_detector, sha256, write_package)

ARQUIVOS = ("relatorio__{}.html", "relatorio__{}.md", "predicoes__{}.csv", "outliers__{}.csv",
            "resumo_por_dispositivo__{}.csv", "execucao__{}.json", "ambiente__{}.txt",
            "graficos/serie_scores__{}.png", "graficos/distribuicao_scores__{}.png",
            "graficos/outliers_por_dispositivo__{}.png")


def run(tmp_path, rows, *args, expected=0, **kwargs):
    code, folders = run_detector(tmp_path, write_package(tmp_path / "p", rows, **kwargs), *args)
    assert code == expected and len(folders) == 1
    return folders[0], read_outputs(folders[0])


def test_execucao_gera_todas_as_saidas_identificadas_pelo_run_id(tmp_path):
    folder, s = run(tmp_path, make_rows(), "--corte-treino", CORTE, dataset_id="dataset_teste_1")
    run_id = folder.name
    assert re.fullmatch(r"detector_001_isolation_forest_sklearn__dataset_teste_1__\d{8}T\d{6}Z__[0-9a-f]{8}", run_id)
    for nome in ARQUIVOS:
        assert (folder / nome.format(run_id)).is_file(), nome
    ex = s["execucao"]
    assert ex["estado"] == "concluida" and ex["run_id"] == run_id
    assert ex["entrada"]["sha256_csv"] == sha256(tmp_path / "p" / "features.csv")
    assert ex["detector"]["script_sha256"] == sha256(Path(det.__file__))
    assert ex["features"] == FEATURES
    assert ex["limiar"]["comparacao"] == "anomaly_score > limiar"
    assert ex["arquivos"][f"predicoes__{run_id}.csv"]["sha256"] == sha256(folder / f"predicoes__{run_id}.csv")
    assert "Por construção" in s["relatorio_md"]          # a taxa do treino acima do limiar vem do percentil
    assert "não foi ajustado depois de observar o teste" not in s["relatorio_md"]


def test_simulation_overlap_nulo_no_teste_nao_interrompe_o_relatorio(tmp_path):
    rows = make_rows()
    for k in range(36, 60):
        rows[position(0, k)]["simulation_overlap"] = None
    folder, s = run(tmp_path, rows, "--corte-treino", CORTE)
    assert s["execucao"]["estado"] == "concluida" and s["relatorio_html"] is not None
    assert s["execucao"]["contagens"]["simulation_overlap"]["nulo"] == 24


def test_saida_que_e_um_arquivo_encerra_com_erro_de_uso(tmp_path, capsys):
    pacote = write_package(tmp_path / "p", make_rows(nodes=(1,), n=5))
    arquivo = tmp_path / "nao_e_pasta"
    arquivo.write_text("x", encoding="utf-8")
    code = det.main(["--entrada", str(pacote), "--saida", str(arquivo)])
    assert code == 2 and "erro" in capsys.readouterr().err
    assert arquivo.read_text(encoding="utf-8") == "x"


def test_falha_ao_gravar_o_relatorio_nao_deixa_relatorio_de_execucao_concluida(tmp_path, monkeypatch):
    def falha(self, title):
        raise OSError("disco cheio simulado")
    monkeypatch.setattr(det.Report, "html", falha)
    folder, s = run(tmp_path, make_rows(), "--corte-treino", CORTE, expected=1)
    assert s["execucao"]["estado"] == "falhou" and "disco cheio" in s["execucao"]["motivo_estado"]
    assert not list(folder.glob("relatorio__*"))


def test_scores_e_decisoes_reproduziveis(tmp_path):
    pacote = write_package(tmp_path / "p", make_rows())
    saidas = []
    for nome in ("a", "b"):
        code, folders = run_detector(tmp_path / nome, pacote, "--corte-treino", CORTE)
        assert code == 0
        saidas.append(read_outputs(folders[0]))
    colunas = ["estado_analise", "anomaly_score", "limiar", "is_outlier"]
    pd.testing.assert_frame_equal(saidas[0]["predicoes"][colunas], saidas[1]["predicoes"][colunas])
    assert saidas[0]["run_id"] != saidas[1]["run_id"]


def test_sem_outliers_gera_arquivo_com_cabecalho(tmp_path):
    rows = make_rows()
    for row in rows:                       # teste no centro da distribuição de cada nó: nada isolado
        if row["window_end_utc"] > "2026-01-01T04:15:00Z":
            row.update(center(int(row["node_id"])))
    _, s = run(tmp_path, rows, "--corte-treino", CORTE)
    assert len(s["outliers"]) == 0 and "anomaly_score" in s["outliers"].columns
    assert (s["resumo"]["outliers_teste"] == "0").all()
    assert "Nenhuma janela do teste ficou acima do limiar" in s["relatorio_md"]


def test_pasta_da_execucao_nunca_e_reutilizada(tmp_path):
    agora = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    sufixos = iter(["aaaa0000", "aaaa0000", "bbbb1111"])
    primeira = det.create_run_dir(tmp_path, "dataset_x", agora, lambda: next(sufixos))
    (primeira / "marca.txt").write_text("original", encoding="utf-8")
    segunda = det.create_run_dir(tmp_path, "dataset_x", agora, lambda: next(sufixos))
    assert primeira.name == "detector_001_isolation_forest_sklearn__dataset_x__20260102T030405Z__aaaa0000"
    assert segunda.name.endswith("__bbbb1111")
    assert (primeira / "marca.txt").read_text(encoding="utf-8") == "original"


def test_dataset_id_sanitizado_no_caminho_e_preservado_nos_metadados(tmp_path):
    folder, s = run(tmp_path, make_rows(), "--corte-treino", CORTE, dataset_id="dataset ../x:y")
    assert folder.parent == tmp_path / "res"
    assert "__dataset_x_y__" in folder.name
    assert s["execucao"]["entrada"]["dataset_id"] == "dataset ../x:y"


def test_resumo_e_relatorio_coerentes_com_no_sem_teste_analisado(tmp_path):
    rows = make_rows()
    for k in range(34, 60):                # o nó 3 para de medir depois do treino
        invalidate(rows[position(2, k)])
    _, s = run(tmp_path, rows, "--corte-treino", CORTE)
    r = s["resumo"].set_index("node_id")
    assert r.loc["3", "teste_analisado"] == "0" and r.loc["3", "taxa_outliers_teste"] == "não aplicável"
    n = r[["janelas_entrada", "treino_candidato", "treino_efetivo", "excluido_treino", "folga", "teste",
           "teste_analisado", "excluido_teste"]].astype(int)
    assert (n.treino_candidato + n.folga + n.teste == n.janelas_entrada).all()
    assert (n.treino_efetivo + n.excluido_treino == n.treino_candidato).all()
    assert (n.teste_analisado + n.excluido_teste == n.teste).all()
    ex = s["execucao"]
    c = ex["contagens"]
    assert c["entrada"] == len(rows) == n.janelas_entrada.sum()
    assert c["treino_efetivo"] + c["folga"] + c["teste_analisado"] + c["excluidas_distintas"] == c["entrada"]
    assert sum(ex["motivos_exclusao"].values()) >= c["excluidas_distintas"]
    assert c["dispositivos_sem_teste_analisado"] == [3]
    html = s["relatorio_html"]
    assert "não aplicável" in html and "data:image/png;base64," in html
    assert not re.search(r"""(src|href)=["']https?:""", html)
    assert "graficos/serie_scores__" in s["relatorio_md"]


def test_no_sem_referencia_propria_e_declarado(tmp_path):
    rows = make_rows()
    for k in range(34):
        rows[position(2, k)]["reference_eligible_by_policy"] = "0"
    _, s = run(tmp_path, rows, "--corte-treino", CORTE)
    r = s["resumo"].set_index("node_id")
    assert r.loc["3", "sem_referencia_propria"] == "1" and r.loc["1", "sem_referencia_propria"] == "0"
    assert int(r.loc["3", "teste_analisado"]) > 0          # o modelo global ainda pontua o nó
    assert "sem referência própria" in s["relatorio_md"]


def test_html_escapa_texto_vindo_do_csv(tmp_path):
    rows = make_rows()
    for row in rows:
        if row["node_id"] == "1":
            row["node_name"] = "<script>alert(1)</script>"
    _, s = run(tmp_path, rows, "--corte-treino", CORTE)
    assert "<script>alert(1)</script>" not in s["relatorio_html"]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in s["relatorio_html"]
    assert "<script>" not in s["relatorio_md"]          # tag do CSV escapada também no Markdown
    assert "corte < fim" in s["relatorio_md"]           # o texto do próprio relatório não é escapado


def test_treino_vazio_gera_diagnostico_sem_deteccao(tmp_path):
    folder, s = run(tmp_path, make_rows(), "--corte-treino", "2025-12-31T00:00:00Z", expected=3)
    assert s["execucao"]["estado"] == "dados_insuficientes"
    assert "treino efetivo" in s["execucao"]["motivo_estado"]
    assert not list(folder.glob("outliers__*"))
    assert "sem detecção" in s["relatorio_md"] and "foram analisadas no teste" not in s["relatorio_md"]
    assert "recebe score e decisão" not in s["relatorio_md"]
    p = s["predicoes"]
    assert p["anomaly_score"].isna().all() and p["is_outlier"].isna().all()
    assert "teste_analisado" not in set(p["estado_analise"])          # nada foi analisado sem modelo
    sem_modelo = p["estado_analise"] == "teste_sem_modelo"
    assert sem_modelo.any()
    assert (p.loc[sem_modelo, "motivos_exclusao"] == "execucao:dados_insuficientes").all()


def test_features_constantes_no_treino_geram_diagnostico(tmp_path):
    rows = make_rows()
    for row in rows:
        row.update(center(1))
    _, s = run(tmp_path, rows, "--corte-treino", CORTE, expected=3)
    assert s["execucao"]["estado"] == "dados_insuficientes"
    assert "constantes" in s["execucao"]["motivo_estado"]


def test_falha_depois_de_criar_a_pasta_fica_registrada(tmp_path, monkeypatch):
    def falha(*args, **kwargs):
        raise RuntimeError("falha simulada nos gráficos")
    monkeypatch.setattr(det, "plot_charts", falha)
    folder, s = run(tmp_path, make_rows(), "--corte-treino", CORTE, expected=1)
    assert s["execucao"]["estado"] == "falhou"
    assert "falha simulada" in s["execucao"]["motivo_estado"]
    assert not list(folder.glob("relatorio__*"))


@pytest.mark.parametrize("args", [
    ["--percentil-limiar", "100"], ["--percentil-limiar", "0"], ["--fracao-treino", "1"],
    ["--corte-treino", "2026-01-01T04:00:00"],
    ["--corte-treino", CORTE, "--fracao-treino", "0.5"],
    ["--corte-treino", CORTE, "--inicio-treino", "2026-01-01T05:00:00Z"],
    ["--n-estimators", "0"], ["--seed", "-1"],
])
def test_argumentos_invalidos_encerram_sem_criar_pasta(tmp_path, capsys, args):
    code, folders = run_detector(tmp_path, write_package(tmp_path / "p", make_rows(nodes=(1,), n=5)), *args)
    assert code == 2 and folders == []
    assert "erro" in capsys.readouterr().err


def test_entrada_com_hash_divergente_encerra_sem_criar_pasta(tmp_path):
    pacote = write_package(tmp_path / "p", make_rows())
    with open(pacote / "features.csv", "a", encoding="utf-8", newline="") as fh:
        fh.write("\r\n")
    code, folders = run_detector(tmp_path, pacote, "--corte-treino", CORTE)
    assert code == 2 and folders == []
