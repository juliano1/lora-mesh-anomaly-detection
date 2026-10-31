#!/usr/bin/env python
"""
Detector 001 — Isolation Forest do scikit-learn sobre o `features.csv` exportado.

Lê a pasta extraída de um pacote **Features por janela** ou **Completo** (ou um
`features.csv` isolado), confere os SHA-256 do manifesto e o contrato das 11
features, separa treino e teste no tempo com um corte único e folga de uma janela,
ajusta RobustScaler + IsolationForest só no treino efetivo e pontua o teste.
anomaly_score = -score_samples (maior = mais anômalo); uma janela de teste é
sinalizada quando anomaly_score > limiar, com o limiar no percentil configurado dos
scores do treino. Cada execução grava uma pasta nova em resultados/, nomeada pelo
run_id, com relatório HTML e Markdown, CSVs, gráficos e metadados de reprodução.

Uso (a partir da raiz do repositório):
  python importacao_arquivos_csv/codigos/detector_001_isolation_forest_sklearn.py \\
      --entrada importacao_arquivos_csv/entrada_csv/<pasta_do_dataset> \\
      [--corte-treino T | --fracao-treino 0.70] [--inicio-treino T]
      [--percentil-limiar 99] [--seed 42] [--n-estimators 300] [--saida DIR]
Instantes (T) em ISO 8601 com fuso explícito (Z ou offset), convertidos para UTC.
Saída: 0 concluída; 1 falha inesperada (pasta com estado "falhou"); 2 erro de uso
ou de entrada (nada é gravado); 3 dados insuficientes (diagnóstico sem detecção;
sem pasta só quando não há janela valid para calcular o corte automático).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import re
import secrets
import shlex
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem interface gráfica

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

DETECTOR_ID = "detector_001_isolation_forest_sklearn"
DETECTOR_VERSION = "1.0.0"
SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
DEFAULT_OUTPUT = SCRIPT.parents[1] / "resultados"

FEATURES = ("rssi_mean", "rssi_std", "snr_mean", "snr_std", "pdr_mean", "battery_delta",
            "packet_loss_rate", "rssi_snr_divergence", "beacon_interval_std",
            "route_change_rate", "link_asymmetry")
REQUIRED = ("source_id", "node_id", "window_start_utc", "window_end_utc", "window_seconds",
            "step_seconds", "feature_status", "defaulted_features", "unavailable_features",
            "origin_status", "simulation_overlap", "reference_eligible_by_policy")
FEATURE_STATUS = ("valid", "insufficient_data", "invalid_numeric", "error")
ORIGIN_STATUS = ("real", "unknown", "mixed", "synthetic", "no_data")
REFERENCE_ORIGINS = ("real", "unknown")
VERSION_FIELDS = ("feature_schema_version", "extractor_revision", "extraction_policy")
ADDED_COLUMNS = ("numero_linha_original", "run_id", "detector", "detector_versao",
                 "particao_temporal", "estado_analise", "motivos_exclusao", "anomaly_score",
                 "limiar", "is_outlier")
SUMMARY_COLUMNS = ("node_id", "node_name", "janelas_entrada", "janelas_valid", "janelas_analisaveis",
                   "treino_candidato", "treino_efetivo", "excluido_treino", "folga", "teste",
                   "teste_analisado", "excluido_teste", "outliers_teste", "taxa_outliers_teste",
                   "teste_analisado_com_simulacao", "outliers_com_simulacao", "score_max_teste",
                   "primeira_janela_fim_utc", "ultima_janela_fim_utc", "sem_referencia_propria")
DEFAULT_FRACTION = 0.70
MIN_TRAIN_ROWS = 2
SMALL_TRAIN_ROWS = 256
TOP_N = 20
NOT_APPLICABLE = "não aplicável"

NEEDS_FEATURES = ("este detector lê o features.csv das exportações 'Features por janela' ou "
                  "'Completo' (scripts/export_dataset.py --kind features ou full); extraia o ZIP e aponte para a pasta que "
                  "contém features.csv, manifest.json e schema.json, ou para o próprio features.csv")
OTHER_PACKAGE_FILES = {
    "telemetry.csv": "medições brutas, uma recepção de relatório por linha",
    "neighbors.csv": "medições brutas, um vizinho relatado por linha",
    "nodes.csv": "cadastro dos nós",
    "networks.csv": "cadastro das redes",
    "simulation_runs.csv": "execuções do simulador",
    "coverage.csv": "resumo das janelas de cada nó",
    "events_template.csv": "formulário de eventos de campo (vem vazio)",
}
REASON_TEXT = {
    "feature_ausente": "ao menos uma das 11 features vazia (nula)",
    "feature_nao_finita": "ao menos uma feature NaN ou infinita",
    "default_declarado": "`defaulted_features` não vazio: valor fixo no lugar de medição",
    "feature_indisponivel": "`unavailable_features` não vazio",
    "default_nao_informado": "`defaulted_features` nulo: default não verificável",
    "indisponibilidade_nao_informada": "`unavailable_features` nulo: default não verificável",
    "antes_inicio_referencia": "janela começa antes de `--inicio-treino`",
    "nao_elegivel_politica": "`reference_eligible_by_policy` = 0",
    "elegibilidade_nao_informada": "`reference_eligible_by_policy` nulo",
    "simulacao_sobreposta": "`simulation_overlap` = 1",
    "simulacao_nao_informada": "`simulation_overlap` nulo",
    "separacao_treino_teste": "janela na folga entre treino e teste (sem score por construção)",
}
GENERAL_LIMITATIONS = (
    "**Defaults.** Janelas com feature por default (`defaulted_features` ou `unavailable_features` "
    "não vazios) ficam fora do treino e do teste: a análise cobre só janelas medidas, e um nó com "
    "muitos defaults (por exemplo, sem enlace reverso) fica sub-representado.",
    "**Proveniência.** `origin_status = unknown` é proveniência não comprovada: trata-se como não "
    "sintético, sem confirmar que os registros são reais.",
    "**Referência não é saúde.** A elegibilidade da política (`reference_eligible_by_policy`) só "
    "exclui janelas não válidas, com registro sintético ou sobrepostas a simulações. Anomalias "
    "naturais que existam no treino passam a contar como comportamento de referência.",
    "**Recepções duplicadas.** Cada relatório dos nós sensores costuma chegar duas vezes; isso "
    "infla `beacon_interval_std` e deixa `route_change_rate` perto da metade da taxa por relatório. "
    "As features foram usadas como exportadas.",
    "**`pdr_mean`** é um índice heurístico derivado de RSSI e SNR, não a taxa de entrega medida.",
    "**`route_change_rate`** acompanha o \"parent\", que é o primeiro vizinho direto da tabela de "
    "rotas, não necessariamente o próximo salto até o gateway.",
    "**Dependência entre janelas.** Janelas vizinhas de um nó se sobrepõem (W − S segundos) e um "
    "registro pode entrar em janelas de nós vizinhos; N janelas sinalizadas não são N incidentes "
    "independentes. Features e scores também dependem da janela e do passo escolhidos na exportação.",
    "**Diferenças entre dispositivos.** O modelo e a escala são globais (os mesmos para todos os "
    "nós), não por nó: um nó com nível típico diferente (distância, posição) pode ser sinalizado "
    "pelo nível, e não por mudança no próprio comportamento.",
    "**Sem rótulos verificados.** Não há acurácia, precisão, recall, F1 nem falsos positivos. "
    "`simulation_overlap` indica uma intervenção conhecida, não um rótulo de falha, e não torna "
    "normais as demais janelas. Ausência de detecção não comprova ausência de falhas.",
    "**Não é o detector online do sistema.** Modelo, período de treino, parâmetros e políticas são "
    "os desta execução; o resultado não reproduz os alertas do sistema.",
    "**Ordem e semente.** Com semente fixa, o Isolation Forest ainda depende da ordem das linhas; "
    "a ordem usada é (`window_end_utc`, `node_id`), uma escolha declarada.",
)
UTC_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|\+00:00)"
INT_PATTERN = r"-?\d{1,18}"   # cabe em int64
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf|infinity)"


class InputError(Exception):
    """Entrada ou argumento fora do contrato; nada é gravado (saída 2)."""


class InsufficientDataError(Exception):
    """Não há dados para ajustar e analisar; diagnóstico sem detecção (saída 3)."""


@dataclass
class Settings:
    entrada: Path
    saida: Path
    cut: pd.Timestamp | None
    cut_text: str | None
    fraction: float | None
    training_start: pd.Timestamp | None
    percentile: float
    seed: int
    n_estimators: int
    argv: list[str]


@dataclass
class Dataset:
    csv_path: Path
    sha256: str
    size: int
    raw: pd.DataFrame          # células como lidas (texto; NaN = célula vazia)
    data: pd.DataFrame         # colunas tipadas usadas pelas regras, na ordem do arquivo
    provenance: dict
    dataset_id: str
    source_id: str
    window_seconds: int
    step_seconds: int


@dataclass
class Plan:
    cut: pd.Timestamp
    cut_mode: str
    fraction: float | None
    training_start: pd.Timestamp | None
    gap_end: pd.Timestamp
    classes: pd.DataFrame      # particao_temporal, estado_analise, motivos, qualidade_ok
    separation: dict
    train_index: pd.Index      # treino efetivo, em ordem (window_end_utc, node_id)
    test_index: pd.Index       # teste analisado, na mesma ordem
    constant_features: list[str]


@dataclass
class Result:
    pipeline: object
    threshold: float
    scores: pd.Series          # score do treino efetivo e do teste analisado; NaN no resto
    flagged: pd.Series         # True só no teste analisado acima do limiar
    ties: int
    train_above: float


# ---------------------------------------------------------------- utilidades


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    # Fora do repositório, o diretório pessoal vira "~" para não gravar o nome do usuário.
    try:
        return "~/" + resolved.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(resolved)


def fmt_utc(ts) -> str | None:
    """Instante em ISO 8601 UTC com `Z`, com fração de segundo só quando houver."""
    if ts is None or pd.isna(ts):
        return None
    ts = pd.Timestamp(ts).tz_convert("UTC")
    text = ts.strftime("%Y-%m-%dT%H:%M:%S")
    if ts.microsecond:
        text += f".{ts.microsecond:06d}".rstrip("0")
    return text + "Z"


def fmt_when(ts) -> str:
    ts = pd.Timestamp(ts).tz_convert("UTC")
    exact_minute = ts.second == 0 and ts.microsecond == 0
    return ts.strftime("%Y-%m-%d %H:%M" if exact_minute else "%Y-%m-%d %H:%M:%S") + " UTC"


def fmt_num(value, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def fmt_rate(count: int, total: int) -> str:
    return NOT_APPLICABLE if total == 0 else fmt_num(100 * count / total, 2) + " %"


def cell_text(value) -> str | None:
    """Valor de célula como o exportador grava: nulo = None; float com precisão total."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value)) if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return fmt_utc(value)
    return str(value)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """CSV UTF-8 com vírgula e CRLF; valor não nulo entre aspas, nulo = célula vazia sem aspas."""
    def quote(text: str) -> str:
        return '"' + text.replace('"', '""') + '"'

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(quote(str(c)) for c in frame.columns) + "\r\n")
        for row in frame.itertuples(index=False, name=None):
            cells = (cell_text(v) for v in row)
            fh.write(",".join("" if c is None else quote(c) for c in cells) + "\r\n")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return fmt_utc(obj)
    if isinstance(obj, (str, Path)):
        return str(obj)
    return str(obj)


def write_json(path: Path, obj) -> None:
    text = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _reject_constant(name):
    raise ValueError(f"constante {name} não é JSON estrito")


def _rows_hint(mask: pd.Series, values: pd.Series | None = None, limit: int = 5) -> str:
    """Primeiras linhas de dados (1 = primeira linha depois do cabeçalho) marcadas em `mask`."""
    positions = list(mask[mask].index[:limit])
    parts = [f"linha {p + 1}" + (f" ({values[p]!r})" if values is not None else "") for p in positions]
    extra = int(mask.sum()) - len(positions)
    return ", ".join(parts) + (f" e mais {extra}" if extra > 0 else "")


# ---------------------------------------------------------------- entrada


def resolve_input(path: Path) -> tuple[Path, Path | None, Path | None]:
    """Devolve (features.csv, manifest.json ou None, schema.json ou None)."""
    path = Path(path)
    if not path.exists():
        raise InputError(f"entrada não encontrada: {path}")
    if path.is_dir():
        csv_path = path / "features.csv"
        if not csv_path.is_file():
            inner = sorted(p.parent for p in path.glob("*/features.csv"))
            if inner:
                raise InputError("a pasta indicada não contém features.csv; aponte para a pasta "
                                 "interna do pacote extraído: " + "; ".join(str(p) for p in inner))
            if any((path / name).exists() for name in ("telemetry.csv", "neighbors.csv", "manifest.json")):
                raise InputError(f"{path} parece um pacote sem features.csv (opção Bruto): {NEEDS_FEATURES}")
            if any(path.glob("*.zip")):
                raise InputError(f"{path} contém ZIP; extraia o pacote mantendo a pasta interna e "
                                 "aponte para ela")
            raise InputError(f"features.csv não encontrado em {path}: {NEEDS_FEATURES}")
    else:
        csv_path = path
        if path.suffix.lower() == ".zip":
            raise InputError(f"{path} é um ZIP: extraia o pacote mantendo a pasta interna e aponte para ela")
        if path.name in OTHER_PACKAGE_FILES:
            raise InputError(f"{path.name} não é a entrada deste detector "
                             f"({OTHER_PACKAGE_FILES[path.name]}): {NEEDS_FEATURES}")
    manifest = csv_path.parent / "manifest.json"
    schema = csv_path.parent / "schema.json"
    return csv_path, (manifest if manifest.is_file() else None), (schema if schema.is_file() else None)


def _load_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_bytes().decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise InputError(f"{path.name} inválido ({path}): {exc}") from exc
    if not isinstance(obj, dict):
        raise InputError(f"{path.name} inválido ({path}): o conteúdo não é um objeto JSON")
    return obj


def _check_metadata(csv_path: Path, csv_sha: str, manifest_path: Path | None,
                    schema_path: Path | None) -> tuple[dict, dict | None]:
    """Confere hashes e contrato nos metadados presentes; nenhum metadado presente é ignorado."""
    prov = {"manifest": None, "schema": None, "verificacoes": [], "versoes": {},
            "versoes_fonte": {}, "dataset_id": None}
    manifest = None
    if manifest_path is not None:
        manifest = _load_json(manifest_path)
        files = manifest.get("files")
        entry = files.get("features.csv") if isinstance(files, dict) else None
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise InputError("manifest.json inválido: falta files['features.csv'].sha256")
        if entry["sha256"] != csv_sha:
            raise InputError(f"SHA-256 de {csv_path.name} ({csv_sha}) diverge do manifest.json "
                             f"({entry['sha256']}): o arquivo foi alterado ou não pertence a este pacote")
        prov["verificacoes"].append("SHA-256 de features.csv confere com manifest.json")
        dataset_id = manifest.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise InputError("manifest.json inválido: dataset_id ausente ou vazio")
        prov["dataset_id"] = dataset_id
        features = manifest.get("features")
        if features is not None and not isinstance(features, dict):
            raise InputError("manifest.json inválido: 'features' não é um objeto")
        if features:
            names = features.get("feature_names")
            if names is not None and list(names) != list(FEATURES):
                raise InputError(f"contrato incompatível: manifest.json features.feature_names = {names}; "
                                 f"o detector 001 exige, nesta ordem: {list(FEATURES)}")
            prov["versoes"] = {k: features[k] for k in VERSION_FIELDS if features.get(k) is not None}
            prov["versoes_fonte"] = {k: "manifest.json" for k in prov["versoes"]}
        prov["manifest"] = {
            "caminho": display_path(manifest_path), "sha256": sha256_file(manifest_path),
            **{k: manifest.get(k) for k in ("format", "export_kind", "created_at", "campaign_id",
                                            "source", "request", "interpretation")},
        }
    if schema_path is not None:
        schema = _load_json(schema_path)
        schema_sha = sha256_file(schema_path)
        if manifest is not None:
            entry = manifest["files"].get("schema.json")
            if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
                raise InputError("manifest.json inválido: não registra o SHA-256 de schema.json")
            if entry["sha256"] != schema_sha:
                raise InputError(f"SHA-256 de schema.json ({schema_sha}) diverge do manifest.json "
                                 f"({entry['sha256']}): o arquivo foi alterado ou não pertence a este pacote")
            prov["verificacoes"].append("SHA-256 de schema.json confere com manifest.json")
        try:
            columns = schema["files"]["features.csv"]["model_input_columns"]
        except (KeyError, TypeError):
            raise InputError("schema.json inválido: falta files['features.csv'].model_input_columns") from None
        if not isinstance(columns, list) or columns != list(FEATURES):
            raise InputError(f"contrato incompatível: schema.json model_input_columns = {columns}; o "
                             f"detector 001 exige, nesta ordem: {list(FEATURES)}; a seleção não é adaptada")
        prov["verificacoes"].append("model_input_columns do schema.json = as 11 features, na ordem do detector")
        prov["schema"] = {"caminho": display_path(schema_path), "sha256": schema_sha,
                          "format": schema.get("format")}
    if manifest_path and schema_path:
        prov["situacao"] = "completa: manifest.json e schema.json presentes e conferidos"
    elif manifest_path:
        prov["situacao"] = "proveniência limitada: schema ausente"
    elif schema_path:
        prov["situacao"] = "proveniência limitada: manifesto ausente (hashes não conferidos)"
    else:
        prov["situacao"] = "proveniência limitada: manifesto/schema ausentes"
    prov["completa"] = bool(manifest_path and schema_path)
    return prov, manifest


def _read_raw(csv_path: Path) -> pd.DataFrame:
    with open(csv_path, "rb") as fh:
        head = fh.read(65536)
    if head.startswith(b"\xef\xbb\xbf"):
        raise InputError(f"{csv_path.name} começa com BOM: o arquivo foi regravado por outro programa; "
                         "use o arquivo original do pacote")
    try:
        first_line = head.split(b"\n", 1)[0].rstrip(b"\r").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{csv_path.name} não está em UTF-8: {exc}") from exc
    header = next(csv.reader([first_line]), [])
    if not header:
        raise InputError(f"{csv_path.name} está vazio")
    if len(header) == 1 and ";" in first_line:
        raise InputError(f"{csv_path.name} usa ';' como separador; o exportador usa vírgula: use o arquivo original")
    repeated = sorted({n for n in header if header.count(n) > 1})
    if repeated:
        raise InputError(f"colunas repetidas no cabeçalho de {csv_path.name}: {', '.join(repeated)}")
    missing = [c for c in (*REQUIRED, *FEATURES) if c not in header]
    if missing:
        raise InputError(f"faltam colunas obrigatórias em {csv_path.name}: {', '.join(missing)}. "
                         f"{NEEDS_FEATURES[0].upper()}{NEEDS_FEATURES[1:]}; não há substitutos para "
                         "metadados ausentes: faça uma nova exportação")
    clash = [c for c in ADDED_COLUMNS if c in header]
    if clash:
        raise InputError(f"{csv_path.name} já tem colunas que o detector acrescenta: {', '.join(clash)}")
    # o pandas completa com nulos uma linha curta; aqui ela é erro estrutural, não célula vazia
    try:
        with open(csv_path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            n_rows, wrong = 0, []
            for row in reader:
                n_rows += 1
                if len(row) != len(header):
                    wrong.append(reader.line_num)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise InputError(f"{csv_path.name} ilegível como CSV UTF-8: {exc}") from exc
    if wrong:
        extra = f" e mais {len(wrong) - 5}" if len(wrong) > 5 else ""
        raise InputError(f"{csv_path.name}: linhas com número de campos diferente do cabeçalho ({len(header)} "
                         f"campos), linhas do arquivo {', '.join(map(str, wrong[:5]))}{extra}")
    try:
        raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise InputError(f"{csv_path.name} ilegível como CSV UTF-8: {exc}") from exc
    if raw.empty:
        raise InputError(f"{csv_path.name} não tem linhas de dados")
    if len(raw) != n_rows:
        raise InputError(f"{csv_path.name}: {n_rows} linhas de dados no arquivo, {len(raw)} lidas pelo pandas")
    return raw.reset_index(drop=True)


def _text(raw: pd.DataFrame, col: str) -> pd.Series:
    missing = raw[col].isna()
    if missing.any():
        raise InputError(f"{col}: valor ausente em {_rows_hint(missing)}")
    return raw[col].astype(str)


def _integer(raw: pd.DataFrame, col: str) -> pd.Series:
    s = raw[col]
    bad = s.isna() | ~s.fillna("").str.fullmatch(INT_PATTERN)
    if bad.any():
        raise InputError(f"{col}: inteiro ausente, inválido ou fora do intervalo em {_rows_hint(bad, s)}")
    return s.astype("int64")


def _choice(raw: pd.DataFrame, col: str, allowed: tuple[str, ...]) -> pd.Series:
    s = raw[col]
    bad = s.isna() | ~s.isin(allowed)
    if bad.any():
        raise InputError(f"{col}: valor fora de {list(allowed)} em {_rows_hint(bad, s)}")
    return s.astype(str)


def _utc(raw: pd.DataFrame, col: str) -> pd.Series:
    s = raw[col]
    bad = s.isna() | ~s.fillna("").str.fullmatch(UTC_PATTERN)
    if bad.any():
        raise InputError(f"{col}: data ausente ou fora de ISO 8601 em UTC (sufixo Z ou +00:00) em "
                         f"{_rows_hint(bad, s)}")
    parsed = pd.to_datetime(s, format="ISO8601", utc=True, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        raise InputError(f"{col}: data inexistente no calendário em {_rows_hint(bad, s)}")
    return parsed


def _boolean(raw: pd.DataFrame, col: str) -> pd.Series:
    """Booleano do exportador: "0", "1" ou nulo (Int64 com <NA>); nada de astype(bool)."""
    s = raw[col]
    bad = s.notna() & ~s.isin(["0", "1"])
    if bad.any():
        raise InputError(f"{col}: valor fora do contrato (0, 1 ou vazio) em {_rows_hint(bad, s)}")
    return s.map({"0": 0, "1": 1}).astype("Int64")


def _json(raw: pd.DataFrame, col: str, kind: type) -> pd.Series:
    expected = "lista" if kind is list else "objeto"

    def parse(position: int, text: str):
        try:
            value = json.loads(text, parse_constant=_reject_constant)
        except ValueError as exc:
            raise InputError(f"{col}: JSON malformado na linha {position + 1} ({text!r}): {exc}") from exc
        if not isinstance(value, kind):
            raise InputError(f"{col}: esperado {expected} JSON na linha {position + 1} ({text!r})")
        return value

    values = [None if pd.isna(t) else parse(p, t) for p, t in raw[col].items()]
    return pd.Series(values, index=raw.index, dtype=object)


def _float(raw: pd.DataFrame, col: str) -> pd.Series:
    s = raw[col]
    bad = s.notna() & ~s.fillna("").str.fullmatch(NUMBER_PATTERN, case=False)
    if bad.any():
        raise InputError(f"{col}: valor não numérico (ponto decimal) em {_rows_hint(bad, s)}")
    return s.map(lambda v: float(v) if isinstance(v, str) else np.nan).astype(float)


def _parse(raw: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=raw.index)
    d["numero_linha_original"] = raw.index + 1
    d["source_id"] = _text(raw, "source_id")
    d["node_id"] = _integer(raw, "node_id")
    d["window_start_utc"] = _utc(raw, "window_start_utc")
    d["window_end_utc"] = _utc(raw, "window_end_utc")
    d["window_seconds"] = _integer(raw, "window_seconds")
    d["step_seconds"] = _integer(raw, "step_seconds")
    d["feature_status"] = _choice(raw, "feature_status", FEATURE_STATUS)
    d["origin_status"] = _choice(raw, "origin_status", ORIGIN_STATUS)
    d["defaulted_features"] = _json(raw, "defaulted_features", list)
    d["unavailable_features"] = _json(raw, "unavailable_features", dict)
    d["simulation_overlap"] = _boolean(raw, "simulation_overlap")
    d["reference_eligible_by_policy"] = _boolean(raw, "reference_eligible_by_policy")
    for feature in FEATURES:
        d[feature] = _float(raw, feature)
    values = d[list(FEATURES)].to_numpy(dtype=float)
    present = raw[list(FEATURES)].notna().to_numpy()
    d["feature_nula"] = ~present.all(axis=1)
    d["feature_nao_finita"] = (present & ~np.isfinite(values)).any(axis=1)
    if "simulation_run_ids" in raw:
        d["simulation_run_ids"] = _json(raw, "simulation_run_ids", list)
    if "node_name" in raw:
        d["node_name"] = raw["node_name"]
    return d


def _validate_structure(d: pd.DataFrame) -> None:
    bad = d["window_start_utc"] >= d["window_end_utc"]
    if bad.any():
        raise InputError(f"window_start_utc não é anterior a window_end_utc em {_rows_hint(bad)}")
    bad = (d["window_seconds"] <= 0) | (d["step_seconds"] <= 0)
    if bad.any():
        raise InputError(f"window_seconds e step_seconds devem ser positivos; falham em {_rows_hint(bad)}")
    duration = (d["window_end_utc"] - d["window_start_utc"]).dt.total_seconds()
    bad = duration != d["window_seconds"]
    if bad.any():
        raise InputError(f"duração da janela (fim − início) diferente de window_seconds em {_rows_hint(bad)}")
    sources = sorted(d["source_id"].unique())
    if len(sources) > 1:
        raise InputError(f"mais de uma fonte (source_id) no arquivo: {sources}; analise uma exportação "
                         "por execução, sem concatenar")
    configs = sorted(set(zip(d["window_seconds"], d["step_seconds"])))
    if len(configs) > 1:
        raise InputError(f"mais de uma configuração (window_seconds, step_seconds): {configs}; analise "
                         "uma exportação por execução")
    repeated = d.duplicated(["source_id", "node_id", "window_end_utc"], keep=False)
    if repeated.any():
        raise InputError(f"chaves repetidas (source_id, node_id, window_end_utc) em {_rows_hint(repeated)}; "
                         "o detector não elimina duplicatas: confira a exportação")


def _check_consistency(raw: pd.DataFrame, d: pd.DataFrame, prov: dict, manifest: dict | None) -> None:
    """Versões, fonte e janela do CSV contra o manifesto, quando houver."""
    for field in VERSION_FIELDS:
        if field not in raw:
            continue
        values = sorted(raw[field].dropna().unique())
        if len(values) > 1:
            raise InputError(f"{field}: mais de um valor no arquivo ({values}); analise uma exportação por execução")
        if not values:
            continue
        declared = prov["versoes"].get(field)
        if declared is not None:
            if declared != values[0]:
                raise InputError(f"{field} diverge entre features.csv ({values[0]}) e manifest.json ({declared})")
        else:
            prov["versoes"][field] = values[0]
            prov["versoes_fonte"][field] = "features.csv"
    if manifest is None:
        return
    source = d["source_id"].iloc[0]
    for block in ("source", "interpretation"):
        declared = (manifest.get(block) or {}).get("source_id")
        if declared is not None and declared != source:
            raise InputError(f"source_id diverge entre features.csv ({source}) e manifest.json {block} ({declared})")
    interpretation = manifest.get("interpretation") or {}
    for field in ("window_seconds", "step_seconds"):
        declared = interpretation.get(field)
        actual = int(d[field].iloc[0])
        if declared is not None and declared != actual:
            raise InputError(f"{field} diverge entre features.csv ({actual}) e manifest.json ({declared})")
    rows = manifest["files"]["features.csv"].get("rows")
    if rows is not None:
        if rows != len(d):
            raise InputError(f"features.csv tem {len(d)} linhas; manifest.json declara {rows}")
        prov["verificacoes"].append("número de linhas de features.csv confere com manifest.json")
    prov["verificacoes"].append("source_id, window_seconds e step_seconds conferem com manifest.json")


def load_dataset(path) -> Dataset:
    """Lê e valida a entrada; erros de contrato levantam InputError antes de qualquer gravação."""
    csv_path, manifest_path, schema_path = resolve_input(Path(path))
    csv_sha = sha256_file(csv_path)
    prov, manifest = _check_metadata(csv_path, csv_sha, manifest_path, schema_path)
    raw = _read_raw(csv_path)
    data = _parse(raw)
    _validate_structure(data)
    _check_consistency(raw, data, prov, manifest)
    prov["verificacoes"].append("datas ISO 8601 em UTC, JSON e booleanos 0/1 conforme o contrato")
    prov["verificacoes"].append("uma fonte, uma configuração de janela/passo e chaves "
                                "(source_id, node_id, window_end_utc) únicas")
    dataset_id = prov["dataset_id"] or f"csv_{csv_sha[:12]}"
    prov["dataset_id_origem"] = "manifest.json" if prov["dataset_id"] else "derivado do SHA-256 do CSV"
    prov["dataset_id"] = dataset_id
    return Dataset(csv_path=csv_path, sha256=csv_sha, size=csv_path.stat().st_size, raw=raw, data=data,
                   provenance=prov, dataset_id=dataset_id, source_id=str(data["source_id"].iloc[0]),
                   window_seconds=int(data["window_seconds"].iloc[0]),
                   step_seconds=int(data["step_seconds"].iloc[0]))


# ---------------------------------------------------------------- partições e filtros


def parse_utc_instant(text: str, option: str) -> pd.Timestamp:
    value = text.strip()
    if value[-1:] in ("Z", "z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise InputError(f"{option}: instante inválido {text!r}; use ISO 8601, ex.: 2026-10-16T00:00:00Z") from None
    if parsed.tzinfo is None:
        raise InputError(f"{option}: {text!r} não tem fuso explícito; use Z ou um offset, "
                         "ex.: 2026-10-16T00:00:00Z ou 2026-10-15T21:00:00-03:00")
    return pd.Timestamp(parsed).tz_convert("UTC")


def automatic_cut(data: pd.DataFrame, fraction: float) -> pd.Timestamp:
    """Quantil `fraction` dos fins das janelas valid, arredondado para baixo ao minuto."""
    ends = data.loc[data["feature_status"] == "valid", "window_end_utc"]
    if ends.empty:
        raise InsufficientDataError("nenhuma janela com feature_status = valid: não há corte automático "
                                    "nem dados para ajustar o modelo")
    return ends.quantile(fraction).floor("min")


def assign_partitions(window_end: pd.Series, cut: pd.Timestamp, window_seconds: int) -> pd.Series:
    """treino_candidato: fim ≤ corte; folga: corte < fim ≤ corte + W; teste: fim > corte + W."""
    gap_end = cut + pd.Timedelta(seconds=window_seconds)
    partition = pd.Series("teste", index=window_end.index, dtype=object)
    partition[window_end <= gap_end] = "folga"
    partition[window_end <= cut] = "treino_candidato"
    return partition


def _append(reasons: list[list[str]], mask, code: str) -> None:
    for position in np.flatnonzero(np.asarray(mask, dtype=bool)):
        reasons[position].append(code)


def quality_reasons(d: pd.DataFrame) -> list[list[str]]:
    """Motivos que impedem a análise de uma linha em qualquer partição."""
    reasons: list[list[str]] = [[] for _ in range(len(d))]
    for status in FEATURE_STATUS[1:]:
        _append(reasons, d["feature_status"] == status, f"estado_{status}")
    _append(reasons, d["feature_nula"], "feature_ausente")
    _append(reasons, d["feature_nao_finita"], "feature_nao_finita")
    defaulted, unavailable = d["defaulted_features"], d["unavailable_features"]
    _append(reasons, defaulted.isna(), "default_nao_informado")
    _append(reasons, defaulted.map(lambda v: isinstance(v, list) and len(v) > 0), "default_declarado")
    _append(reasons, unavailable.isna(), "indisponibilidade_nao_informada")
    _append(reasons, unavailable.map(lambda v: isinstance(v, dict) and len(v) > 0), "feature_indisponivel")
    return reasons


def reference_reasons(d: pd.DataFrame, training_start: pd.Timestamp | None) -> list[list[str]]:
    """Motivos que tiram uma linha do treino efetivo (não se aplicam ao teste)."""
    reasons: list[list[str]] = [[] for _ in range(len(d))]
    if training_start is not None:
        _append(reasons, d["window_start_utc"] < training_start, "antes_inicio_referencia")
    eligible, overlap = d["reference_eligible_by_policy"], d["simulation_overlap"]
    _append(reasons, eligible.isna(), "elegibilidade_nao_informada")
    _append(reasons, eligible.eq(0).fillna(False), "nao_elegivel_politica")
    _append(reasons, overlap.isna(), "simulacao_nao_informada")
    _append(reasons, overlap.eq(1).fillna(False), "simulacao_sobreposta")
    for origin in ORIGIN_STATUS:
        if origin not in REFERENCE_ORIGINS:
            _append(reasons, d["origin_status"] == origin, f"origem_{origin}")
    return reasons


def classify(d: pd.DataFrame, partition: pd.Series, training_start: pd.Timestamp | None) -> pd.DataFrame:
    quality = quality_reasons(d)
    reference = reference_reasons(d, training_start)
    states, reasons = [], []
    for position, part in enumerate(partition.to_numpy()):
        if part == "folga":
            states.append("folga")
            reasons.append(["folga:separacao_treino_teste"])
        elif part == "treino_candidato":
            codes = quality[position] + reference[position]
            states.append("excluido_treino" if codes else "treino_efetivo")
            reasons.append([f"treino:{c}" for c in codes])
        else:
            codes = quality[position]
            states.append("excluido_teste" if codes else "teste_analisado")
            reasons.append([f"teste:{c}" for c in codes])
    return pd.DataFrame({"particao_temporal": partition, "estado_analise": states, "motivos": reasons,
                         "qualidade_ok": [not q for q in quality]}, index=d.index)


def check_separation(d: pd.DataFrame, partition: pd.Series, cut: pd.Timestamp) -> dict:
    """Nenhuma janela de teste compartilha instante de observação com o treino candidato."""
    train_end = d.loc[partition == "treino_candidato", "window_end_utc"].max()
    test_start = d.loc[partition == "teste", "window_start_utc"].min()
    ok = (pd.isna(train_end) or train_end <= cut) and (pd.isna(test_start) or test_start > cut)
    if not ok:
        raise RuntimeError("separação temporal violada entre treino e teste")
    return {"fim_max_treino_candidato_utc": fmt_utc(train_end), "inicio_min_teste_utc": fmt_utc(test_start),
            "regra": "janelas (início, fim]: fim do treino ≤ corte < início do teste", "sem_sobreposicao": True}


def plan_analysis(ds: Dataset, settings: Settings) -> Plan:
    d = ds.data
    if settings.cut is not None:
        cut, mode, fraction = settings.cut, "explicito", None
    else:
        fraction = settings.fraction
        cut, mode = automatic_cut(d, fraction), "automatico"
    if settings.training_start is not None and settings.training_start >= cut:
        raise InputError(f"--inicio-treino ({fmt_utc(settings.training_start)}) deve ser anterior ao "
                         f"corte ({fmt_utc(cut)})")
    partition = assign_partitions(d["window_end_utc"], cut, ds.window_seconds)
    classes = classify(d, partition, settings.training_start)
    separation = check_separation(d, partition, cut)
    order = d.sort_values(["window_end_utc", "node_id"], kind="mergesort").index
    states = classes["estado_analise"].loc[order]
    train_index = order[(states == "treino_efetivo").to_numpy()]
    test_index = order[(states == "teste_analisado").to_numpy()]
    constant = []
    if len(train_index):
        spread = np.ptp(d.loc[train_index, list(FEATURES)].to_numpy(dtype=float), axis=0)
        constant = [f for f, s in zip(FEATURES, spread) if s == 0]
    return Plan(cut=cut, cut_mode=mode, fraction=fraction, training_start=settings.training_start,
                gap_end=cut + pd.Timedelta(seconds=ds.window_seconds), classes=classes,
                separation=separation, train_index=train_index, test_index=test_index,
                constant_features=constant)


# ---------------------------------------------------------------- modelo


def decide(scores, threshold: float) -> np.ndarray:
    """Sinaliza score estritamente maior que o limiar; empate no limiar não é sinalizado."""
    return np.asarray(scores, dtype=float) > threshold


def fit_and_score(X_train: np.ndarray, X_test: np.ndarray, *, n_estimators: int, seed: int,
                  percentile: float) -> dict:
    """Ajusta escala + Isolation Forest só no treino; limiar = percentil dos scores do treino."""
    pipeline = make_pipeline(
        RobustScaler(),
        IsolationForest(n_estimators=n_estimators, max_samples="auto", contamination="auto",
                        random_state=seed, n_jobs=1),
    )
    pipeline.fit(X_train)
    train_scores = -pipeline.score_samples(X_train)
    test_scores = -pipeline.score_samples(X_test) if len(X_test) else np.empty(0)
    threshold = float(np.percentile(train_scores, percentile, method="linear"))
    return {"pipeline": pipeline, "train_scores": train_scores, "test_scores": test_scores,
            "threshold": threshold, "is_outlier": decide(test_scores, threshold)}


def analyse(ds: Dataset, plan: Plan, settings: Settings) -> Result:
    n_train, n_test = len(plan.train_index), len(plan.test_index)
    if n_train < MIN_TRAIN_ROWS:
        raise InsufficientDataError(f"treino efetivo com {n_train} linha(s); são necessárias ao menos "
                                    f"{MIN_TRAIN_ROWS} para ajustar o modelo")
    if len(plan.constant_features) == len(FEATURES):
        raise InsufficientDataError("todas as 11 features são constantes no treino efetivo; o Isolation "
                                    "Forest não tem como isolar janelas")
    if n_test == 0:
        raise InsufficientDataError("nenhuma janela analisável no teste (depois do corte e da folga)")
    features = list(FEATURES)
    fitted = fit_and_score(ds.data.loc[plan.train_index, features].to_numpy(dtype=float),
                           ds.data.loc[plan.test_index, features].to_numpy(dtype=float),
                           n_estimators=settings.n_estimators, seed=settings.seed,
                           percentile=settings.percentile)
    scores = pd.Series(np.nan, index=ds.data.index)
    scores.loc[plan.train_index] = fitted["train_scores"]
    scores.loc[plan.test_index] = fitted["test_scores"]
    flagged = pd.Series(False, index=ds.data.index)
    flagged.loc[plan.test_index] = fitted["is_outlier"]
    threshold = fitted["threshold"]
    return Result(pipeline=fitted["pipeline"], threshold=threshold, scores=scores, flagged=flagged,
                  ties=int(np.sum(fitted["test_scores"] == threshold)),
                  train_above=float(np.mean(fitted["train_scores"] > threshold)))


# ---------------------------------------------------------------- tabelas e registro


def node_names(ds: Dataset) -> dict[int, str | None]:
    if "node_name" not in ds.data:
        return {}
    names = ds.data.dropna(subset=["node_name"]).groupby("node_id")["node_name"].first()
    return {int(k): str(v) for k, v in names.items()}


def node_label(node: int, names: dict) -> str:
    name = names.get(int(node))
    return f"{int(node)} ({name})" if name else str(int(node))


def _simulation_mask(d: pd.DataFrame) -> pd.Series:
    return d["simulation_overlap"].eq(1).fillna(False).astype(bool)


def _run_ids(d: pd.DataFrame, mask: pd.Series) -> tuple[list[str] | None, list[str] | None]:
    """`simulation_run_ids` das linhas: (ids de simulation_runs.csv, marcadores como legacy_node:<n>)."""
    if "simulation_run_ids" not in d:
        return None, None
    values = {x for ids in d.loc[mask, "simulation_run_ids"] if isinstance(ids, list) for x in ids}
    is_id = [x for x in values if isinstance(x, int) and not isinstance(x, bool)]
    return [str(x) for x in sorted(is_id)], sorted(str(x) for x in values if x not in is_id)


def count_runs(ends: pd.Series, step_seconds: int) -> int:
    """Trechos de janelas consecutivas da grade (fins separados por um passo) entre os fins dados."""
    if ends.empty:
        return 0
    gaps = ends.sort_values().diff().dt.total_seconds()
    return int(1 + (gaps > step_seconds).sum())


def count_summary(ds: Dataset, plan: Plan) -> tuple[dict, dict]:
    d, cls = ds.data, plan.classes
    state, part = cls["estado_analise"], cls["particao_temporal"]
    excluded = state.str.startswith("excluido")
    reasons = Counter(code for codes in cls.loc[excluded, "motivos"] for code in codes)
    nodes = sorted(int(n) for n in d["node_id"].unique())

    def nodes_without(mask: pd.Series) -> list[int]:
        present = set(int(n) for n in d.loc[mask, "node_id"])
        return [n for n in nodes if n not in present]

    overlap = d["simulation_overlap"]
    counts = {
        "entrada": len(d), "dispositivos": len(nodes),
        "treino_candidato": int((part == "treino_candidato").sum()),
        "treino_efetivo": int((state == "treino_efetivo").sum()),
        "excluido_treino": int((state == "excluido_treino").sum()),
        "folga": int((state == "folga").sum()),
        "teste": int((part == "teste").sum()),
        "teste_analisado": int((state == "teste_analisado").sum()),
        "excluido_teste": int((state == "excluido_teste").sum()),
        "excluidas_distintas": int(excluded.sum()),
        "linhas_com_mais_de_um_motivo": int(cls.loc[excluded, "motivos"].map(len).gt(1).sum()),
        "analisaveis_pela_qualidade": int(cls["qualidade_ok"].sum()),
        "feature_status": {s: int((d["feature_status"] == s).sum()) for s in FEATURE_STATUS},
        "origin_status": {s: int((d["origin_status"] == s).sum()) for s in ORIGIN_STATUS},
        "simulation_overlap": {"1": int(overlap.eq(1).fillna(False).sum()),
                               "0": int(overlap.eq(0).fillna(False).sum()), "nulo": int(overlap.isna().sum())},
        "com_default_declarado": int(d["defaulted_features"].map(lambda v: isinstance(v, list) and len(v) > 0).sum()),
        "com_feature_indisponivel": int(d["unavailable_features"].map(lambda v: isinstance(v, dict) and len(v) > 0).sum()),
        "com_feature_nula": int(d["feature_nula"].sum()),
        "com_feature_nao_finita": int(d["feature_nao_finita"].sum()),
        "unknown_no_treino_efetivo": int(((state == "treino_efetivo") & (d["origin_status"] == "unknown")).sum()),
        "dispositivos_sem_referencia_propria": nodes_without(state == "treino_efetivo"),
        "dispositivos_sem_teste_analisado": nodes_without(state == "teste_analisado"),
        "dispositivos_sem_janela_analisavel": nodes_without(cls["qualidade_ok"]),
    }
    return counts, dict(sorted(reasons.items()))


def simulation_summary(ds: Dataset, plan: Plan, result: Result | None) -> dict:
    d, cls = ds.data, plan.classes
    sim = _simulation_mask(d)
    state, part = cls["estado_analise"], cls["particao_temporal"]
    tested = state == "teste_analisado"
    train_runs, train_markers = _run_ids(d, sim & (part == "treino_candidato"))
    test_runs, test_markers = _run_ids(d, sim & tested)
    summary = {
        "coluna_simulation_run_ids": "simulation_run_ids" in d,
        "treino_candidato": int((sim & (part == "treino_candidato")).sum()),
        "treino_candidato_execucoes": train_runs, "treino_candidato_marcadores_legados": train_markers,
        "folga": int((sim & (part == "folga")).sum()),
        "teste": int((sim & (part == "teste")).sum()),
        "teste_analisado": int((sim & tested).sum()),
        "teste_analisado_execucoes": test_runs, "teste_analisado_marcadores_legados": test_markers,
        "teste_analisado_sem_simulacao": int((~sim & tested).sum()),
        "teste_sinalizadas": None, "teste_sinalizadas_sem_simulacao": None,
    }
    if result is not None:
        summary["teste_sinalizadas"] = int((sim & result.flagged).sum())
        summary["teste_sinalizadas_sem_simulacao"] = int((~sim & result.flagged).sum())
    return summary


def device_summary(ds: Dataset, plan: Plan, result: Result | None) -> pd.DataFrame:
    d, cls = ds.data, plan.classes
    state, part = cls["estado_analise"], cls["particao_temporal"]
    sim = _simulation_mask(d)
    names = node_names(ds)
    rows = []
    for node in sorted(d["node_id"].unique()):
        mine = d["node_id"] == node
        tested = mine & (state == "teste_analisado")
        n_test = int(tested.sum())
        row = {
            "node_id": int(node), "node_name": names.get(int(node)), "janelas_entrada": int(mine.sum()),
            "janelas_valid": int((mine & (d["feature_status"] == "valid")).sum()),
            "janelas_analisaveis": int((mine & cls["qualidade_ok"]).sum()),
            "treino_candidato": int((mine & (part == "treino_candidato")).sum()),
            "treino_efetivo": int((mine & (state == "treino_efetivo")).sum()),
            "excluido_treino": int((mine & (state == "excluido_treino")).sum()),
            "folga": int((mine & (state == "folga")).sum()),
            "teste": int((mine & (part == "teste")).sum()),
            "teste_analisado": n_test,
            "excluido_teste": int((mine & (state == "excluido_teste")).sum()),
            "outliers_teste": None, "taxa_outliers_teste": None,
            "teste_analisado_com_simulacao": int((tested & sim).sum()),
            "outliers_com_simulacao": None, "score_max_teste": None,
            "primeira_janela_fim_utc": d.loc[mine, "window_end_utc"].min(),
            "ultima_janela_fim_utc": d.loc[mine, "window_end_utc"].max(),
            "sem_referencia_propria": int(not (mine & (state == "treino_efetivo")).any()),
        }
        if result is not None:
            flagged = tested & result.flagged
            row["outliers_teste"] = int(flagged.sum())
            row["taxa_outliers_teste"] = row["outliers_teste"] / n_test if n_test else NOT_APPLICABLE
            row["outliers_com_simulacao"] = int((flagged & sim).sum())
            row["score_max_teste"] = float(result.scores[tested].max()) if n_test else None
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS), dtype=object)


def prediction_frame(ds: Dataset, plan: Plan, run_id: str, result: Result | None) -> pd.DataFrame:
    """Todas as linhas, na ordem do arquivo, com as colunas originais como lidas."""
    cls, index = plan.classes, ds.data.index
    states, reasons = cls["estado_analise"], cls["motivos"]
    if result is None:    # sem modelo, nenhuma linha foi analisada: o estado não pode dizer o contrário
        unscored = states.isin(["treino_efetivo", "teste_analisado"])
        states = states.replace({"treino_efetivo": "treino_sem_modelo", "teste_analisado": "teste_sem_modelo"})
        reasons = reasons.where(~unscored, pd.Series([["execucao:dados_insuficientes"]] * len(index), index=index))
    score = pd.Series(None, index=index, dtype=object)
    threshold = pd.Series(None, index=index, dtype=object)
    outlier = pd.Series(None, index=index, dtype=object)
    if result is not None:
        scored = result.scores.notna()
        score[scored] = result.scores[scored].map(lambda v: repr(float(v)))
        threshold[scored] = repr(result.threshold)
        tested = cls["estado_analise"] == "teste_analisado"
        outlier[tested] = result.flagged[tested].map(lambda v: "1" if v else "0")
    added = pd.DataFrame({
        "run_id": run_id, "detector": DETECTOR_ID, "detector_versao": DETECTOR_VERSION,
        "particao_temporal": cls["particao_temporal"], "estado_analise": states,
        "motivos_exclusao": reasons.map(lambda m: ";".join(m) if m else None),
        "anomaly_score": score, "limiar": threshold, "is_outlier": outlier,
    }, index=index)
    first = pd.DataFrame({"numero_linha_original": ds.data["numero_linha_original"]}, index=index)
    return pd.concat([first, ds.raw, added], axis=1)


def outlier_frame(ds: Dataset, predictions: pd.DataFrame, result: Result) -> pd.DataFrame:
    """Só o teste sinalizado, em score decrescente (empate: fim da janela, nó)."""
    idx = result.flagged[result.flagged].index
    keys = pd.DataFrame({"score": result.scores[idx], "fim": ds.data.loc[idx, "window_end_utc"],
                         "no": ds.data.loc[idx, "node_id"]})
    order = keys.sort_values(["score", "fim", "no"], ascending=[False, True, True], kind="mergesort").index
    return predictions.loc[order]


def caveats(ds: Dataset, plan: Plan, settings: Settings, counts: dict, simulations: dict) -> list[str]:
    """Ressalvas que dependem desta execução (as gerais estão em GENERAL_LIMITATIONS)."""
    out = []
    if not ds.provenance["completa"]:
        out.append(f"{ds.provenance['situacao']}: identidade e versões vêm só do que o arquivo traz.")
    n_train = counts["treino_efetivo"]
    if counts["unknown_no_treino_efetivo"]:
        out.append(f"{counts['unknown_no_treino_efetivo']} das {n_train} janelas do treino efetivo têm "
                   "origin_status = unknown: aceitas como não sintéticas, de proveniência não comprovada; "
                   "não são registros reais confirmados nem janelas saudáveis.")
    if n_train:
        first = ds.data.loc[plan.train_index, "window_start_utc"].min()
        last = ds.data.loc[plan.train_index, "window_end_utc"].max()
        hours = (last - first) / pd.Timedelta(hours=1)
        if hours < 24:
            out.append(f"O treino efetivo cobre {fmt_num(hours, 1)} h ({fmt_when(first)} a {fmt_when(last)}), "
                       "menos que um ciclo diário: variações que se repetem ao longo do dia e não aparecem "
                       "no treino podem surgir como isolamento no teste.")
    if 0 < n_train < SMALL_TRAIN_ROWS:
        out.append(f"Treino efetivo com {n_train} linhas (< {SMALL_TRAIN_ROWS}): referência pequena; cada "
                   f"árvore usa todas as {n_train} linhas. Mínimos operacionais não comprovam "
                   "suficiência científica.")
    if counts["dispositivos_sem_referencia_propria"]:
        out.append("Dispositivos sem referência própria (nenhuma janela no treino efetivo): "
                   f"{counts['dispositivos_sem_referencia_propria']}. O modelo global os pontua, com "
                   "comparabilidade limitada.")
    if plan.constant_features:
        out.append(f"Features constantes no treino efetivo: {', '.join(plan.constant_features)}. As "
                   "árvores não dividem por elas, então um valor diferente no teste nessas features "
                   "não aumenta o score; o RobustScaler só as centraliza.")
    if plan.training_start is None:
        out.append("Sem --inicio-treino: o treino candidato começa na primeira janela do arquivo; "
                   "nenhum período inicial (bancada, instalação) foi excluído por data.")
    if plan.cut_mode == "automatico":
        out.append(f"Corte automático (quantil {plan.fraction} dos fins das janelas valid): convenção "
                   "exploratória, não a escolha de um período saudável; pode deixar simulações fora do teste.")
    out.append(f"Limiar estimado no próprio treino efetivo (percentil {settings.percentile:g}): calibração "
               "exploratória, sem período independente de validação.")
    if simulations["teste_analisado"] == 0:
        out.append("Nenhuma janela com simulação ficou no teste analisado.")
    return out


def git_info() -> dict:
    info = {"git_commit": None, "git_script_versionado": None, "git_alteracoes_locais_no_script": None}

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=SCRIPT.parent, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()

    try:
        info["git_commit"] = git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return info
    try:
        git("ls-files", "--error-unmatch", SCRIPT.name)
        info["git_script_versionado"] = True
        info["git_alteracoes_locais_no_script"] = bool(git("status", "--porcelain", "--", SCRIPT.name))
    except subprocess.CalledProcessError:
        info["git_script_versionado"] = False
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for dist in ("pandas", "numpy", "scikit-learn", "matplotlib", "scipy", "joblib", "threadpoolctl"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def reproduction_command(ds: Dataset, plan: Plan, settings: Settings) -> str:
    parts = ["python", display_path(SCRIPT), "--entrada", display_path(settings.entrada)]
    if plan.cut_mode == "explicito":
        parts += ["--corte-treino", fmt_utc(plan.cut)]
    else:
        parts += ["--fracao-treino", repr(plan.fraction)]
    if plan.training_start is not None:
        parts += ["--inicio-treino", fmt_utc(plan.training_start)]
    parts += ["--percentil-limiar", repr(settings.percentile), "--seed", str(settings.seed),
              "--n-estimators", str(settings.n_estimators)]
    if Path(settings.saida).resolve() != DEFAULT_OUTPUT.resolve():
        parts += ["--saida", display_path(settings.saida)]
    return " ".join(shlex.quote(p) for p in parts)


def base_record(ds: Dataset, plan: Plan, settings: Settings, run_id: str | None, started: datetime) -> dict:
    counts, reasons = count_summary(ds, plan)
    simulations = simulation_summary(ds, plan, None)
    d = ds.data
    names = node_names(ds)
    return {
        "formato": "detector001-execucao-v1",
        "run_id": run_id,
        "estado": "em_execucao",
        "motivo_estado": "execução iniciada; se este estado permanecer, ela não terminou",
        "detector": {"id": DETECTOR_ID, "versao": DETECTOR_VERSION, "script": display_path(SCRIPT),
                     "script_sha256": sha256_file(SCRIPT), **git_info()},
        "inicio_utc": started, "fim_utc": None, "duracao_segundos": None,
        "comando": {"argv": ["python", display_path(SCRIPT), *settings.argv], "diretorio": display_path(os.getcwd()),
                    "reproducao": reproduction_command(ds, plan, settings),
                    "reproducao_diretorio": "<raiz do repositório>"},
        "argumentos": {"entrada": display_path(settings.entrada), "saida": display_path(settings.saida),
                       "corte_treino": settings.cut_text, "fracao_treino": settings.fraction,
                       "inicio_treino": fmt_utc(settings.training_start),
                       "percentil_limiar": settings.percentile, "seed": settings.seed,
                       "n_estimators": settings.n_estimators},
        "entrada": {
            "csv": display_path(ds.csv_path), "sha256_csv": ds.sha256, "bytes_csv": ds.size,
            "linhas": len(d), "colunas": list(ds.raw.columns), "dataset_id": ds.dataset_id,
            "dataset_id_origem": ds.provenance["dataset_id_origem"],
            "proveniencia": ds.provenance["situacao"], "manifest": ds.provenance["manifest"],
            "schema": ds.provenance["schema"], "verificacoes": ds.provenance["verificacoes"],
            "versoes": ds.provenance["versoes"], "versoes_fonte": ds.provenance["versoes_fonte"],
            "source_id": ds.source_id, "window_seconds": ds.window_seconds, "step_seconds": ds.step_seconds,
            "primeiro_inicio_utc": d["window_start_utc"].min(), "ultimo_fim_utc": d["window_end_utc"].max(),
            "dispositivos": [{"node_id": int(n), "node_name": names.get(int(n))}
                             for n in sorted(d["node_id"].unique())],
        },
        "features": list(FEATURES),
        "features_constantes_treino": plan.constant_features,
        "filtros": {
            "qualidade": "feature_status = valid, 11 features numéricas finitas, defaulted_features = [] e "
                         "unavailable_features = {}; sem imputação",
            "referencia": "treino candidato que passa na qualidade, com reference_eligible_by_policy = 1, "
                          "simulation_overlap = 0 e origin_status em {real, unknown}"
                          + ("; janela começa em ou depois de --inicio-treino" if plan.training_start is not None else ""),
            "teste": "todas as linhas do teste que passam na qualidade, de qualquer origem e com ou sem simulação",
        },
        "corte": {"instante_utc": plan.cut, "modo": plan.cut_mode, "valor_informado": settings.cut_text,
                  "fracao_treino": plan.fraction,
                  "regra": ("instante informado em --corte-treino, convertido para UTC" if plan.cut_mode == "explicito"
                            else f"quantil {plan.fraction} dos fins das janelas valid (pandas, linear), "
                                 "arredondado para baixo ao minuto")},
        "inicio_treino_utc": plan.training_start,
        "folga": {"segundos": ds.window_seconds, "de_exclusivo_utc": plan.cut, "ate_inclusivo_utc": plan.gap_end},
        "verificacao_sobreposicao": plan.separation,
        "modelo": {"pipeline": "make_pipeline(RobustScaler(), IsolationForest(...))",
                   "escala": "global (a mesma para todos os nós), ajustada só no treino efetivo",
                   "ordem_linhas": "window_end_utc, node_id (ordenação estável)",
                   "parametros_robust_scaler": None, "parametros_isolation_forest": None},
        "score": {"definicao": "anomaly_score = -pipeline.score_samples(X)",
                  "direcao": "maior = mais anômalo", "probabilidade": False,
                  "nota": "isolamento estatístico relativo ao treino; não é probabilidade nem percentual de anomalia"},
        "limiar": {"percentil": settings.percentile,
                   "metodo": "numpy.percentile(scores_treino_efetivo, percentil, method='linear')",
                   "valor": None, "comparacao": "anomaly_score > limiar",
                   "empates": "score igual ao limiar não é sinalizado", "empates_teste_no_limiar": None,
                   "fracao_treino_acima_limiar": None,
                   "nao_usado": "predict() e o limiar interno de contamination não entram na decisão"},
        "contagens": counts,
        "motivos_exclusao": reasons,
        "simulacoes": simulations,
        "ressalvas": caveats(ds, plan, settings, counts, simulations),
        "limitacoes_gerais": [re.sub(r"[*`]", "", t) for t in GENERAL_LIMITATIONS],
        "ambiente": {**library_versions(), "executavel": Path(sys.executable).name,
                     "plataforma": f"{platform.system()} {platform.machine()}"},
        "arquivos": {},
    }


def complete_record(record: dict, ds: Dataset, plan: Plan, result: Result) -> None:
    scaler, forest = result.pipeline[0], result.pipeline[-1]
    record["modelo"]["parametros_robust_scaler"] = scaler.get_params()
    record["modelo"]["parametros_isolation_forest"] = forest.get_params()
    record["modelo"]["max_samples_efetivo"] = int(forest.max_samples_)
    record["limiar"].update(valor=result.threshold, empates_teste_no_limiar=result.ties,
                            fracao_treino_acima_limiar=result.train_above)
    record["contagens"]["outliers_teste"] = int(result.flagged.sum())
    record["simulacoes"] = simulation_summary(ds, plan, result)


# ---------------------------------------------------------------- gráficos


def _naive_utc(series: pd.Series) -> pd.Series:
    return series.dt.tz_convert("UTC").dt.tz_localize(None)


def _merged_spans(starts, ends) -> list[list]:
    spans: list[list] = []
    for a, b in sorted(zip(starts, ends)):
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return spans


def _plot_series(path: Path, ds: Dataset, plan: Plan, result: Result) -> None:
    d, state = ds.data, plan.classes["estado_analise"]
    names = node_names(ds)
    nodes = sorted(d["node_id"].unique())
    ends = _naive_utc(d["window_end_utc"])
    starts = _naive_utc(d["window_start_utc"])
    sim = _simulation_mask(d)
    cut = plan.cut.tz_convert("UTC").tz_localize(None)
    gap_end = plan.gap_end.tz_convert("UTC").tz_localize(None)
    fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 1.2 + 1.9 * len(nodes)), sharex=True, squeeze=False)
    handles = {}
    for ax, node in zip(axes[:, 0], nodes):
        mine = d["node_id"] == node
        train, test = mine & (state == "treino_efetivo"), mine & (state == "teste_analisado")
        flagged = test & result.flagged
        handles["folga"] = ax.axvspan(cut, gap_end, color="#bdbdbd", alpha=0.7, lw=0)
        for a, b in _merged_spans(starts[mine & sim], ends[mine & sim]):
            handles["janela com simulação"] = ax.axvspan(a, b, color="#f0a830", alpha=0.35, lw=0)
        handles["treino efetivo"], = ax.plot(ends[train], result.scores[train], ".", ms=3, color="#6e6e6e")
        handles["teste analisado"], = ax.plot(ends[test], result.scores[test], ".", ms=3, color="#1f4e9c")
        handles["sinalizada (score > limiar)"], = ax.plot(ends[flagged], result.scores[flagged], "o", ms=6,
                                                          mfc="none", mec="#b03a2e", mew=1.3)
        handles["limiar"] = ax.axhline(result.threshold, color="#b03a2e", ls="--", lw=1)
        handles["corte"] = ax.axvline(cut, color="#222222", lw=0.9)
        ax.set_title(f"nó {node_label(node, names)}", loc="left", fontsize=9)
        ax.set_ylabel("score", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        if not (train | test).any():
            ax.text(0.5, 0.5, "sem janelas pontuadas", transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#444444")
    bottom = axes[-1, 0]
    locator = mdates.AutoDateLocator()
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    bottom.set_xlabel("fim da janela (UTC)", fontsize=9)
    fig.suptitle("anomaly_score por dispositivo (maior = mais anômalo)", fontsize=11, y=0.995)
    fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_distribution(path: Path, plan: Plan, result: Result) -> None:
    train = result.scores.loc[plan.train_index].to_numpy()
    test = result.scores.loc[plan.test_index].to_numpy()
    bins = np.histogram_bin_edges(np.concatenate([train, test]), bins=40)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for ax, values, color, title in ((axes[0], train, "#6e6e6e", f"treino efetivo (n = {len(train)})"),
                                     (axes[1], test, "#1f4e9c", f"teste analisado (n = {len(test)})")):
        ax.set_axisbelow(True)
        ax.hist(values, bins=bins, color=color)
        ax.axvline(result.threshold, color="#b03a2e", ls="--", lw=1.2,
                   label=f"limiar = {fmt_num(result.threshold)}")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("janelas", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    axes[1].set_xlabel("anomaly_score = −score_samples (maior = mais anômalo)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_devices(path: Path, summary: pd.DataFrame) -> None:
    labels = [str(n) for n in summary["node_id"]]
    counts = [int(v) for v in summary["outliers_teste"]]
    totals = [int(v) for v in summary["teste_analisado"]]
    x = np.arange(len(labels))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.9))
    left.set_axisbelow(True)
    right.set_axisbelow(True)
    left.bar(x, counts, color="#b03a2e")
    for xi, k, n in zip(x, counts, totals):
        left.annotate(f"{k}/{n}", (xi, k), ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                      textcoords="offset points")
    rates = [100 * k / n if n else 0.0 for k, n in zip(counts, totals)]
    right.bar(x, rates, color="#1f4e9c")
    for xi, rate, n in zip(x, rates, totals):
        text = NOT_APPLICABLE if n == 0 else fmt_num(rate, 2) + " %"
        right.annotate(text, (xi, rate), ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                       textcoords="offset points")
    left.set_title("janelas sinalizadas (k / teste analisado)", loc="left", fontsize=10)
    right.set_title("proporção sinalizada (% do teste analisado)", loc="left", fontsize=10)
    for ax in (left, right):
        ax.set_xticks(x, labels)
        ax.set_xlabel("dispositivo (node_id)", fontsize=9)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.margins(y=0.2)
    left.set_ylabel("janelas", fontsize=9)
    right.set_ylabel("%", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_charts(folder: Path, run_id: str, ds: Dataset, plan: Plan, result: Result,
                summary: pd.DataFrame) -> dict[str, Path]:
    folder.mkdir(exist_ok=True)
    paths = {kind: folder / f"{kind}__{run_id}.png"
             for kind in ("serie_scores", "distribuicao_scores", "outliers_por_dispositivo")}
    _plot_series(paths["serie_scores"], ds, plan, result)
    _plot_distribution(paths["distribuicao_scores"], plan, result)
    _plot_devices(paths["outliers_por_dispositivo"], summary)
    return paths


# ---------------------------------------------------------------- relatório


class Report:
    """Blocos simples renderizados em Markdown e em HTML autossuficiente."""

    def __init__(self) -> None:
        self.blocks: list[tuple] = []

    def heading(self, level: int, text: str) -> None:
        self.blocks.append(("h", level, text))

    def para(self, text: str, kind: str | None = None) -> None:
        self.blocks.append(("p", text, kind))

    def bullets(self, items) -> None:
        self.blocks.append(("ul", list(items)))

    def table(self, header: list[str], rows: list[list], numeric: tuple[int, ...] = ()) -> None:
        self.blocks.append(("table", header, [[("" if c is None else str(c)) for c in r] for r in rows],
                            set(numeric)))

    def figure(self, rel_path: str, png: bytes, caption: str) -> None:
        self.blocks.append(("fig", rel_path, png, caption))

    @staticmethod
    def _md(text: str) -> str:
        """Escapa só o que um visualizador de Markdown leria como tag HTML (texto vindo do CSV)."""
        parts = str(text).split("`")
        return "`".join(p if i % 2 else re.sub(r"<(?=[A-Za-z/!?])", "&lt;", p) for i, p in enumerate(parts))

    @staticmethod
    def _html(text: str) -> str:
        parts = str(text).split("`")
        out = []
        for i, part in enumerate(parts):
            if i % 2:
                out.append(f"<code>{html.escape(part)}</code>")
            else:
                out.append(re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(part)))
        return "".join(out)

    def markdown(self) -> str:
        lines: list[str] = []
        for block in self.blocks:
            kind = block[0]
            if kind == "h":
                lines += ["#" * block[1] + " " + self._md(block[2]), ""]
            elif kind == "p":
                prefix = "> " if block[2] else ""
                lines += [prefix + self._md(block[1]), ""]
            elif kind == "ul":
                lines += [f"- {self._md(item)}" for item in block[1]] + [""]
            elif kind == "table":
                _, header, rows, numeric = block

                def cell(text: str) -> str:
                    return self._md(text).replace("|", "\\|").replace("\n", " ")

                lines.append("| " + " | ".join(cell(h) for h in header) + " |")
                lines.append("|" + "|".join("---:" if i in numeric else "---" for i in range(len(header))) + "|")
                lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
                lines.append("")
            elif kind == "fig":
                lines += [f"![{self._md(block[3])}]({block[1]})", "", f"*{self._md(block[3])}*", ""]
        return "\n".join(lines).rstrip() + "\n"

    def html(self, title: str) -> str:
        body: list[str] = []
        for block in self.blocks:
            kind = block[0]
            if kind == "h":
                body.append(f"<h{block[1]}>{self._html(block[2])}</h{block[1]}>")
            elif kind == "p":
                css = f' class="{block[2]}"' if block[2] else ""
                body.append(f"<p{css}>{self._html(block[1])}</p>")
            elif kind == "ul":
                body.append("<ul>" + "".join(f"<li>{self._html(i)}</li>" for i in block[1]) + "</ul>")
            elif kind == "table":
                _, header, rows, numeric = block
                head = "".join(f"<th>{self._html(h)}</th>" for h in header)
                lines = []
                for row in rows:
                    cells = "".join(f'<td class="num">{self._html(c)}</td>' if i in numeric
                                    else f"<td>{self._html(c)}</td>" for i, c in enumerate(row))
                    lines.append(f"<tr>{cells}</tr>")
                body.append(f'<div class="tabela"><table><thead><tr>{head}</tr></thead>'
                            f'<tbody>{"".join(lines)}</tbody></table></div>')
            elif kind == "fig":
                data = base64.b64encode(block[2]).decode("ascii")
                caption = self._html(block[3])
                body.append(f'<figure><img alt="{html.escape(re.sub(r"[`*]", "", block[3]))}" '
                            f'src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>')
        return HTML_TEMPLATE.format(title=html.escape(title), body="\n".join(body))


HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; background: #ffffff; color: #1a1a1a; line-height: 1.5;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "DejaVu Sans", Arial, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 1.6rem; margin: 0 0 .4rem; padding-bottom: .4rem; border-bottom: 3px solid #1f4e9c; }}
h2 {{ font-size: 1.25rem; margin-top: 2.2rem; padding-bottom: .2rem; color: #12325f; border-bottom: 1px solid #c4cad3; }}
h3 {{ font-size: 1.05rem; margin-top: 1.4rem; color: #12325f; }}
p, li {{ max-width: 80ch; }}
code {{ font-family: ui-monospace, "DejaVu Sans Mono", Consolas, monospace; font-size: .86em;
  background: #eef1f5; padding: 0 .25em; border-radius: 3px; overflow-wrap: anywhere; }}
.tabela {{ overflow-x: auto; margin: .6rem 0 1.1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: .86rem; }}
th, td {{ border: 1px solid #c4cad3; padding: 4px 8px; text-align: left; vertical-align: top; }}
th {{ background: #e8edf4; color: #12325f; }}
tbody tr:nth-child(even) {{ background: #f7f9fb; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
figure {{ margin: 1rem 0 1.4rem; }}
figure img {{ max-width: 100%; height: auto; border: 1px solid #d3d8df; }}
figcaption {{ font-size: .86rem; color: #3d3d3d; margin-top: .3rem; }}
p.aviso, p.erro {{ max-width: none; padding: 8px 12px; border-radius: 3px; }}
p.aviso {{ background: #fdf4e3; border-left: 4px solid #9a6200; }}
p.erro {{ background: #fbeceb; border-left: 4px solid #a33025; }}
@media print {{
  body {{ font-size: 10pt; }}
  main {{ max-width: none; padding: 0; }}
  h2, h3 {{ break-after: avoid; }}
  table, figure {{ break-inside: avoid; }}
  .tabela {{ overflow: visible; }}
  tbody tr:nth-child(even) {{ background: #f2f2f2; }}
}}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def _reason_text(code: str) -> str:
    _, _, bare = code.partition(":")
    if bare.startswith("estado_"):
        return f"`feature_status` = {bare[len('estado_'):]}"
    if bare.startswith("origem_"):
        return f"`origin_status` = {bare[len('origem_'):]} (fora da referência)"
    return REASON_TEXT.get(bare, bare)


def build_report(ds: Dataset, plan: Plan, settings: Settings, record: dict, result: Result | None,
                 summary: pd.DataFrame, charts: dict[str, Path] | None) -> Report:
    rep = Report()
    c, sims, prov = record["contagens"], record["simulacoes"], ds.provenance
    d, cls = ds.data, plan.classes
    names = node_names(ds)
    run_id = record["run_id"]
    started = pd.Timestamp(record["inicio_utc"])
    W, S = ds.window_seconds, ds.step_seconds

    rep.heading(1, "Relatório de detecção de outliers — Isolation Forest")
    rep.para(f"Gerado por `{DETECTOR_ID}` {DETECTOR_VERSION} em {fmt_when(started)}. As janelas "
             "sinalizadas são **candidatas a anomalia estatística**, não diagnósticos de falha física.")
    if result is None:
        rep.para(f"**Execução encerrada sem detecção** (estado `{record['estado']}`): "
                 f"{record['motivo_estado']}. Nenhum score ou janela sinalizada é apresentado; as "
                 "contagens abaixo servem de diagnóstico. Nas predições, as linhas que seriam pontuadas "
                 "têm estado `treino_sem_modelo` ou `teste_sem_modelo`.", kind="erro")

    versions = ", ".join(f"`{k}` = `{v}` ({prov['versoes_fonte'].get(k, '—')})"
                         for k, v in prov["versoes"].items()) or "não informadas"
    devices = ", ".join(node_label(n, names) for n in sorted(d["node_id"].unique()))
    rep.heading(2, "Identificação")
    rep.table(["Campo", "Valor"], [
        ["Execução", f"`{run_id}`"],
        ["Detector / versão", f"`{DETECTOR_ID}` / {DETECTOR_VERSION} (script SHA-256 "
                              f"`{record['detector']['script_sha256'][:16]}…`)"],
        ["Dataset", f"`{ds.dataset_id}` ({prov['dataset_id_origem']})"],
        ["Arquivo analisado / SHA-256", f"`{display_path(ds.csv_path)}` / `{ds.sha256}`"],
        ["Data da análise", fmt_when(started)],
        ["Período / dispositivos", f"{fmt_when(d['window_start_utc'].min())} a {fmt_when(d['window_end_utc'].max())} "
                                   f"/ {c['dispositivos']}: {devices}"],
        ["Janela / passo", f"{W} s / {S} s"],
        ["Fonte (`source_id`)", f"`{ds.source_id}`"],
        ["Versões dos dados (fonte)", versions],
        ["Proveniência", prov["situacao"] + (": " + "; ".join(prov["verificacoes"]) if prov["verificacoes"] else "")],
        ["Estado da execução", f"`{record['estado']}`"],
    ])

    # 1. Resumo executivo
    rep.heading(2, "1. Resumo executivo")
    text = (f"Foram recebidas {c['entrada']} janelas de {c['dispositivos']} dispositivos. Após os filtros de "
            f"qualidade e a separação temporal, {c['treino_efetivo']} janelas compuseram a referência de "
            f"treino e {c['teste_analisado']} foram analisadas no teste.")
    if result is None:
        text = (f"Foram recebidas {c['entrada']} janelas de {c['dispositivos']} dispositivos. Após os filtros "
                f"de qualidade e a separação temporal, {c['treino_efetivo']} janelas formariam a referência de "
                f"treino e {c['teste_analisado']} seriam analisáveis no teste, mas nenhum modelo foi ajustado.")
    else:
        flagged = int(result.flagged.sum())
        text += (f" Foram sinalizadas {flagged} janelas, correspondentes a "
                 f"{fmt_rate(flagged, c['teste_analisado'])} das janelas efetivamente analisadas no teste.")
    rep.para(text)
    if result is not None:
        tested = cls["estado_analise"] == "teste_analisado"
        items = []
        for node in sorted(d["node_id"].unique()):
            mine = (d["node_id"] == node) & result.flagged
            if mine.any():
                ends = d.loc[mine, "window_end_utc"]
                runs = count_runs(ends, S)
                items.append(f"nó {node_label(node, names)}: {int(mine.sum())} de "
                             f"{int((tested & (d['node_id'] == node)).sum())} janelas analisadas, em {runs} "
                             f"{'trecho' if runs == 1 else 'trechos'} de janelas consecutivas, a primeira com fim "
                             f"em {fmt_when(ends.min())} e a última em {fmt_when(ends.max())}; maior score "
                             f"{fmt_num(result.scores[mine].max())}")
        if items:
            rep.para("Dispositivos e períodos que merecem investigação (sem atribuição de causa física):")
            rep.bullets(items)
        else:
            rep.para("Nenhuma janela do teste ficou acima do limiar. Isso não comprova ausência de falhas: "
                     "indica só que nenhuma janela analisada ficou mais isolada que o percentil "
                     f"{settings.percentile:g} do treino.")
        if c["teste_analisado"]:
            rep.para(f"Por construção, cerca de {fmt_num(100 - settings.percentile, 2)} % do treino efetivo fica "
                     f"acima do limiar (aqui {fmt_num(100 * result.train_above, 2)} %), porque o limiar é o percentil "
                     f"{settings.percentile:g} dos scores do próprio treino. Compare a taxa do teste com essa "
                     "proporção; ela não é taxa de falhas. Mediana do score: "
                     f"{fmt_num(result.scores.loc[plan.train_index].median())} no treino efetivo e "
                     f"{fmt_num(result.scores.loc[plan.test_index].median())} no teste analisado.")
        if sims["teste_analisado"]:
            rep.para(f"No teste analisado, {sims['teste_analisado']} janelas se sobrepõem a simulações e "
                     f"{sims['teste_sinalizadas']} delas foram sinalizadas (§5).")
    rep.para(f"Janelas excluídas ({c['excluidas_distintas']}) e as da folga ({c['folga']}) não entram no "
             "denominador da taxa. Janelas vizinhas se sobrepõem: N janelas sinalizadas não equivalem a N "
             "incidentes independentes.")

    # 2. Dados e qualidade
    rep.heading(2, "2. Dados e qualidade")
    rep.para("Cada contador conta linhas de `features.csv` (um dispositivo × uma janela). Treino efetivo, "
             "folga, teste analisado e excluídas são disjuntos e somam a entrada.")
    rep.table(["Contador", "Janelas", "Definição"], [
        ["Entrada", c["entrada"], "linhas de `features.csv`"],
        ["Treino candidato", c["treino_candidato"], f"fim da janela ≤ corte ({fmt_when(plan.cut)})"],
        ["Treino efetivo", c["treino_efetivo"], "treino candidato que passa na qualidade e na seleção da referência"],
        ["Excluídas do treino", c["excluido_treino"], "treino candidato com ao menos um motivo de exclusão"],
        ["Folga", c["folga"], f"corte < fim ≤ corte + W ({fmt_when(plan.gap_end)}); sem score, separa treino e teste"],
        ["Teste", c["teste"], "fim da janela > corte + W"],
        ["Teste analisado", c["teste_analisado"], "teste que passa na qualidade; "
         + ("recebe score e decisão" if result is not None else "receberia score e decisão (sem modelo nesta execução)")],
        ["Excluídas do teste", c["excluido_teste"], "teste com ao menos um motivo de qualidade"],
        ["Excluídas (distintas)", c["excluidas_distintas"], "excluídas do treino + excluídas do teste; cada linha conta uma vez"],
        ["Janelas sinalizadas", "—" if result is None else int(result.flagged.sum()),
         "teste analisado com `anomaly_score` > limiar"],
    ], numeric=(1,))
    rep.para(f"Conferência: {c['treino_efetivo']} + {c['folga']} + {c['teste_analisado']} + "
             f"{c['excluidas_distintas']} = {c['treino_efetivo'] + c['folga'] + c['teste_analisado'] + c['excluidas_distintas']} "
             f"(entrada: {c['entrada']}).")
    rep.heading(3, "Estados, origens, simulações e defaults")
    rep.table(["Coluna", "Valor", "Janelas"],
              [["`feature_status`", s, n] for s, n in c["feature_status"].items()]
              + [["`origin_status`", s, n] for s, n in c["origin_status"].items()]
              + [["`simulation_overlap`", s, n] for s, n in c["simulation_overlap"].items()]
              + [["`defaulted_features`", "não vazio", c["com_default_declarado"]],
                 ["`unavailable_features`", "não vazio", c["com_feature_indisponivel"]],
                 ["11 features", "alguma vazia (nula)", c["com_feature_nula"]],
                 ["11 features", "alguma NaN ou infinita", c["com_feature_nao_finita"]],
                 ["qualidade", "analisáveis em qualquer partição", c["analisaveis_pela_qualidade"]]],
              numeric=(2,))
    rep.para("`origin_status = unknown` aparece como desconhecida: proveniência não comprovada, nem real "
             "confirmada nem sintética.")
    rep.heading(3, "Motivos de exclusão")
    reasons = record["motivos_exclusao"]
    if reasons:
        rep.table(["Motivo", "Linhas", "Significado"],
                  [[f"`{code}`", n, _reason_text(code)] for code, n in reasons.items()], numeric=(1,))
        rep.para(f"Uma linha pode ter mais de um motivo ({c['linhas_com_mais_de_um_motivo']} linhas têm mais de "
                 f"um): a soma dos motivos ({sum(reasons.values())}) passa do total de excluídas distintas "
                 f"({c['excluidas_distintas']}). O prefixo diz a partição (`treino:` ou `teste:`). "
                 "`reference_exclusion_reason`, quando existir, complementa o diagnóstico nas predições. As "
                 f"{c['folga']} janelas da folga têm o motivo `folga:separacao_treino_teste` e não contam como excluídas.")
    else:
        rep.para("Nenhuma linha foi excluída.")
    rep.heading(3, "Cobertura por dispositivo")
    rep.table(["Dispositivo", "Janelas", "Primeiro fim", "Último fim", "valid", "Analisáveis",
               "Treino efetivo", "Teste analisado"],
              [[node_label(r.node_id, names), r.janelas_entrada, fmt_when(r.primeira_janela_fim_utc),
                fmt_when(r.ultima_janela_fim_utc), r.janelas_valid, r.janelas_analisaveis, r.treino_efetivo,
                r.teste_analisado] for r in summary.itertuples()], numeric=(1, 4, 5, 6, 7))
    lacks = []
    if c["dispositivos_sem_janela_analisavel"]:
        lacks.append(f"sem nenhuma janela analisável: {c['dispositivos_sem_janela_analisavel']}")
    if c["dispositivos_sem_referencia_propria"]:
        lacks.append(f"sem referência própria (nenhuma janela no treino efetivo): {c['dispositivos_sem_referencia_propria']}")
    if c["dispositivos_sem_teste_analisado"]:
        lacks.append(f"sem teste analisado: {c['dispositivos_sem_teste_analisado']}")
    rep.para("Dispositivos " + "; ".join(lacks) + "." if lacks else "Todos os dispositivos têm treino efetivo e teste analisado.")
    rep.para(f"Proveniência: {prov['situacao']}. Versões: {versions}. Fonte `{ds.source_id}`.")

    # 3. Método e reprodução
    rep.heading(2, "3. Método e reprodução")
    start_text = (f"janelas que começam antes de {fmt_when(plan.training_start)} (`--inicio-treino`) ficam fora "
                  "da referência" if plan.training_start is not None else
                  "não informado: o treino candidato começa na primeira janela do arquivo")
    cut_text = (f"{fmt_when(plan.cut)}, informado pelo usuário (`--corte-treino {settings.cut_text}`)"
                if plan.cut_mode == "explicito" else
                f"{fmt_when(plan.cut)}, automático: quantil {plan.fraction} dos fins das janelas valid, "
                "arredondado para baixo ao minuto (convenção exploratória)")
    rep.bullets([
        "Entradas do modelo (11, nesta ordem): " + ", ".join(f"`{f}`" for f in FEATURES) + ". Identificadores, "
        "datas, contagens, origem, simulação e estados ficam fora da matriz.",
        "Qualidade: uma linha é analisável se `feature_status = valid`, as 11 features são numéricas finitas, "
        "`defaulted_features = []` e `unavailable_features = {}`. Não há imputação.",
        "Referência (treino efetivo): treino candidato analisável com `reference_eligible_by_policy = 1`, "
        "`simulation_overlap = 0` e `origin_status` em {real, unknown}. O teste analisa todas as origens.",
        f"Corte único para todos os dispositivos: {cut_text}.",
        f"Início da referência: {start_text}.",
        f"Folga de W = {W} s: janelas com fim em ({fmt_when(plan.cut)}, {fmt_when(plan.gap_end)}] ficam sem score. "
        f"Verificação: último fim do treino candidato {plan.separation['fim_max_treino_candidato_utc'] or '—'}; "
        f"primeiro início do teste {plan.separation['inicio_min_teste_utc'] or '—'}; sem instante de observação comum.",
        "Ordem das linhas antes do ajuste: (`window_end_utc`, `node_id`), estável; sem embaralhamento nem divisão aleatória.",
        f"Modelo global: `make_pipeline(RobustScaler(), IsolationForest(n_estimators={settings.n_estimators}, "
        f"max_samples=\"auto\", contamination=\"auto\", random_state={settings.seed}, n_jobs=1))`, ajustado só no "
        "treino efetivo. A escala é global (não por nó) e não é reajustada no teste.",
        "Score: `anomaly_score = -pipeline.score_samples(X)`; maior = mais anômalo. Expressa isolamento "
        "estatístico relativo ao treino; não é probabilidade de falha nem percentual de anomalia.",
    ])
    if result is not None:
        rep.para(f"Limiar: percentil {settings.percentile:g} dos scores do treino efetivo, "
                 f"`numpy.percentile(..., method=\"linear\")` = {fmt_num(result.threshold)} (`{result.threshold!r}`). "
                 f"Regra: `anomaly_score > limiar` (estrita); {result.ties} janela(s) de teste com score igual ao "
                 f"limiar (não sinalizadas). Fração do treino acima do limiar: {fmt_num(100 * result.train_above, 2)} %. "
                 "`predict()` e o limiar interno de `contamination` não entram na decisão.")
        rep.para(f"Árvores com `max_samples` efetivo = {int(result.pipeline[-1].max_samples_)} linhas.")
    if plan.constant_features:
        rep.para("Features constantes no treino efetivo: " + ", ".join(f"`{f}`" for f in plan.constant_features)
                 + ". As árvores não dividem por elas; um valor diferente no teste nessas features não aumenta o score.",
                 kind="aviso")
    env = record["ambiente"]
    rep.para("Bibliotecas: " + ", ".join(f"{k} {env[k]}" for k in ("python", "pandas", "numpy", "scikit-learn", "matplotlib")) + ".")
    rep.para(f"Comando executado: `{' '.join(shlex.quote(a) for a in record['comando']['argv'])}`")
    rep.para(f"Reprodução, a partir da raiz do repositório: `{record['comando']['reproducao']}`")

    # 4. Resultados
    rep.heading(2, "4. Resultados no período de teste")
    if result is None:
        rep.para("Sem resultados: a execução terminou antes do ajuste do modelo.")
    else:
        rep.table(["Dispositivo", "Treino", "Teste analisado", "Janelas sinalizadas", "Taxa no teste"],
                  [[node_label(r.node_id, names) + (" — sem referência própria" if r.sem_referencia_propria else ""),
                    r.treino_efetivo, r.teste_analisado, r.outliers_teste,
                    fmt_rate(int(r.outliers_teste), int(r.teste_analisado))] for r in summary.itertuples()],
                  numeric=(1, 2, 3, 4))
        rep.para("Treino = janelas do dispositivo no treino efetivo; taxa = sinalizadas ÷ teste analisado do "
                 f"dispositivo (`{NOT_APPLICABLE}` quando o teste analisado é 0). O modelo é global: um dispositivo sem "
                 "referência própria é pontuado contra a referência dos demais, com comparabilidade limitada.")
        for kind, caption in (
            ("serie_scores", "Score por dispositivo ao longo do tempo: treino efetivo (cinza), teste analisado (azul), "
                             "janelas sinalizadas (círculos vermelhos), limiar (tracejado), corte (linha preta), folga "
                             "(faixa cinza) e janelas com `simulation_overlap = 1` (faixa laranja)."),
            ("distribuicao_scores", "Distribuição dos scores do treino efetivo e do teste analisado, com o limiar."),
            ("outliers_por_dispositivo", "Janelas sinalizadas por dispositivo e proporção sobre o teste analisado "
                                         f"(`{NOT_APPLICABLE}` quando não há teste analisado)."),
        ):
            path = charts[kind]
            rep.figure(f"graficos/{path.name}", path.read_bytes(), caption)
        top = result.scores.loc[plan.test_index].sort_values(ascending=False, kind="mergesort").head(TOP_N)
        rep.heading(3, f"Até {TOP_N} janelas de teste com maior score")
        has_ids = "simulation_run_ids" in d
        rows = []
        sim_mask = _simulation_mask(d)
        for rank, (i, score) in enumerate(top.items(), 1):
            sim = "sim" if sim_mask[i] else ("não informado" if pd.isna(d.at[i, "simulation_overlap"]) else "não")
            if has_ids and d.at[i, "simulation_run_ids"]:
                sim += " (" + ", ".join(str(x) for x in d.at[i, "simulation_run_ids"]) + ")"
            rows.append([rank, node_label(d.at[i, "node_id"], names), fmt_when(d.at[i, "window_start_utc"]),
                         fmt_when(d.at[i, "window_end_utc"]), fmt_num(score), "sim" if result.flagged[i] else "não",
                         d.at[i, "origin_status"], sim, int(d.at[i, "numero_linha_original"])])
        rep.table(["#", "Dispositivo", "Início", "Fim", "Score", "Sinalizada", "Origem", "Simulação", "Linha"],
                  rows, numeric=(0, 4, 8))
        rep.heading(3, "Valores observados das features nessas janelas")
        median = d.loc[plan.train_index, list(FEATURES)].median()
        values = [[rank, node_label(d.at[i, "node_id"], names), fmt_when(d.at[i, "window_end_utc"])]
                  + [fmt_num(d.at[i, f], 3) for f in FEATURES] for rank, i in enumerate(top.index, 1)]
        values.append(["—", "mediana do treino efetivo", "—"] + [fmt_num(median[f], 3) for f in FEATURES])
        rep.table(["#", "Dispositivo", "Fim"] + [f"`{f}`" for f in FEATURES], values,
                  numeric=tuple(range(3, 3 + len(FEATURES))))
        rep.para("A comparação com a mediana do treino é descritiva: não explica por que o Isolation Forest "
                 "isolou a janela nem indica causa física.")

    # 5. Simulações
    rep.heading(2, "5. Simulações e evidências de campo")
    if sims["teste_analisado"] == 0:
        rep.para(f"Nenhuma janela com simulação ficou no teste analisado ({sims['teste']} no teste antes da "
                 f"qualidade; {sims['treino_candidato']} no treino candidato, fora da referência; {sims['folga']} na folga).")
    else:
        runs, markers = sims["teste_analisado_execucoes"], sims["teste_analisado_marcadores_legados"]
        text = f"No teste analisado, {sims['teste_analisado']} janelas têm `simulation_overlap = 1`"
        if runs is None:
            text += " (o arquivo não traz `simulation_run_ids`)"
        else:
            text += (f", ligadas a {len(runs)} {'execução' if len(runs) == 1 else 'execuções'} de "
                     f"`simulation_runs.csv`" + (f" (`id` {', '.join(runs)})" if runs else ""))
            if markers:
                text += (f" e aos marcadores {', '.join(f'`{m}`' for m in markers)}, que guardam só a última "
                         "simulação de cada nó e podem repetir uma execução já listada por `id` (confira em "
                         "`simulation_runs.csv`)")
        rep.para(text + f". No treino candidato, {sims['treino_candidato']} janelas com simulação ficaram fora "
                 f"da referência; na folga, {sims['folga']}. As janelas se sobrepõem e incluem as dos vizinhos "
                 "do nó simulado: contam janelas, não simulações.")
    if result is not None and c["teste_analisado"]:
        rep.table(["Grupo do teste analisado", "Janelas", "Sinalizadas", "Proporção"], [
            ["com `simulation_overlap = 1`", sims["teste_analisado"], sims["teste_sinalizadas"],
             fmt_rate(sims["teste_sinalizadas"], sims["teste_analisado"])],
            ["sem simulação marcada", sims["teste_analisado_sem_simulacao"], sims["teste_sinalizadas_sem_simulacao"],
             fmt_rate(sims["teste_sinalizadas_sem_simulacao"], sims["teste_analisado_sem_simulacao"])],
        ], numeric=(1, 2, 3))
    rep.para("O cruzamento é exploratório. `simulation_overlap` marca uma intervenção conhecida (também nas "
             "janelas de vizinhos que recebem registros sintéticos pelos enlaces reversos): não é rótulo de "
             "falha observável, e as demais janelas não são normais por não terem simulação. Sem rótulos "
             "verificados, não se calcula acurácia, precisão, recall nem F1.")
    rep.para("Para investigar os achados: `simulation_runs.csv` (perfil, nó alvo e horários de cada execução), "
             "`telemetry.csv` e `neighbors.csv` (registros brutos, com `data_origin`) do pacote Completo, e as "
             "anotações de campo. O `events_template.csv` original é um formulário vazio, não um conjunto de "
             "rótulos validado.")

    # 6. Limitações
    rep.heading(2, "6. Limitações e interpretação")
    rep.para("Desta execução:")
    rep.bullets(record["ressalvas"])
    rep.para("Gerais deste detector e dos dados exportados:")
    rep.bullets(GENERAL_LIMITATIONS)
    rep.para("A calibração do limiar no próprio treino é exploratória. Zero detecções é um resultado possível "
             "e válido. O programa estima o limiar só com o treino efetivo; escolher o percentil (ou o corte) "
             "depois de ver o resultado do teste e executar de novo desfaz essa separação.")

    # 7. Arquivos e próximos passos
    rep.heading(2, "7. Arquivos e próximos passos")
    rep.para(f"Todos na pasta da execução, com `<run_id>` = `{run_id}`:")
    files = [
        ("relatorio__<run_id>.html", "este relatório, com os gráficos incorporados"),
        ("relatorio__<run_id>.md", "versão Markdown, com os gráficos em `graficos/`"),
        ("predicoes__<run_id>.csv", "todas as linhas da entrada, com partição, estado, motivos, score, limiar e decisão"),
    ]
    if result is not None:
        files.append(("outliers__<run_id>.csv", "só as janelas de teste sinalizadas, em score decrescente"))
    files += [
        ("resumo_por_dispositivo__<run_id>.csv", "contagens e taxa por dispositivo"),
        ("execucao__<run_id>.json", "identidade, hashes, parâmetros, corte, limiar, contagens e ressalvas"),
        ("ambiente__<run_id>.txt", "versões de Python e bibliotecas e comando de reprodução"),
    ]
    if charts:
        files += [(f"graficos/{kind}__<run_id>.png", "gráfico") for kind in charts]
    rep.bullets(f"`{name}`: {desc}" for name, desc in files)
    if result is not None and result.flagged.any():
        top_flagged = result.scores[result.flagged].sort_values(ascending=False, kind="mergesort").head(5)
        rep.para("Janelas prioritárias para conferência com os dados brutos, as anotações de campo e as simulações:")
        rep.bullets(f"nó {node_label(d.at[i, 'node_id'], names)}, janela ({fmt_when(d.at[i, 'window_start_utc'])}, "
                    f"{fmt_when(d.at[i, 'window_end_utc'])}], score {fmt_num(s)} (linha {int(d.at[i, 'numero_linha_original'])})"
                    for i, s in top_flagged.items())
    rep.para("Para uso em artigo: defina antes a referência com a cronologia da campanha, calibre o limiar em um "
             "período cronológico independente e só então avalie o teste, uma vez.")
    return rep


# ---------------------------------------------------------------- execução


def create_run_dir(root: Path, dataset_id: str, now: datetime, suffix=lambda: secrets.token_hex(4)) -> Path:
    """Cria a pasta nova da execução; nunca reutiliza uma pasta existente."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset_id).strip("_-")[:80] or "dataset"
    stamp = pd.Timestamp(now).tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        folder = root / f"{DETECTOR_ID}__{safe}__{stamp}__{suffix()}"
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise RuntimeError(f"não foi possível criar uma pasta nova em {root}")


def write_environment(path: Path, record: dict) -> None:
    env = record["ambiente"]
    lines = [f"Detector: {DETECTOR_ID} {DETECTOR_VERSION}", f"Execução: {record['run_id']}",
             f"Python: {env['python']} ({env['executavel']})", f"Plataforma: {env['plataforma']}", ""]
    lines += [f"{name}=={env[name]}" for name in ("pandas", "numpy", "scikit-learn", "matplotlib")]
    lines += ["", "Dependências indiretas:"]
    lines += [f"{name}=={env[name]}" for name in ("scipy", "joblib", "threadpoolctl")]
    lines += ["", f"Commit Git: {record['detector']['git_commit'] or 'indisponível'}",
              f"Script: {record['detector']['script']} (SHA-256 {record['detector']['script_sha256']})", "",
              f"Comando executado (diretório {record['comando']['diretorio']}):",
              " ".join(shlex.quote(a) for a in record["comando"]["argv"]), "",
              f"Comando de reprodução (a partir de {record['comando']['reproducao_diretorio']}):",
              record["comando"]["reproducao"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(folder: Path, run_id: str, ds: Dataset, plan: Plan, settings: Settings, record: dict,
                  result: Result | None) -> None:
    predictions = prediction_frame(ds, plan, run_id, result)
    write_csv(folder / f"predicoes__{run_id}.csv", predictions)
    summary = device_summary(ds, plan, result)
    charts = None
    if result is not None:
        write_csv(folder / f"outliers__{run_id}.csv", outlier_frame(ds, predictions, result))
    write_csv(folder / f"resumo_por_dispositivo__{run_id}.csv", summary)
    write_environment(folder / f"ambiente__{run_id}.txt", record)
    if result is not None:
        charts = plot_charts(folder / "graficos", run_id, ds, plan, result, summary)
    report = build_report(ds, plan, settings, record, result, summary, charts)
    markdown, html = report.markdown(), report.html(f"Relatório {run_id}")   # os dois antes de gravar
    (folder / f"relatorio__{run_id}.md").write_text(markdown, encoding="utf-8")
    (folder / f"relatorio__{run_id}.html").write_text(html, encoding="utf-8")


def execute(ds: Dataset, plan: Plan, settings: Settings) -> int:
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    record = base_record(ds, plan, settings, None, started)
    try:
        folder = create_run_dir(settings.saida, ds.dataset_id, started)
    except OSError as exc:
        print(f"erro: não foi possível criar a pasta da execução em {display_path(settings.saida)}: {exc}",
              file=sys.stderr)
        return 2
    run_id = record["run_id"] = folder.name
    record_path = folder / f"execucao__{run_id}.json"
    try:
        write_json(record_path, record)
        try:
            result = analyse(ds, plan, settings)
        except InsufficientDataError as exc:
            record["estado"], record["motivo_estado"] = "dados_insuficientes", str(exc)
            write_outputs(folder, run_id, ds, plan, settings, record, None)
            code = 3
        else:
            complete_record(record, ds, plan, result)
            record["estado"], record["motivo_estado"] = "concluida", None
            write_outputs(folder, run_id, ds, plan, settings, record, result)
            code = 0
    except Exception as exc:  # a pasta já existe: o registro precisa dizer que não terminou
        record["estado"], record["motivo_estado"] = "falhou", f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        for partial in folder.glob("relatorio__*"):   # relatório só fica se a execução terminou
            partial.unlink(missing_ok=True)
        code = 1
    record["fim_utc"] = datetime.now(timezone.utc)
    record["duracao_segundos"] = round(time.perf_counter() - clock, 3)
    record["arquivos"] = {p.relative_to(folder).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                          for p in sorted(folder.rglob("*")) if p.is_file() and p != record_path}
    write_json(record_path, record)
    stream = sys.stdout if code == 0 else sys.stderr
    print(f"estado: {record['estado']}" + (f" ({record['motivo_estado']})" if record["motivo_estado"] else ""),
          file=stream)
    print(f"pasta: {folder}", file=stream)
    report = folder / f"relatorio__{run_id}.html"
    if report.exists():
        print(f"relatório: {report}", file=stream)
    c = record["contagens"]
    flagged = c.get("outliers_teste")
    print(f"treino efetivo: {c['treino_efetivo']} | teste analisado: {c['teste_analisado']}"
          + ("" if flagged is None else f" | sinalizadas: {flagged} ({fmt_rate(flagged, c['teste_analisado'])})"),
          file=stream)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detector_001_isolation_forest_sklearn.py",
        description="Isolation Forest (scikit-learn) sobre o features.csv exportado por scripts/export_dataset.py.")
    parser.add_argument("--entrada", required=True,
                        help="pasta extraída do pacote (com features.csv, manifest.json e schema.json) ou um features.csv")
    parser.add_argument("--saida", default=str(DEFAULT_OUTPUT),
                        help="pasta onde criar a pasta da execução (padrão: importacao_arquivos_csv/resultados)")
    parser.add_argument("--corte-treino", help="instante ISO 8601 com fuso explícito; treino = fim da janela ≤ corte")
    parser.add_argument("--fracao-treino", type=float,
                        help=f"sem --corte-treino: corte no quantil dos fins das janelas valid (padrão {DEFAULT_FRACTION})")
    parser.add_argument("--inicio-treino",
                        help="opcional: instante ISO 8601 com fuso; janelas que começam antes ficam fora da referência")
    parser.add_argument("--percentil-limiar", type=float, default=99.0,
                        help="percentil dos scores do treino usado como limiar, 0 < p < 100 (padrão 99)")
    parser.add_argument("--seed", type=int, default=42, help="random_state do Isolation Forest (padrão 42)")
    parser.add_argument("--n-estimators", type=int, default=300, help="número de árvores (padrão 300)")
    return parser


def parse_settings(argv: list[str]) -> Settings:
    args = build_parser().parse_args(argv)
    if args.corte_treino is not None and args.fracao_treino is not None:
        raise InputError("use --corte-treino ou --fracao-treino, não os dois: a fração só vale sem corte explícito")
    fraction = DEFAULT_FRACTION if args.fracao_treino is None else args.fracao_treino
    if not (math.isfinite(fraction) and 0 < fraction < 1):
        raise InputError("--fracao-treino deve satisfazer 0 < valor < 1")
    if not (math.isfinite(args.percentil_limiar) and 0 < args.percentil_limiar < 100):
        raise InputError("--percentil-limiar deve satisfazer 0 < valor < 100")
    if args.n_estimators < 1:
        raise InputError("--n-estimators deve ser ao menos 1")
    if not 0 <= args.seed <= 2**32 - 1:
        raise InputError("--seed deve ser um inteiro entre 0 e 4294967295")
    cut = parse_utc_instant(args.corte_treino, "--corte-treino") if args.corte_treino is not None else None
    start = parse_utc_instant(args.inicio_treino, "--inicio-treino") if args.inicio_treino is not None else None
    if cut is not None and start is not None and start >= cut:
        raise InputError("--inicio-treino deve ser anterior a --corte-treino")
    if Path(args.saida).exists() and not Path(args.saida).is_dir():
        raise InputError(f"--saida deve ser uma pasta: {args.saida} é um arquivo")
    return Settings(entrada=Path(args.entrada), saida=Path(args.saida), cut=cut, cut_text=args.corte_treino,
                    fraction=None if cut is not None else fraction, training_start=start,
                    percentile=float(args.percentil_limiar), seed=args.seed, n_estimators=args.n_estimators,
                    argv=list(argv))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        settings = parse_settings(argv)
        ds = load_dataset(settings.entrada)
        plan = plan_analysis(ds, settings)
    except SystemExit as exc:          # argparse: --help (0) ou uso inválido (2)
        return exc.code if isinstance(exc.code, int) else 2
    except InputError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except InsufficientDataError as exc:
        print(f"dados insuficientes: {exc}", file=sys.stderr)
        return 3
    return execute(ds, plan, settings)


if __name__ == "__main__":
    sys.exit(main())
