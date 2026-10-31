"""Pacotes sintéticos para os testes do detector 001.

Dados inventados, gravados no formato do exportador (UTF-8, vírgula, CRLF, valor
não nulo entre aspas, nulo = célula vazia sem aspas). Não vêm da coleta e ficam só
nas pastas temporárias dos testes, nunca em `resultados/`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import detector_001_isolation_forest_sklearn as det

# O contrato das colunas vem do README do detector e do formato do exportador,
# não do código testado.
FEATURES = ["rssi_mean", "rssi_std", "snr_mean", "snr_std", "pdr_mean", "battery_delta",
            "packet_loss_rate", "rssi_snr_divergence", "beacon_interval_std",
            "route_change_rate", "link_asymmetry"]
COLUMNS = ["source_id", "node_id", "node_name", "network_id", "window_start_utc", "window_end_utc",
           "available_at_utc", "window_seconds", "step_seconds", *FEATURES, "feature_status",
           "invalid_reason", "defaulted_features", "unavailable_features", "origin_status",
           "simulation_overlap", "simulation_run_ids", "reference_eligible_by_policy",
           "reference_exclusion_reason", "feature_schema_version", "extractor_revision",
           "extraction_policy"]
W, S, N = 900, 420, 60
START = pd.Timestamp("2026-01-01T00:07:00Z")
CORTE = "2026-01-01T04:00:00Z"   # por nó: treino k <= 33 (34 janelas); folga k = 34, 35; teste k >= 36
LOC = np.array([-70.0, 1.0, 8.0, 0.5, 9.0, 0.0, 0.01, 0.2, 0.03, 0.0, 1.5])
SCALE = np.array([1.0, 0.1, 0.5, 0.05, 0.2, 0.01, 0.005, 0.05, 0.005, 0.01, 0.2])


def utc(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def position(node_pos: int, k: int, n: int = N) -> int:
    """Posição, na lista de make_rows, da k-ésima janela do nó na posição node_pos."""
    return node_pos * n + k


def center(node: int) -> dict:
    """Valores das 11 features no centro da distribuição do nó."""
    loc = LOC.copy()
    loc[0] -= node
    return {f: repr(float(v)) for f, v in zip(FEATURES, loc)}


def make_rows(nodes=(1, 2, 3), n: int = N, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for node in nodes:
        for k in range(n):
            end = START + pd.Timedelta(seconds=S * k)
            values = rng.normal(LOC, SCALE)
            values[0] -= node
            rows.append({
                "source_id": "fonte-teste", "node_id": str(node), "node_name": f"Nó {node}",
                "network_id": "1", "window_start_utc": utc(end - pd.Timedelta(seconds=W)),
                "window_end_utc": utc(end), "available_at_utc": utc(end),
                "window_seconds": str(W), "step_seconds": str(S),
                **{f: repr(float(v)) for f, v in zip(FEATURES, values)},
                "feature_status": "valid", "invalid_reason": None, "defaulted_features": "[]",
                "unavailable_features": "{}", "origin_status": "unknown", "simulation_overlap": "0",
                "simulation_run_ids": "[]", "reference_eligible_by_policy": "1",
                "reference_exclusion_reason": None, "feature_schema_version": "spec065-features-v2",
                "extractor_revision": "sha256:teste", "extraction_policy": "all_origins_v1",
            })
    return rows


def invalidate(row: dict) -> None:
    """Transforma a linha numa janela insufficient_data, como o exportador grava."""
    row.update({f: None for f in FEATURES})
    row.update(feature_status="insufficient_data", invalid_reason="neighbor_records_lt_2",
               reference_eligible_by_policy="0", reference_exclusion_reason="features_not_valid")


def _cell(value) -> str:
    return "" if value is None else '"' + str(value).replace('"', '""') + '"'


def write_csv(path: Path, rows: list[dict], columns=COLUMNS) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for values in [columns, *([row.get(c) for c in columns] for row in rows)]:
            fh.write(",".join(_cell(v) for v in values) + "\r\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_package(folder: Path, rows: list[dict], *, columns=COLUMNS, model_inputs=FEATURES,
                  dataset_id: str = "dataset_fonte-teste_1") -> Path:
    """Grava features.csv, schema.json e manifest.json (com os SHA-256) em `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / "features.csv", rows, columns)
    schema = {"format": "spec064-dataset-v1",
              "files": {"features.csv": {"table": "features",
                                         "columns": [{"name": c} for c in columns],
                                         "model_input_columns": list(model_inputs)}}}
    (folder / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    files = {name: {"rows": None, "sha256": sha256(folder / name),
                    "bytes": (folder / name).stat().st_size}
             for name in ("features.csv", "schema.json")}
    files["features.csv"]["rows"] = len(rows)
    manifest = {
        "format": "spec064-dataset-v1", "dataset_id": dataset_id, "export_kind": "features",
        "created_at": "2026-01-02T00:00:00Z", "campaign_id": None,
        "source": {"source_id": "fonte-teste"},
        "interpretation": {"source_id": "fonte-teste", "window_seconds": W, "step_seconds": S},
        "features": {"feature_schema_version": "spec065-features-v2",
                     "extractor_revision": "sha256:teste", "extraction_policy": "all_origins_v1",
                     "feature_names": list(FEATURES), "rows": len(rows)},
        "files": files,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return folder


def run_detector(base: Path, entrada: Path, *args: str, n_estimators: str = "40"):
    """Roda a CLI com saída em `base/res`; devolve (código de saída, pastas criadas)."""
    saida = Path(base) / "res"
    code = det.main(["--entrada", str(entrada), "--saida", str(saida),
                     "--n-estimators", n_estimators, *args])
    folders = sorted(p for p in saida.iterdir() if p.is_dir()) if saida.exists() else []
    return code, folders


def read_outputs(folder: Path) -> dict:
    """Lê as saídas de uma execução (None para o arquivo que não existir)."""
    run_id = folder.name

    def table(name):
        path = folder / f"{name}__{run_id}.csv"
        if not path.exists():
            return None
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])

    def text(name):
        path = folder / name.format(run_id)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def reject(constant):
        raise AssertionError(f"JSON não estrito: {constant}")

    return {
        "run_id": run_id,
        "predicoes": table("predicoes"),
        "outliers": table("outliers"),
        "resumo": table("resumo_por_dispositivo"),
        "execucao": json.loads(text("execucao__{}.json"), parse_constant=reject),
        "relatorio_md": text("relatorio__{}.md"),
        "relatorio_html": text("relatorio__{}.html"),
    }
