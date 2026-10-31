"""
Pacote `raw`: CSVs brutos, schema, README, manifesto e ZIP.

Fluxo: snapshot da origem → SQLite temporário (recorte + fecho referencial) →
FK check → CSVs ordenados (cada linha com `source_id` e `record_role`) →
schema/README/modelo de eventos → manifesto com hashes e contagens → ZIP escrito
em `.partial` e renomeado só depois de completo. Temporários sempre removidos.
"""
from __future__ import annotations

import csv
import io
import json
import hashlib
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlmodel import Session

from ai.sequence_builder import FEATURE_NAMES
from ai.anomaly_definitions import FEATURE_DEFINITIONS

from dataset_export.codec import csv_writer, encode, encode_value, kind, read_csv
from dataset_export.features import (COVERAGE_COLUMNS, EXTRACTION_POLICY, FEATURE_COLUMNS,
                                     FEATURE_SCHEMA_VERSION, FEATURE_STATUSES,
                                     CoverageAccumulator, extractor_revision, iter_feature_rows)
from dataset_export.interpretation import InterpretationParams
from dataset_export.provenance import (code_revision, dependency_versions, sha256_bytes,
                                       sha256_file, source_hashes)
from dataset_export.request import iso_z
from dataset_export.slice import (BATCH, ExportError, create_target, finalize_sqlite,
                                  materialize_slice, table)
from dataset_export.snapshot import describe_source, snapshot_session

FORMAT = "spec064-dataset-v1"
MANIFEST_FILE = "manifest.json"
# Tetos de leitura de pacotes recebidos (o zipfile não lê além do tamanho declarado).
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_UNCOMPRESSED_BYTES = 8 * 1024 ** 3
RAW_FILES = {
    "telemetry.csv": "telemetry",
    "neighbors.csv": "neighbor",
    "nodes.csv": "node",
    "networks.csv": "network",
    "simulation_runs.csv": "simulation_run",
}
ORDER = {
    "telemetry": ("node_id", "timestamp", "id"),
    "neighbor": ("node_id", "timestamp", "id"),
    "node": ("node_id",),
    "network": ("id",),
    "simulation_run": ("node_id", "started_at", "id"),
}
EXTRA_COLUMNS = ("source_id", "record_role")
EVENTS_HEADER = ("event_id", "campaign_id", "node_id", "event_type", "origin",
                 "start_at_utc", "end_at_utc", "observer", "evidence", "notes")

# Dicionário das colunas derivadas (sem coluna de banco correspondente).
EVENTS_COLUMNS = {
    "event_id": ("string", False, "Identificador único do evento anotado (livre, ex.: EV-001)."),
    "campaign_id": ("string", True, "Campanha a que o evento pertence; igual ao campaign_id do manifesto quando houver."),
    "node_id": ("integer", True, "Nó afetado; vazio quando o evento atinge a rede inteira."),
    "event_type": ("string", False, "Tipo do evento (ex.: desligamento, troca de antena, obstrução, simulação, manutenção)."),
    "origin": ("string", False, "Quem/que causou: operador, ambiente, firmware, simulador, desconhecido."),
    "start_at_utc": ("datetime_utc", False, "Início do evento em UTC (ISO 8601 com Z)."),
    "end_at_utc": ("datetime_utc", True, "Fim em UTC; vazio se ainda aberto ou instantâneo."),
    "observer": ("string", False, "Pessoa ou sistema que observou/registrou o evento."),
    "evidence": ("string", True, "Evidência independente (foto, log, mensagem, protocolo); sem evidência o evento não é rótulo verificado."),
    "notes": ("string", True, "Observações livres."),
}

COVERAGE_DESCRIPTIONS = {
    "source_id": "Identificador da instalação de origem (DATASET_SOURCE_ID).",
    "node_id": "Nó solicitado.",
    "node_name": "Nome do nó no snapshot do cadastro.",
    "requested_start_utc": "Início (exclusivo) do período solicitado.",
    "requested_end_utc": "Fim (inclusivo) do período solicitado.",
    "first_instant_utc": "Primeiro instante da grade (fim da primeira janela); vazio se a grade for vazia.",
    "last_instant_utc": "Último instante da grade; vazio se a grade for vazia.",
    "expected_windows": "Instantes da grade no período (igual para todos os nós).",
    "written_windows": "Linhas escritas em features.csv para o nó; deve ser igual a expected_windows.",
    "status_valid": "Janelas com 11 valores finitos (podem conter defaults; ver unavailable_features).",
    "status_insufficient_data": "Janelas com menos de 2 registros de vizinhos ou com alguma feature sem valor "
                                "calculável (invalid_feature:missing:<feature>).",
    "status_invalid_numeric": "Janelas com valor não finito em alguma feature.",
    "status_error": "Janelas em que o extrator lançou exceção (motivo em invalid_reason).",
    "origin_real": "Janelas cujos registros são todos reais e a partir de data_origin_tracked_since.",
    "origin_synthetic": "Janelas cujos registros são todos sintéticos (simulador).",
    "origin_mixed": "Janelas com registros de mais de uma origem.",
    "origin_unknown": "Janelas só com registros marcados como reais antes de "
                      "data_origin_tracked_since (ou sem esse valor): limitação de proveniência, não "
                      "defeito do dado.",
    "origin_no_data": "Janelas sem registro algum, ou em que o extrator falhou antes de registrar a "
                      "proveniência.",
    "reference_eligible": "Janelas válidas e aprovadas pela política de referência (sem simulação/linhas "
                          "sintéticas).",
    "reference_excluded": "Janelas não elegíveis (inválidas ou excluídas pela política).",
    "windows_with_defaults": "Janelas com pelo menos uma feature em defaulted_features (não observada); "
                             "desde as features v4 nenhuma delas é válida.",
    "simulation_overlap_windows": "Janelas que intersectam, em qualquer nó fonte, alguma execução de "
                                  "simulation_runs.csv ou o intervalo legado sim_started_at/sim_ended_at do "
                                  "cadastro, ambos ± margem contextual; o legado é sempre avaliado.",
}

# Textos das colunas das tabelas brutas, conferidos contra o código de ingestão.
# RAW_COLUMN_TEXT (arquivo, coluna) prevalece sobre a descrição do modelo; RAW_GENERIC só
# preenche colunas genéricas sem descrição no modelo.
RAW_GENERIC = {
    "id": "Identificador original na instalação de origem.",
    "data_origin": "real (campo) ou synthetic (simulador). real não prova saúde.",
}

_DETECTOR_EMPTY = (
    "sem detecção vinculada ou quando o alerta vinculado é anterior ao registro estruturado de "
    "proveniência, introduzido depois do início da coleta; detected_at pode "
    "estar preenchido.")

RAW_COLUMN_TEXT = {
    # telemetry.csv
    ("telemetry.csv", "node_id"): "Nó que enviou a telemetria.",
    ("telemetry.csv", "timestamp"): "Instante de recepção da telemetria no backend (UTC), não de transmissão no nó.",
    ("telemetry.csv", "rssi"): "RSSI em dBm: o r do vizinho marcado como parent, quando o relatório traz uma "
                               "lista de vizinhos com parent; senão, o gw_rssi da mensagem (RSSI do último pacote "
                               "de roteamento que o gateway ouviu diretamente do nó, 0 se não o ouve "
                               "diretamente). Não é medido sobre o pacote de rádio do relatório; com parent, é o "
                               "sinal com que o nó ouve o parent, não necessariamente o do salto até o gateway; "
                               "0 indica ausência.",
    ("telemetry.csv", "snr"): "SNR em dB, pela mesma regra do rssi (s do parent ou gw_snr); 0 indica ausência, "
                              "mas 0,0 não se distingue de um SNR real de 0 dB.",
    ("telemetry.csv", "packet_id"): "Campo packet_id do formato antigo de mensagem, copiado como recebido; vazio "
                                    "quando ausente (os beacons atuais não o enviam; a ordem do pacote está em sequence).",
    ("telemetry.csv", "parent_id"): "node_id do vizinho marcado como parent no relatório: o primeiro vizinho "
                                    "direto da tabela de rotas do nó, não necessariamente o próximo salto até o "
                                    "gateway. Em relatório sem lista de vizinhos, cópia do campo parent_id da "
                                    "mensagem, que pode não ser um node_id. Vazio sem parent marcado ou com "
                                    "endereço não cadastrado; não comprova o caminho da mensagem.",
    ("telemetry.csv", "uptime"): "Tempo desde a inicialização do nó, em minutos.",
    ("telemetry.csv", "sequence"): "Número de sequência do pacote informado pelo nó.",
    ("telemetry.csv", "battery_level"): "Indicador de bateria recebido: 0–100 = %, −1 = alimentação USB, "
                                        "−2 = sensor inválido; vazio se não informado.",
    ("telemetry.csv", "temperature"): "Temperatura do campo temp do payload do nó, como recebida (sem conversão de "
                                      "unidade); vazio se o nó não a enviou.",
    ("telemetry.csv", "humidity"): "Umidade do campo hum do payload do nó, como recebida (sem conversão de unidade); "
                                   "vazio se o nó não a enviou.",
    ("telemetry.csv", "payload_raw"): "JSON bruto da mensagem recebida via MQTT. Os campos gw_rssi/gw_snr, "
                                      "acrescentados pelo gateway, são o sinal do último pacote de roteamento que "
                                      "o gateway ouviu diretamente do nó, não o do pacote de rádio desta "
                                      "mensagem: 0/0 se o gateway não ouve o nó diretamente e nos relatórios do "
                                      "próprio gateway; −85/0 (valor padrão) se o nó não está na tabela de "
                                      "rotas do gateway.",
    # neighbors.csv
    ("neighbors.csv", "link_quality"): "Índice heurístico 0–10 calculado pelo firmware a partir do RSSI e do SNR do "
                                       "vizinho; não é taxa de entrega medida.",
    ("neighbors.csv", "hops"):
        "Contagem de saltos (hop count) reportada para o vizinho; vazio se o nó não a informou.",
    ("neighbors.csv", "distance_m"):
        "Distância estimada em metros: a do beacon, quando presente, ou calculada no backend, na "
        "ingestão, a partir do RSSI por modelo log-distância com os parâmetros vigentes na época (perfil "
        "de ambiente da rede quando cadastrado); estimativa, não medição.",
    ("neighbors.csv", "timestamp"): "Instante de recepção no backend (UTC) do relatório que trouxe este vizinho; "
                                    "vizinhos do mesmo relatório têm instantes praticamente iguais. Não é quando o "
                                    "nó ouviu o vizinho.",
    ("neighbors.csv", "is_parent"): "Flag reportada pelo nó para o vizinho que ele marca como parent: o primeiro "
                                    "vizinho direto da sua tabela de rotas, não necessariamente o próximo salto até "
                                    "o gateway; não comprova a rota usada em cada mensagem.",
    # nodes.csv
    ("nodes.csv", "node_id"):
        "Identificador numérico do nó (ex.: 2 para \"N2\"), usado como chave em todos os arquivos; o "
        "endereço LoRa está em lora_address; em neighbors.csv o vizinho é identificado por "
        "neighbor_address (= lora_address).",
    ("nodes.csv", "status"): "Estado no snapshot: online ou offline; unknown aparece brevemente para nó descoberto "
                             "como vizinho que ainda não enviou telemetria.",
    ("nodes.csv", "last_seen"): "Último contato registrado no snapshot do cadastro (não é observação do período).",
    ("nodes.csv", "battery_level"): "Último valor do indicador de bateria recebido (cache do cadastro no snapshot): "
                                    "0–100 = %, −1 = alimentação USB, −2 = sensor inválido. Não é série temporal; "
                                    "o histórico está em telemetry.csv.",
    ("nodes.csv", "led_state"):
        "Último estado do LED (on/off) enviado ao broker MQTT (comando unicast ou broadcast), sem "
        "confirmação do nó; cache otimista, default off.",
    ("nodes.csv", "is_gateway"): "1 se o nó é gateway da rede, 0 caso contrário.",
    ("nodes.csv", "sim_mode"):
        "1 se o nó estava em modo de simulação de anomalia no momento do snapshot, 0 caso contrário.",
    ("nodes.csv", "sim_anomaly"):
        "Código da anomalia (A1–A7) da simulação registrada no cadastro; o encerramento de uma execução "
        "de simulation_runs.csv o limpa, mas encerramentos sem execução vinculada (legado ou ativação só "
        "por beacon) o mantêm — pode estar preenchido com sim_mode = 0. Histórico em simulation_runs.csv.",
    ("nodes.csv", "sim_started_at"): "Início (UTC) da simulação mais recente registrada no cadastro do nó.",
    ("nodes.csv", "sim_ended_at"):
        "Fim (UTC) da simulação mais recente registrada no cadastro do nó; vazio se não encerrada.",
    ("nodes.csv", "current_sim_run_id"): "id (simulation_runs.csv) da execução do simulador ativa no nó no momento do "
                                         "snapshot; vazio fora de simulação.",
    # networks.csv
    ("networks.csv", "active"): "1 se a rede está ativa para operação, 0 caso contrário.",
    ("networks.csv", "created_at"): "Criação do cadastro da rede.",
    ("networks.csv", "updated_at"): "Última atualização do cadastro da rede.",
    # simulation_runs.csv
    ("simulation_runs.csv", "node_id"): "Nó em que a simulação foi executada.",
    ("simulation_runs.csv", "anomaly"): "Código da anomalia simulada (A1–A7).",
    ("simulation_runs.csv", "started_at"): "Início da execução do simulador (instante do comando), em UTC; a primeira "
                                           "recepção sintética pode ser posterior.",
    ("simulation_runs.csv", "ended_at"): "Fim (UTC) da execução; vazio se ainda aberta no snapshot.",
    ("simulation_runs.csv", "duration_planned"): "Duração planejada em segundos.",
    ("simulation_runs.csv", "ended_reason"):
        "Motivo do encerramento: timeout (duração planejada esgotada, verificada periodicamente pelo "
        "backend), operator_stop (parada pelo operador), firmware_stopped (beacon sem sim após o período "
        "de carência, com telemetria sintética já recebida), firmware_rejected (nenhuma telemetria "
        "sintética até o fim da carência, ou falha ao publicar o comando); os limiares de carência "
        "mudaram durante a coleta; vazio se aberta no snapshot.",
    ("simulation_runs.csv", "telemetry_count"): "Telemetrias sintéticas (beacon com o campo sim) recebidas do nó "
                                                "enquanto a execução estava aberta; telemetrias reais no intervalo não entram.",
    ("simulation_runs.csv", "anomaly_onset_at"):
        "Instante de recepção no backend (UTC) da primeira telemetria sintética da execução; aproxima o "
        "início da emissão da anomalia pelo firmware. Vazio se nenhuma telemetria sintética chegou ou se "
        "a execução é anterior ao registro deste campo (sem preenchimento retroativo).",
    ("simulation_runs.csv", "detected_at"): "Instante (relógio do backend, UTC) em que foi gravado o primeiro alerta do nó "
                                            "vinculado à execução — qualquer tipo de anomalia, não necessariamente a "
                                            "simulada; o vínculo aceita alertas até o contexto do detector (janela do IF ou sequência do LSTM, o que for maior) após ended_at. Vazio sem alerta "
                                            "vinculado.",
    ("simulation_runs.csv", "detector_primary"):
        "Algoritmo do primeiro alerta vinculado à execução; vazio " + _DETECTOR_EMPTY,
    ("simulation_runs.csv", "detector_secondary"):
        "Segundo algoritmo do alerta (cascade e fusion); vazio em single e shadow, " + _DETECTOR_EMPTY,
    ("simulation_runs.csv", "detector_combo"):
        "Modo de combinação do primeiro alerta vinculado (single, cascade, shadow ou fusion); vazio "
        + _DETECTOR_EMPTY,
}

FEATURE_META_DESCRIPTIONS = {
    "source_id": "Identificador da instalação de origem (DATASET_SOURCE_ID).",
    "node_id": "Nó solicitado a que a janela se refere.",
    "node_name": "Nome do nó no snapshot do cadastro.",
    "network_id": "Rede do nó.",
    "window_start_utc": "Início (exclusivo) da janela.",
    "window_end_utc": "Fim (inclusivo) da janela; instante da grade.",
    "window_seconds": "Duração da janela em segundos.",
    "step_seconds": "Passo da grade em segundos.",
    "grid_origin_utc": "Origem da grade (instante de referência dos passos).",
    "invalid_reason": "Motivo quando feature_status ≠ valid (ex.: neighbor_records_lt_2, invalid_feature:…).",
    "neighbor_record_count": "Registros de vizinhos do nó dentro da janela.",
    "distinct_neighbor_count": "Vizinhos distintos relatados na janela.",
    "telemetry_record_count": "Registros de telemetria do nó dentro da janela.",
    "reverse_record_count": "Registros de enlaces reversos (vizinhos observando o nó) na janela.",
    "first_observation_utc": "Primeiro registro do próprio nó na janela (vizinhos relatados e telemetria); "
                             "enlaces reversos não contam.",
    "last_observation_utc": "Último registro do próprio nó na janela (vizinhos relatados e telemetria); "
                            "enlaces reversos não contam.",
    "real_record_count": "Registros classificados como reais (a partir de data_origin_tracked_since).",
    "synthetic_record_count": "Registros marcados como sintéticos.",
    "unknown_origin_record_count": "Registros marcados como reais antes de data_origin_tracked_since (ou sem esse valor): "
                                   "limitação de proveniência, não defeito do dado.",
    "source_node_ids": "JSON: nós cujos dados alimentaram a janela (o próprio e os de enlaces reversos).",
    "simulation_overlap": "1 se alguma execução de simulation_runs.csv ou o intervalo legado "
                          "sim_started_at/sim_ended_at do cadastro, ambos ± margem contextual, intersecta "
                          "a janela em um dos nós fonte; o legado é sempre avaliado.",
    "simulation_run_ids": "JSON: IDs de simulation_runs.csv ou o intervalo legado do cadastro (marcadores "
                          "legacy_node:<node_id>, do par sim_started_at/sim_ended_at), ambos ± margem "
                          "contextual, que intersectam a janela em qualquer nó fonte; o legado é sempre "
                          "avaliado e pode repetir a última execução do nó já listada por ID.",
    "reference_exclusion_reason": "Motivo da inelegibilidade: features_not_valid (feature_status diferente de "
                                  "valid), synthetic_rows (há registros sintéticos na janela) ou "
                                  "simulation_overlap (a janela intersecta uma simulação); vazio se elegível.",
    "feature_schema_version": "Identidade (versão) da definição das 11 features usada na extração.",
    "extractor_revision": "Hash (sha256, 16 hex) dos fontes feature_engineering.py, reference_policy.py e "
                          "sequence_builder.py; não cobre dataset_export/features.py.",
    "extraction_policy": "Política de extração aplicada (all_origins_v1: todas as origens entram no "
                         "cálculo).",
}
LIMITATIONS = [
    "Timestamps representam a recepção no backend, não a transmissão no nó.",
    "Cadastros (nós, redes) refletem o snapshot da exportação, não comprovam o estado de todo o período.",
    "data_origin='real' indica origem registrada, não saúde física; antes de data_origin_tracked_since a origem é desconhecida.",
    "O coletor remove vizinhos com mais de 90 dias; o pacote não recupera histórico já removido.",
    "O pacote não contém modelos, decisões online completas nem rótulos independentes.",
]

UNITS = {
    ("telemetry", "rssi"): "dBm", ("telemetry", "snr"): "dB",
    ("telemetry", "battery_level"): "percentual (0–100); −1 = USB; −2 = sensor inválido",
    ("neighbor", "rssi"): "dBm", ("neighbor", "snr"): "dB",
    ("neighbor", "distance_m"): "m (estimativa)",
    ("neighbor", "link_quality"): "indicador 0–10",
    ("node", "battery_level"): "percentual (0–100); −1 = USB; −2 = sensor inválido",
    ("node", "battery_capacity_mah"): "mAh",
}


FEATURE_FILES = ("features.csv", "coverage.csv")
# Mesmas unidades e descrições usadas nas regras e no prompt.
FEATURE_UNITS = {key: (feature.unit, feature.description)
                 for key, feature in FEATURE_DEFINITIONS.items()}


def write_feature_files(target_engine, request, interpretation, requested: list, files_dir: Path) -> dict:
    """Grava features.csv e coverage.csv sobre o recorte materializado."""
    from models import Node
    with Session(target_engine) as session:
        node_rows = {n: {"name": session.get(Node, n).name,
                         "network_id": session.get(Node, n).network_id} for n in requested}
        coverage = CoverageAccumulator(request, node_rows)
        status_counts = {s: 0 for s in FEATURE_STATUSES}
        written = 0
        with open(files_dir / "features.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv_writer(fh)
            writer.writerow(FEATURE_COLUMNS)
            for row in iter_feature_rows(session, request, interpretation, node_rows):
                writer.writerow([encode_value(row[c]) for c in FEATURE_COLUMNS])
                coverage.add(row)
                status_counts[row["feature_status"]] += 1
                written += 1
    with open(files_dir / "coverage.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv_writer(fh)
        writer.writerow(COVERAGE_COLUMNS)
        for row in coverage.rows():
            writer.writerow([encode_value(row[c]) for c in COVERAGE_COLUMNS])
    expected = coverage.expected * len(requested)
    return {"rows": written, "expected_rows": expected, "complete": written == expected,
            "status_counts": status_counts, "coverage_rows": len(requested)}



class PackageError(Exception):
    """Pacote ausente, incompleto, corrompido ou inconsistente com o próprio manifesto.

    Os hashes ficam no manifesto do próprio ZIP: detectam corrupção, não adulteração
    intencional.
    """


def _model_classes() -> dict:
    from models import Neighbor, Network, Node, SimulationRun, Telemetry
    return {"network": Network, "node": Node, "simulation_run": SimulationRun,
            "telemetry": Telemetry, "neighbor": Neighbor}


def record_role(table_name: str, row: dict, request, requested: set) -> str:
    """`requested` para o escopo pedido; `context` para contexto e fecho referencial."""
    def in_period(ts):
        return request.start < ts <= request.end
    if table_name in ("telemetry", "neighbor"):
        return "requested" if row["node_id"] in requested and in_period(row["timestamp"]) else "context"
    if table_name == "node":
        return "requested" if row["node_id"] in requested else "context"
    if table_name == "network":
        return "requested" if row["id"] == request.network_id else "context"
    ended = row["ended_at"]
    overlaps = row["started_at"] < request.end and (ended is None or ended > request.start)
    return "requested" if row["node_id"] in requested and overlaps else "context"


def write_table_csv(engine, table_name: str, path: Path, source_id: str, role) -> int:
    source = table(table_name)
    columns = list(source.columns)
    order = [source.c[name] for name in ORDER[table_name]]
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as fh, engine.connect() as connection:
        writer = csv_writer(fh)
        writer.writerow([c.name for c in columns] + list(EXTRA_COLUMNS))
        result = connection.execute(select(source).order_by(*order).execution_options(yield_per=BATCH))
        for row in result.mappings():
            writer.writerow([encode(row[c.name], c) for c in columns]
                            + [source_id, role(table_name, row)])
            written += 1
    return written


def build_schema() -> dict:
    classes = _model_classes()
    files = {}
    for file_name, table_name in RAW_FILES.items():
        fields = classes[table_name].model_fields
        columns = [{
            "name": c.name,
            "type": kind(c) if kind(c) != "datetime" else "datetime_utc",
            "nullable": bool(c.nullable),
            "unit": UNITS.get((table_name, c.name), "as_recorded" if kind(c) in ("float", "integer") else None),
            "description": (fields[c.name].description or "") if c.name in fields else "",
        } for c in table(table_name).columns]
        columns += [
            {"name": "source_id", "type": "string", "nullable": False,
             "description": "Identificador da instalação de origem (DATASET_SOURCE_ID)."},
            {"name": "record_role", "type": "string", "nullable": False,
             "description": "requested = escopo pedido (nós e rede pedidos; registros desses nós em "
                            "(início, fim], inclusive os enlaces reversos que eles relatam; simulações "
                            "desses nós que intersectam (início, fim]); context = demais registros, "
                            "por exemplo registro anterior ao início, enlace reverso relatado por nó não "
                            "pedido ou registro incluído por referência."},
        ]
        files[file_name] = {"table": table_name, "columns": columns}
    for file_name in RAW_FILES:
        for column in files[file_name]["columns"]:
            specific = RAW_COLUMN_TEXT.get((file_name, column["name"]))
            if specific:
                column["description"] = specific
            elif not column["description"]:
                column["description"] = RAW_GENERIC.get(column["name"], "")
    feature_columns = []
    for name in FEATURE_COLUMNS:
        column = {"name": name, "nullable": True, "type": "string",
                  "description": FEATURE_META_DESCRIPTIONS.get(name, "")}
        if name in FEATURE_UNITS:
            column.update(type="float", unit=FEATURE_UNITS[name][0], description=FEATURE_UNITS[name][1])
        elif name == "unavailable_features":
            column.update(type="json_object",
                          description="JSON feature → motivo de indisponibilidade (no_eligible_link, "
                                      "no_link_quality, insufficient_battery_readings, "
                                      "insufficient_distinct_counters, insufficient_telemetry, "
                                      "insufficient_parent_records, no_reverse_link); "
                                      "preenchido também em linhas inválidas.")
        elif name.endswith("_utc"):
            column.update(type="datetime_utc")
        elif name.endswith("_count") or name in ("node_id", "network_id", "window_seconds", "step_seconds"):
            column.update(type="integer")
        elif name in ("simulation_overlap", "reference_eligible_by_policy"):
            column.update(type="boolean")
        elif name in ("defaulted_features", "source_node_ids", "simulation_run_ids"):
            column.update(type="json_array")
        feature_columns.append(column)
    by_name = {c["name"]: c for c in feature_columns}
    by_name["feature_status"].update(
        enum=list(FEATURE_STATUSES), nullable=False,
        description="valid = 11 features observadas e finitas; feature não observada (unavailable_features) "
                    "deixa a janela insufficient_data, como no detector; demais estados têm as 11 entradas nulas.")
    by_name["origin_status"].update(enum=["real", "synthetic", "mixed", "unknown", "no_data"], nullable=False,
                                    description="Origem dos registros da janela; unknown = antes de data_origin_tracked_since (ou sem esse valor): "
                                                "limitação de proveniência, não defeito do dado.")
    by_name["reference_eligible_by_policy"]["description"] = (
        "Elegível pelas evidências disponíveis à política (features válidas e sem simulação/linhas sintéticas). Não é rótulo de saúde.")
    by_name["defaulted_features"]["description"] = "Features não observadas na janela, que o extrator preenche só para apresentação; a janela não é válida."
    by_name["available_at_utc"]["description"] = "Fim da janela: instante em que a feature fica disponível. Não é o horário de decisão do serviço."
    files["features.csv"] = {"table": None, "columns": feature_columns,
                             "model_input_columns": list(FEATURE_NAMES)}
    files["coverage.csv"] = {"table": None, "columns": [
        {"name": c, "type": ("datetime_utc" if c.endswith("_utc") else
                             "string" if c in ("source_id", "node_name") else "integer"),
         "nullable": c in ("first_instant_utc", "last_instant_utc", "node_name"),
         "description": COVERAGE_DESCRIPTIONS[c]}
        for c in COVERAGE_COLUMNS]}
    files["events_template.csv"] = {"table": None, "columns": [
        {"name": c, "type": t, "nullable": n, "description": d}
        for c, (t, n, d) in EVENTS_COLUMNS.items()]}

    return {
        "format": FORMAT,
        "conventions": {
            "encoding": "UTF-8, RFC 4180, separador vírgula, decimal ponto",
            "null": "célula vazia sem aspas; todo valor não nulo vem entre aspas (QUOTE_NOTNULL)",
            "boolean": "0/1",
            "datetime_utc": "ISO 8601 em UTC com sufixo Z; microssegundos quando armazenados",
            "float": "representação de ida e volta exata (repr do Python); sem NaN/Inf",
            "ordering": "telemetria/vizinhos por (node_id, timestamp, id); nós por node_id; "
                        "redes por id; simulações por (node_id, started_at, id)",
        },
        "files": files,
    }


_README_HEAD = """# Pacote de dados — {dataset_id}

Exportação (formato `{format}`, opção **{kind_label}**) da fonte `{source_id}`,
rede {network_id}, período UTC ({start}, {end}]. O tipo, a nulabilidade, a unidade e o
significado de cada coluna estão em `schema.json`.

## Arquivos
- `schema.json`: tipos, nulabilidade, unidade e significado de cada coluna.
- `manifest.json`: filtros, nós, contagens, hashes SHA-256, revisão do código e parâmetros
  de interpretação.
- `events_template.csv`: modelo vazio para anotar intervenções/eventos de forma independente.
"""

_README_RAW_FILES = """- `telemetry.csv`, `neighbors.csv`, `nodes.csv`, `networks.csv`, `simulation_runs.csv`: registros
  originais, sem arredondamento, deduplicação ou correção; mesmos IDs da origem. Registros
  anteriores ao início (contexto da primeira janela), enlaces reversos relatados por nós não
  pedidos e registros do fecho referencial vêm marcados com `record_role=context`; os registros
  dos nós pedidos dentro do período, inclusive os enlaces reversos que eles relatam, são
  `requested`.
"""

_README_RAW_SECTIONS = """
## Leitura em Python (bruto)
```python
import pandas as pd
nb = pd.read_csv("neighbors.csv", parse_dates=["timestamp"], date_format="ISO8601")
nb = nb[nb.record_role == "requested"]          # só o período/nós pedidos
```

Leitores comuns podem converter texto vazio, "NA" e nulo no mesmo valor (por exemplo, o
pandas com opções padrão). Para preservar essas diferenças, use `dataset_export.codec.read_csv`
ou o importador do projeto (`scripts/import_dataset.py`). Sem `date_format="ISO8601"`, o pandas deixa como texto uma coluna
de datas em que algum instante vem sem fração de segundo.

## Reconstruir um banco SQLite (mesmo esquema do projeto)
```bash
python scripts/import_dataset.py --package {zip_name} --out dados.db
```
"""

_README_FEATURE_FILES = """- `features.csv`: uma linha por nó solicitado × instante da grade, inclusive sem dados.
- `coverage.csv`: por nó, janelas esperadas/escritas, estados, origem, elegibilidade, defaults e sobreposição com simulação.
"""

_README_FEATURE_SECTIONS = """
## Features por janela
Use como entrada de modelos apenas as 11 colunas canônicas (`rssi_mean` … `link_asymmetry`)
das linhas com `feature_status == "valid"`, que têm as 11 features observadas e finitas.
Desde as features v4, a janela com feature não observada (`unavailable_features`; por
exemplo, sem indicador de qualidade, com menos de dois contadores ou sem enlace reverso)
sai como `insufficient_data`, como no detector do backend; `defaulted_features` lista
as que o extrator preencheu só para apresentação, e esse valor nunca aparece como medida. A bateria parte da última leitura até 60 min antes da janela,
porque o firmware só a envia quando muda 2 p.p. ou a cada 5 min.
`node_id`, horários, flags de simulação, `origin_status` e `reference_eligible_by_policy`
são metadados: usá-los como entrada pode vazar informação. Uma observação real não é
automaticamente saudável.

```python
import json
import pandas as pd
f = pd.read_csv("features.csv", parse_dates=["window_end_utc"], date_format="ISO8601")
X_cols = ["rssi_mean","rssi_std","snr_mean","snr_std","pdr_mean","battery_delta",
          "packet_loss_rate","rssi_snr_divergence","beacon_interval_std",
          "route_change_rate","link_asymmetry"]
valid = f[f.feature_status == "valid"].sort_values("window_end_utc")
measured = valid[valid.unavailable_features.map(lambda s: json.loads(s) == {{}})]  # só medições
corte = measured.window_end_utc.quantile(0.7)                    # separar no tempo ANTES de ajustar
folga = pd.Timedelta(seconds=int(f.window_seconds.iloc[0]))      # janelas vizinhas compartilham dados
treino = measured[(measured.window_end_utc <= corte) & (measured.reference_eligible_by_policy == 1)]
teste = measured[measured.window_end_utc > corte + folga]
```

Ajuste scaler e modelo só no treino. Não use `fit_predict` no arquivo inteiro como
avaliação de generalização.
"""

_README_TAIL = """
## Limitações
{limitations}
"""

KIND_LABELS = {"raw": "Bruto", "features": "Features por janela", "full": "Completo"}


def build_readme(kind: str, **fields) -> str:
    """README específico da opção exportada: só descreve o que o pacote contém."""
    raw, feats = kind in ("raw", "full"), kind in ("features", "full")
    parts = [_README_HEAD]
    if raw:
        parts.append(_README_RAW_FILES)
    if feats:
        parts.append(_README_FEATURE_FILES)
    if raw:
        parts.append(_README_RAW_SECTIONS)
    if feats:
        parts.append(_README_FEATURE_SECTIONS)
    parts.append(_README_TAIL)
    return "".join(parts).format(kind_label=KIND_LABELS[kind], **fields)


def _write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _package_root(request, dataset_suffix: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", request.source_id)
    return (f"dataset_{safe}_{request.start:%Y%m%dT%H%M%SZ}_"
            f"{request.end:%Y%m%dT%H%M%SZ}_{dataset_suffix}")


def new_dataset_suffix() -> str:
    """Sufixo aleatório (8 hex) que distingue pacotes gerados para o mesmo pedido."""
    return uuid.uuid4().hex[:8]


def dataset_id_for(request, dataset_suffix: str) -> str:
    """`dataset_id` (= raiz do ZIP) que `export_dataset(..., dataset_suffix=...)` grava."""
    if not re.fullmatch(r"[0-9a-f]{8}", dataset_suffix or ""):
        raise ValueError(f"dataset_suffix inválido: {dataset_suffix!r} (esperado 8 hex)")
    return _package_root(request, dataset_suffix)


# Prefixo do nome padronizado do ZIP: a opção fica visível sem abrir o arquivo.
# O dataset_id (pasta do ZIP e manifesto) não muda.
DOWNLOAD_PREFIX = {"raw": "bruto", "features": "feature_por_janela", "full": "completo"}


def download_filename(dataset_id: str, export_kind: str) -> str:
    """Nome padronizado do ZIP: `<opção>_<dataset_id>.zip`."""
    return f"{DOWNLOAD_PREFIX[export_kind]}_{dataset_id}.zip"


def export_dataset(engine, request, out_zip, *, settings, sqlite_file=None, now=None,
                   dataset_suffix=None) -> dict:
    """Gera o pacote (`raw`, `features` ou `full`) em `out_zip`. Retorna o manifesto.

    O README cita `out_zip.name` no exemplo de importação. Quem precisa nomear o ZIP pelo
    `dataset_id` antes de gerá-lo sorteia o sufixo com
    `new_dataset_suffix()`, obtém o nome com `dataset_id_for()` e repassa o mesmo sufixo aqui.
    """
    now = now or datetime.utcnow()
    request.validate(now=now, max_days=float(getattr(settings, "DATASET_EXPORT_MAX_DAYS", 31.0)))
    out_zip = Path(out_zip)
    partial = out_zip.with_name(out_zip.name + ".partial")
    if out_zip.exists() or partial.exists():
        raise ExportError(f"{out_zip} (ou .partial) já existe — pacotes não são sobrescritos")
    interpretation = InterpretationParams.for_export(request, settings)
    root = dataset_id_for(request, new_dataset_suffix() if dataset_suffix is None else dataset_suffix)
    timings = {}
    workdir = Path(tempfile.mkdtemp(prefix="spec064_"))
    target = None
    own_partial = False
    try:
        started = time.monotonic()
        target = create_target(workdir / "slice.db")
        with snapshot_session(engine) as session:
            summary = materialize_slice(session, request, target,
                                        interpretation.reference_margin_minutes)
        timings["read_and_materialize"] = round(time.monotonic() - started, 3)

        started = time.monotonic()
        violations = finalize_sqlite(target)
        if violations:
            raise ExportError(f"foreign_key_check: {len(violations)} violação(ões): {violations[:5]}")
        timings["finalize"] = round(time.monotonic() - started, 3)

        started = time.monotonic()
        files_dir = workdir / root
        files_dir.mkdir()
        requested = set(summary.requested_node_ids)

        def role(table_name, row):
            return record_role(table_name, row, request, requested)

        files, tables = {}, {}
        if request.export_kind in ("raw", "full"):
            for file_name, table_name in RAW_FILES.items():
                rows = write_table_csv(target, table_name, files_dir / file_name,
                                       request.source_id, role)
                tables[table_name] = {"rows": rows}
                files[file_name] = {"rows": rows}
        timings["write_csv"] = round(time.monotonic() - started, 3)
        features_info = None
        if request.export_kind in ("features", "full"):
            started = time.monotonic()
            features_info = write_feature_files(target, request, interpretation,
                                                summary.requested_node_ids, files_dir)
            files["features.csv"] = {"rows": features_info["rows"]}
            files["coverage.csv"] = {"rows": features_info["coverage_rows"]}
            timings["features"] = round(time.monotonic() - started, 3)
            if not features_info["complete"]:
                raise ExportError("features.csv incompleto: linhas escritas ≠ esperadas")
        target.dispose()
        target = None
        started = time.monotonic()

        buf = io.StringIO()
        csv_writer(buf).writerow(EVENTS_HEADER)
        _write_bytes(files_dir / "events_template.csv", buf.getvalue().encode("utf-8"))
        files["events_template.csv"] = {"rows": 0}
        _write_bytes(files_dir / "schema.json",
                     json.dumps(build_schema(), indent=2, ensure_ascii=False).encode("utf-8"))
        files["schema.json"] = {}
        readme = build_readme(
            request.export_kind,
            dataset_id=root, format=FORMAT, source_id=request.source_id,
            network_id=request.network_id, start=iso_z(request.start), end=iso_z(request.end),
            zip_name=out_zip.name, limitations="\n".join(f"- {item}" for item in LIMITATIONS))
        _write_bytes(files_dir / "README.md", readme.encode("utf-8"))
        files["README.md"] = {}
        for name, info in files.items():
            path = files_dir / name
            info.update(sha256=sha256_file(path), bytes=path.stat().st_size)

        grid_count, grid_first, grid_last = request.grid_bounds()
        manifest = {
            "format": FORMAT,
            "dataset_id": root,
            "package_root": root,
            "export_kind": request.export_kind,
            "created_at": iso_z(now.replace(microsecond=0)),
            "campaign_id": request.campaign_id,
            "source": {"source_id": request.source_id,
                       **describe_source(engine, sqlite_file=sqlite_file)},
            "request": {
                "network_id": request.network_id,
                "node_ids": list(request.node_ids) if request.node_ids is not None else None,
                "start": iso_z(request.start), "end": iso_z(request.end),
                "window_seconds": request.window_seconds, "step_seconds": request.step_seconds,
                "grid_origin": iso_z(request.grid_origin),
            },
            "resolved": {
                "interval_convention": "(start, end]",
                "data_start": iso_z(summary.data_start), "end": iso_z(summary.end),
                "grid_instants": grid_count,
                "first_instant": iso_z(grid_first) if grid_first else None,
                "last_instant": iso_z(grid_last) if grid_last else None,
            },
            "nodes": {"requested": summary.requested_node_ids,
                      "context": summary.context_node_ids},
            "networks": summary.network_ids,
            "simulation_runs": summary.simulation_run_ids,
            "tables": tables,
            "slice_rows": summary.rows,
            "features": None if features_info is None else {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "extractor_revision": extractor_revision(),
                "extraction_policy": EXTRACTION_POLICY,
                "feature_names": list(FEATURE_NAMES),
                **features_info,
            },

            "interpretation": interpretation.to_dict(),
            "files": files,
            "validation": {
                "foreign_key_violations": 0,
                "counts_match": (all(tables[t]["rows"] == summary.rows[t] for t in tables)
                                 and (features_info is None or features_info["complete"])),
            },
            "code": code_revision(),
            "source_hashes": source_hashes(),
            "dependencies": dependency_versions(),
            "labels_status": "not_provided",
            "limitations": LIMITATIONS,
            "timings_seconds": timings,
        }
        if not manifest["validation"]["counts_match"]:
            raise ExportError("contagens dos CSVs diferem do recorte materializado")
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        # Reserva exclusiva: um pedido concorrente não apaga nem substitui outro.
        with partial.open("xb"):
            pass
        own_partial = True
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(files_dir.iterdir()):
                zf.write(path, f"{root}/{path.name}")
            timings["package"] = round(time.monotonic() - started, 3)
            manifest["timing_scope"] = "package inclui metadados e compactação dos dados, antes de gravar o manifesto e validar o ZIP"
            zf.writestr(f"{root}/{MANIFEST_FILE}", json.dumps(
                manifest, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        verify_package(partial)
        # link cria o destino atomicamente e falha se ele já existir.
        os.link(partial, out_zip)
        partial.unlink()
        own_partial = False
        return manifest
    except BaseException:
        if own_partial:
            partial.unlink(missing_ok=True)
        raise
    finally:
        if target is not None:
            target.dispose()
        shutil.rmtree(workdir, ignore_errors=True)


def verify_package(zip_path) -> tuple:
    """Valida arquivos obrigatórios, hashes, tamanhos e linhas em streaming."""
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise PackageError(f"pacote ausente: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            roots = {name.split("/", 1)[0] for name in names}
            if len(roots) != 1 or len(names) != len(set(names)):
                raise PackageError("pacote deve ter uma raiz e arquivos sem duplicatas")
            root = roots.pop()
            if root in ("", ".", "..") or any(
                    name.count("/") != 1 or name.split("/")[1] in ("", ".", "..")
                    for name in names):
                raise PackageError("caminhos inválidos no pacote")
            sizes = {info.filename: info.file_size for info in zf.infolist()}
            if sizes.get(f"{root}/{MANIFEST_FILE}", 0) > MAX_MANIFEST_BYTES:
                raise PackageError("manifesto acima do limite de tamanho")
            if sum(sizes.values()) > MAX_PACKAGE_UNCOMPRESSED_BYTES:
                raise PackageError("pacote descomprimido acima do limite de tamanho")
            manifest = json.loads(zf.read(f"{root}/{MANIFEST_FILE}"))
            if not isinstance(manifest, dict):
                raise PackageError("manifesto deve ser um objeto JSON")
            if manifest.get("format") != FORMAT or manifest.get("package_root") != root:
                raise PackageError("formato ou raiz desconhecidos no manifesto")
            export_kind = manifest.get("export_kind")
            if export_kind not in ("raw", "features", "full"):
                raise PackageError("export_kind desconhecido no manifesto")
            required = {"schema.json", "README.md", "events_template.csv"}
            if export_kind in ("raw", "full"):
                required.update(RAW_FILES)
            if export_kind in ("features", "full"):
                required.update(("features.csv", "coverage.csv"))
            files = manifest.get("files", {})
            if not required.issubset(files):
                raise PackageError("manifesto omite arquivos obrigatórios")
            if any("/" in name or "\\" in name or name in ("", ".", "..") for name in files):
                raise PackageError("nome inválido no manifesto")
            expected = {f"{root}/{name}" for name in files} | {f"{root}/{MANIFEST_FILE}"}
            if set(names) != expected:
                raise PackageError("lista de arquivos difere do manifesto")
            InterpretationParams.from_dict(manifest["interpretation"])
            if manifest["source"]["source_id"] != manifest["interpretation"]["source_id"]:
                raise PackageError("source_id inconsistente no manifesto")
            for name, info in files.items():
                if sizes[f"{root}/{name}"] != info["bytes"]:
                    raise PackageError(f"{name}: tamanho declarado no ZIP difere do manifesto")
                digest, size = hashlib.sha256(), 0
                with zf.open(f"{root}/{name}") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        digest.update(chunk)
                        size += len(chunk)
                if digest.hexdigest() != info["sha256"] or size != info["bytes"]:
                    raise PackageError(f"{name}: sha256 ou tamanho não confere")
                if name.endswith(".csv"):
                    with zf.open(f"{root}/{name}") as raw:
                        reader = read_csv(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                        header = next(reader, None)
                        if not header or any(not isinstance(c, str) for c in header):
                            raise PackageError(f"{name}: cabeçalho inválido")
                        count = 0
                        for record in reader:
                            if len(record) != len(header):
                                raise PackageError(f"{name}: quantidade de colunas inválida")
                            count += 1
                    if count != info["rows"]:
                        raise PackageError(f"{name}: número de linhas não confere")
                    if name in RAW_FILES and count != manifest["tables"][RAW_FILES[name]]["rows"]:
                        raise PackageError(f"{name}: contagem da tabela não confere")
            validation = manifest.get("validation", {})
            if validation.get("foreign_key_violations") != 0 or validation.get("counts_match") is not True:
                raise PackageError("manifesto indica validação incompleta")
            return manifest, root
    except PackageError:
        raise
    except (zipfile.BadZipFile, KeyError, ValueError, TypeError, AttributeError, UnicodeError,
            csv.Error) as exc:
        raise PackageError(f"pacote inválido ({type(exc).__name__})") from exc
