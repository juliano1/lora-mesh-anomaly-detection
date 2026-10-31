#!/usr/bin/env python3
"""Autoteste da detecção com dados sintéticos em CSV, sem banco, broker ou hardware.

Avalia o Isolation Forest e o LSTM-AE do pipeline online, isoladamente, e as
combinações cascata (IF>LSTM) e fusão (IF+LSTM), cada uma registrada em separado.

Duas fases, ambas por CSV:

1. Geração: séries temporais contínuas de um nó com dois enlaces, no formato do
   firmware (beacon a cada 20 s, até dois vizinhos, bateria enviada quando muda
   2 p.p. ou a cada 5 min, contador de sequência e uptime em minutos):
   ``treino.csv`` (referência normal de 24 h), ``avaliacao.csv`` (cenários,
   posteriores à referência e com outra semente) e ``instantes.csv`` (instantes de
   decisão com o cenário de origem).
2. Avaliação: lê os CSVs e executa o código do backend sem alterações: extração das
   11 features (``compute_window_features``), ``SequenceBuilder``, runners,
   ``OnlinePipeline`` (treino e calibração de τ) e regras A1–A7, com a configuração
   padrão do backend (``resolve_effective_config``).

Divisão temporal, sem sobreposição (cortes globais, com purga nas fronteiras):
  treino      60% iniciais das sequências de referência: pesos do LSTM
  validação   20% seguintes: só a parada antecipada (escolha da época) do LSTM
  calibração  20% finais: limiar p99 do LSTM e τ da fusão; o IF treina até o início
              deste bloco (não usa validação; o limiar dele vem da contaminação no treino)
  teste       cenários, depois da referência; modelos congelados após o treino

Cenários (episódios de 120 min: 60 min normais e 60 min do cenário):
  controle       sem alteração                              → normal
  rssi_snr       RSSI ≈ −106 dBm, SNR ≈ −6 dB                → anomalia A1
  perda          ≈50% dos beacons do nó não chegam           → anomalia A2
  bateria        queda de 30 p.p. nos 15 min finais          → anomalia A3
  esparsa        nó sem vizinhos (1 registro a cada 20 min)  → dados insuficientes
  nao_finito     RSSI NaN num registro                       → entrada inválida
  sem_modelo     primeira tentativa de treino sem referência → sem modelo

Instantes: ``estabelecida`` (fim do episódio) entra no critério. Nele, a janela de
15 min do IF e a sequência de 60 min do LSTM ficam inteiras no cenário; na bateria, a
queda ocupa a janela do IF e 3 das 12 subjanelas do LSTM. ``inicio`` (15 min após o
início do cenário) é informativo. No controle, os dois instantes contam.

Cada decisão recebe exatamente um estado: normal, anomalia, dados_insuficientes,
entrada_invalida, sem_modelo ou erro. O score do detector mede o afastamento em
relação à referência; não é probabilidade de falha.

Saída: esperado × obtido com APROVADO/REPROVADO por detector e cenário, falsos
positivos (inclusive na calibração) e anomalias não detectadas. Código de saída 0
se todos os detectores passam em todos os cenários, 1 se algum reprova e 2 se a
execução falha (inclusive LSTM indisponível sem ``--sem-lstm``).

Uso:
    python scripts/self_test_detection.py [--seed N] [--episodios N] [--sem-lstm]
        [--dados DIR | --entrada DIR] [--csv resultados.csv] [--json]

--dados guarda os CSVs gerados em DIR; --entrada reavalia CSVs já gravados sem gerar.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
# O backend cria um engine ao importar; o autoteste não usa banco algum.
os.environ["DATABASE_URL"] = "sqlite://"

import numpy as np  # noqa: E402

from ai.anomaly_ml import classify_anomaly_type  # noqa: E402
from ai.feature_engineering import (  # noqa: E402
    BATTERY_LOOKBACK, FeatureExtractor, _default_origin, _fill_provenance, compute_window_features,
)
from ai.online.combiner import COMBO_VALID, Combiner  # noqa: E402
from ai.online.effective_config import resolve_effective_config  # noqa: E402
from ai.online.registry import DetectionContext  # noqa: E402
from ai.reference_policy import ReferencePolicy  # noqa: E402
from ai.sequence_builder import feature_dict_to_vector  # noqa: E402

BEACON_SECONDS = 20                   # IDENTITY_BROADCAST_INTERVAL do firmware
BATTERY_HEARTBEAT_SECONDS = 300       # BATTERY_SEND_INTERVAL
BATTERY_CHANGE = 2                    # BATTERY_CHANGE_THRESHOLD (p.p.)
NODE, PARENT, OTHER = "0001", "0002", "0003"
NODE_IDS = {NODE: 1, PARENT: 2, OTHER: 3}
REFERENCE_HOURS = 24                  # = ML_HISTORY_HOURS (histórico de treino padrão)
EPISODE_MINUTES, ONSET_MINUTES = 120, 60
REFERENCE_START = datetime(2026, 1, 1)
TRAIN_FILE, EVAL_FILE, INSTANTS_FILE = "treino.csv", "avaliacao.csv", "instantes.csv"
COLUMNS = ("registro", "timestamp", "no", "endereco", "rssi", "snr", "link_quality",
           "is_parent", "battery_level", "sequence", "uptime")
INSTANT_COLUMNS = ("cenario", "episodio", "instante", "timestamp", "criterio")
STATES = ("normal", "anomalia", "dados_insuficientes", "entrada_invalida", "sem_modelo", "erro")

# Limites de aceite (fração dos instantes do critério).
MAX_FALSE_POSITIVE_RATE = 0.10        # controle: contaminação 0,03 no IF, p99 no LSTM
MIN_DETECTION_RATE = 0.90             # cenários degradados detectados
MIN_TYPE_RATE = 0.90                  # das detecções, com o tipo esperado
EXACT_STATE_RATE = 1.0                # dados insuficientes/inválidos/sem modelo


@dataclass(frozen=True)
class Profile:
    rssi: float = -75.0
    snr: float = 8.0
    link_quality: float = 8.5
    # Referência com a variação de um nó real: perda ocasional, dreno lento e ruído
    # de leitura do indicador de bateria (inteiro).
    loss: float = 0.05
    drain_per_hour: float = 0.9
    battery_noise: float = 0.5
    sparse: bool = False


BOTH_INSTANTS = (("inicio", ONSET_MINUTES + 15, False), ("estabelecida", EPISODE_MINUTES, True))
ONE_INSTANT = (("estabelecida", EPISODE_MINUTES, True),)


@dataclass(frozen=True)
class Scenario:
    key: str
    description: str
    profile: Profile
    expected_state: str
    expected_type: str | None = None
    battery_drop: float = 0.0         # p.p. nos 15 min finais do episódio
    nan_rssi: bool = False
    instants: tuple = BOTH_INSTANTS
    episodes_factor: float = 1.0


SCENARIOS = (
    Scenario("controle", "tráfego normal", Profile(), "normal",
             instants=(("inicio", ONSET_MINUTES + 15, True), ("estabelecida", EPISODE_MINUTES, True))),
    Scenario("rssi_snr", "RSSI ≈ −106 dBm e SNR ≈ −6 dB", Profile(rssi=-106, snr=-6, link_quality=2),
             "anomalia", "A1"),
    Scenario("perda", "≈50% dos beacons do nó não chegam", Profile(loss=0.5), "anomalia", "A2"),
    Scenario("bateria", "queda de 30 p.p. em 15 min", Profile(), "anomalia", "A3",
             battery_drop=30.0, instants=ONE_INSTANT),
    Scenario("esparsa", "nó sem vizinhos (1 registro a cada 20 min)", Profile(sparse=True),
             "dados_insuficientes", instants=ONE_INSTANT, episodes_factor=1 / 3),
    Scenario("nao_finito", "RSSI NaN num registro", Profile(), "entrada_invalida", nan_rssi=True,
             instants=ONE_INSTANT, episodes_factor=1 / 3),
    Scenario("sem_modelo", "primeira tentativa de treino sem referência", Profile(), "sem_modelo",
             instants=ONE_INSTANT, episodes_factor=1 / 3),
)
SCENARIO_BY_KEY = {s.key: s for s in SCENARIOS}
DETECTORS = ("isolation_forest", "lstm_ae", "cascata", "fusao")
DETECTOR_LABELS = {"isolation_forest": "Isolation Forest", "lstm_ae": "LSTM-AE",
                   "cascata": "cascata IF>LSTM", "fusao": "fusão IF+LSTM"}


# ─── Geração ────────────────────────────────────────────────────────────────

class _Node:
    """Estado do nó monitorado entre beacons (contador, uptime, bateria enviada)."""

    def __init__(self, rng: np.random.Generator, start: datetime):
        self.rng = rng
        self.sequence = 0
        self.boot = start - timedelta(hours=3)
        self.battery = 90.0
        self.last_sent = None             # (instante, valor)

    def battery_field(self, ts: datetime, noise: float):
        measured = int(round(self.battery + float(self.rng.normal(0, noise))))
        due = (self.last_sent is None or abs(measured - self.last_sent[1]) >= BATTERY_CHANGE
               or (ts - self.last_sent[0]).total_seconds() >= BATTERY_HEARTBEAT_SECONDS)
        if not due:
            return ""
        self.last_sent = (ts, measured)
        return measured


def _beacons(writer, rng: np.random.Generator, node: _Node, start: datetime, minutes: int,
             profile: Profile, battery_drop: float = 0.0, nan_at: datetime | None = None) -> None:
    """Beacons de ``minutes`` minutos a partir de ``start``; ``battery_drop`` nos 15 min finais."""
    n = minutes * 60 // BEACON_SECONDS
    drop_from = n - 15 * 60 // BEACON_SECONDS
    nan_pending = nan_at is not None
    sparse_every = 20 * 60 // BEACON_SECONDS
    for i in range(n):
        ts = start + timedelta(seconds=i * BEACON_SECONDS + float(rng.uniform(0.5, 1.5)))
        node.battery -= profile.drain_per_hour * BEACON_SECONDS / 3600
        if battery_drop and i >= drop_from:
            node.battery -= battery_drop / (n - drop_from)
        node.sequence += 1
        rssi = [float(rng.normal(profile.rssi - 4 * k, 1.5)) for k in range(2)]
        # Os vizinhos observam o nó (enlace reverso) nos próprios beacons, que não se perdem.
        for k, addr in enumerate((PARENT, OTHER)):
            writer.writerow(("vizinho", (ts + timedelta(seconds=7 + 6 * k)).isoformat(), addr, NODE,
                             round(rssi[k] + float(rng.normal(-1, 1.5)), 2), "", "", "", "", "", ""))
        battery = node.battery_field(ts, profile.battery_noise)
        if profile.loss and rng.random() < profile.loss:
            continue
        uptime = int((ts - node.boot).total_seconds() // 60)
        writer.writerow(("telemetria", ts.isoformat(), NODE, "", "", "", "", "", battery,
                         node.sequence, uptime))
        if profile.sparse and i % sparse_every:
            continue
        links = ((PARENT, 0),) if profile.sparse else ((PARENT, 0), (OTHER, 1))
        for addr, k in links:
            value = round(rssi[k], 2)
            if nan_pending and ts >= nan_at:
                value, nan_pending = "nan", False
            writer.writerow(("vizinho", ts.isoformat(), NODE, addr, value,
                             round(float(rng.normal(profile.snr - k, 0.8)), 2),
                             int(np.clip(round(rng.normal(profile.link_quality, 0.5)), 0, 10)),
                             int(addr == PARENT), "", "", ""))


def generate(directory: Path, seed: int, episodes: int = 30) -> None:
    """Grava treino.csv, avaliacao.csv e instantes.csv em ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    train_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1)  # gerador separado da referência
    reference_end = REFERENCE_START + timedelta(hours=REFERENCE_HOURS)
    with (directory / TRAIN_FILE).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        _beacons(writer, train_rng, _Node(train_rng, REFERENCE_START), REFERENCE_START,
                 REFERENCE_HOURS * 60, Profile())
    start = reference_end + timedelta(hours=1)
    node = _Node(eval_rng, start)
    with (directory / EVAL_FILE).open("w", encoding="utf-8", newline="") as fh, \
            (directory / INSTANTS_FILE).open("w", encoding="utf-8", newline="") as fi:
        writer, instants = csv.writer(fh), csv.writer(fi)
        writer.writerow(COLUMNS)
        instants.writerow(INSTANT_COLUMNS)
        for scenario in SCENARIOS:
            for ep in range(max(1, round(episodes * scenario.episodes_factor))):
                node.battery = float(eval_rng.uniform(80, 95))
                _beacons(writer, eval_rng, node, start, ONSET_MINUTES, Profile())
                onset = start + timedelta(minutes=ONSET_MINUTES)
                end = start + timedelta(minutes=EPISODE_MINUTES)
                _beacons(writer, eval_rng, node, onset, EPISODE_MINUTES - ONSET_MINUTES, scenario.profile,
                         battery_drop=scenario.battery_drop,
                         nan_at=end - timedelta(minutes=2) if scenario.nan_rssi else None)
                for name, minute, criterion in scenario.instants:
                    instants.writerow((scenario.key, ep, name, (start + timedelta(minutes=minute)).isoformat(),
                                       int(criterion)))
                start = end
            start += timedelta(hours=1)


# ─── Leitura: CSV como fonte do extrator de features do backend ─────────────

def _number(text: str):
    return None if text == "" else float(text)  # "nan" vira NaN e é tratado adiante


class CsvFeatureExtractor(FeatureExtractor):
    """``FeatureExtractor`` que lê do CSV em vez do banco.

    Só ``extract_for_window`` é substituído (mesmo intervalo ``(início, fim]``, enlaces
    reversos e proveniência); janelas deslizantes, ``extract_features`` e o cálculo
    das features são os do backend.
    """

    def __init__(self, paths):
        super().__init__()
        neighbors, telemetry = defaultdict(list), defaultdict(list)
        for path in paths:
            with path.open(encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                if tuple(reader.fieldnames or ()) != COLUMNS:
                    raise ValueError(f"{path.name}: colunas inesperadas")
                for row in reader:
                    node_id = NODE_IDS.get(row["no"])
                    if node_id is None:
                        raise ValueError(f"{path.name}: nó desconhecido {row['no']!r}")
                    ts = datetime.fromisoformat(row["timestamp"])
                    if row["registro"] == "telemetria":
                        telemetry[node_id].append(SimpleNamespace(
                            node_id=node_id, timestamp=ts, data_origin="real",
                            battery_level=None if row["battery_level"] == "" else int(row["battery_level"]),
                            sequence=int(row["sequence"]), uptime=int(row["uptime"])))
                    elif row["registro"] == "vizinho":
                        neighbors[node_id].append(SimpleNamespace(
                            node_id=node_id, timestamp=ts, data_origin="real",
                            neighbor_address=row["endereco"], rssi=_number(row["rssi"]),
                            snr=_number(row["snr"]),
                            link_quality=None if row["link_quality"] == "" else int(row["link_quality"]),
                            is_parent=row["is_parent"] == "1"))
                    else:
                        raise ValueError(f"{path.name}: tipo de registro desconhecido")
        self._telemetry = {k: self._index(v) for k, v in telemetry.items()}
        self._neighbors = {k: self._index(v) for k, v in neighbors.items()}
        reverse = defaultdict(list)
        for node_id, rows in neighbors.items():
            for row in rows:
                reverse[(node_id, row.neighbor_address)].append(row)
        self._reverse = {k: self._index(v) for k, v in reverse.items()}
        self._cache: dict = {}
        stamps = [r.timestamp for index in (*self._telemetry.values(), *self._neighbors.values())
                  for r in index[1]]
        self.time_range = (min(stamps), max(stamps)) if stamps else (None, None)

    @staticmethod
    def _index(rows):
        rows.sort(key=lambda r: r.timestamp)
        return [r.timestamp for r in rows], rows

    @staticmethod
    def _slice(index, start, end):
        if index is None:
            return []
        stamps, rows = index
        return rows[bisect.bisect_right(stamps, start):bisect.bisect_right(stamps, end)]

    def extract_for_window(self, session, node_id, window_start, window_end, include_synthetic=True,
                           provenance=None, origin_classifier=None):
        key = (node_id, window_start, window_end)
        if key not in self._cache:
            self._cache[key] = self._compute(node_id, window_start, window_end)
        result, prov = self._cache[key]
        if provenance is not None:
            provenance.update(prov)
        return None if result is None else dict(result)

    def _compute(self, node_id, start, end):
        address = next(a for a, i in NODE_IDS.items() if i == node_id)
        neighbors = self._slice(self._neighbors.get(node_id), start, end)
        telemetries = self._slice(self._telemetry.get(node_id), start, end)
        reverse_by_link, reverse_all, sources = {}, [], {node_id}
        for nb_addr in sorted({n.neighbor_address for n in neighbors}):
            nb_id = NODE_IDS.get(nb_addr)
            reverse = self._slice(self._reverse.get((nb_id, address)), start, end) if nb_id else []
            reverse_all.extend(reverse)
            if reverse:
                sources.add(nb_id)
                reverse_by_link[nb_addr] = reverse
        prov: dict = {}
        if len(neighbors) < 2:
            _fill_provenance(prov, neighbors, telemetries, reverse_all, sources, [], _default_origin)
            return None, prov
        # Última leitura de bateria até BATTERY_LOOKBACK antes da janela, como no backend.
        before = [t for t in self._slice(self._telemetry.get(node_id), start - BATTERY_LOOKBACK, start)
                  if t.battery_level is not None and t.battery_level >= 0]
        baseline = before[-1] if before else None
        result = compute_window_features(neighbors, telemetries, reverse_by_link,
                                         battery_baseline=baseline, window_end=end)
        defaulted = [name for name in self.feature_names
                     if name in result["_unavailable"] and result[name] is not None]
        # Rede sem contexto de propagação, como um cadastro sem posições no backend.
        result.update({"_classification_mode": "contextual_fallback",
                       "_context_reason": "network_without_context"})
        _fill_provenance(prov, neighbors, telemetries, reverse_all, sources, defaulted, _default_origin,
                         extra_rows=[baseline] if baseline is not None else [])
        return result, prov


class _Session:
    """Substitui a sessão do banco onde o runner consulta o cadastro do nó."""

    def get(self, model, key):
        return SimpleNamespace(node_id=key, sim_mode=False)


def read_instants(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != INSTANT_COLUMNS:
            raise ValueError(f"{path.name}: colunas inesperadas")
        rows = []
        for row in reader:
            if row["cenario"] not in SCENARIO_BY_KEY:
                raise ValueError(f"cenário desconhecido em {path.name}: {row['cenario']}")
            rows.append({**row, "timestamp": datetime.fromisoformat(row["timestamp"]),
                         "criterio": row["criterio"] == "1"})
    return rows


# ─── Decisão e estados ──────────────────────────────────────────────────────

def component_state(result) -> str:
    """Estado de um componente (AlgoResult); indisponibilidade nunca é normal."""
    if result is None:
        return "erro"
    if result.status == "valid":
        return "anomalia" if result.is_anomaly else "normal"
    if result.status in ("warming_up", "disabled"):
        return "sem_modelo"
    if "non_finite" in (result.reason or ""):
        return "entrada_invalida"
    if result.status == "insufficient_history":
        return "dados_insuficientes"
    return "erro"


PRIORITY = ("erro", "entrada_invalida", "sem_modelo", "dados_insuficientes")


def combination_state(outcome, primary_state: str, secondary_state: str) -> str:
    if outcome.status == COMBO_VALID:
        return "anomalia" if outcome.is_anomaly else "normal"
    if outcome.reason == "tau_unavailable":
        return "sem_modelo"
    for state in PRIORITY:
        if state in (primary_state, secondary_state):
            return state
    return "erro"


def _types(features, vector, score_norm) -> list[str]:
    """Tipos A1–A7 como no laço do backend: só após a decisão, sobre a janela do IF."""
    if vector is None or features is None:
        return ["AX"]
    return [a["anomaly_type"] for a in classify_anomaly_type(features, -min(score_norm * 0.5, 0.5))]


def decide(pipeline, extractor, cfg, t: datetime) -> dict:
    """Decisão de cada detector no instante ``t`` (mesmo caminho de _run_online_pipeline)."""
    features = extractor.extract_features(None, 1, window_minutes=cfg.if_window_minutes, reference_time=t)
    vec, reason = feature_dict_to_vector(features)
    ctx = DetectionContext(session=_Session(), node_id=1, feature_extractor=extractor,
                           features_vec=None if vec is None else np.array(vec, dtype=np.float32),
                           network_node_ids=[1], network_id=1, reference_time=t,
                           reference_policy=ReferencePolicy.empty(), features_reason=reason)
    outcome = pipeline.detect_detailed(ctx)
    p, s = outcome.primary, outcome.secondary
    out = {}
    for key, res in (("isolation_forest", p), ("lstm_ae", s)):
        if res is None:
            continue
        state = component_state(res)
        out[key] = {"estado": state, "score_norm": res.score_norm,
                    "motivo": None if res.is_valid else res.reason,
                    "tipos": _types(features, vec, res.score_norm) if state == "anomalia" else []}
    if s is None:
        return out
    # Mesmo cálculo do combinador em detect_detailed, sobre os componentes já avaliados.
    combos = {"cascata": Combiner("cascade").evaluate(p, s),
              "fusao": Combiner("fusion").evaluate(p, s, alpha=pipeline.fusion_alpha, tau=pipeline.tau,
                                                   fusion_scores=pipeline.fusion_ranks(p, s))}
    for key, combo in combos.items():
        state = combination_state(combo, out["isolation_forest"]["estado"], out["lstm_ae"]["estado"])
        out[key] = {"estado": state, "score_norm": combo.score, "motivo": combo.reason,
                    "tipos": _types(features, vec, combo.score) if state == "anomalia" else []}
    return out


def _build_pipeline(cfg, with_lstm: bool):
    from ai.online.pipeline import OnlinePipeline
    if with_lstm:
        pipeline = OnlinePipeline("isolation_forest", "lstm_ae", "cascade", cfg.fusion_alpha)
    else:
        pipeline = OnlinePipeline("isolation_forest", None, "single", cfg.fusion_alpha)
    pipeline.build_runners(cfg)
    if not pipeline.is_ready():
        raise RuntimeError("pipeline não subiu")
    if with_lstm and pipeline.status()["secondary_id"] != "lstm_ae":
        raise RuntimeError("LSTM-AE indisponível (PyTorch ausente ou ML_LSTM_ENABLED=false); "
                           "use --sem-lstm para avaliar só o Isolation Forest")
    return pipeline


def evaluate(directory: Path, with_lstm: bool = True) -> dict:
    """Treina na referência, calibra e avalia os instantes; decisões por detector."""
    from config import settings
    cfg = resolve_effective_config(None)
    reference = CsvFeatureExtractor([directory / TRAIN_FILE])
    trial = CsvFeatureExtractor([directory / EVAL_FILE])
    if reference.time_range[1] is None or trial.time_range[0] is None:
        raise ValueError("CSV sem registros")
    if reference.time_range[1] >= trial.time_range[0]:
        raise ValueError("avaliação deve ser posterior ao treino")
    instants = read_instants(directory / INSTANTS_FILE)
    missing = [s.key for s in SCENARIOS if s.key not in {i["cenario"] for i in instants}]
    if missing:
        raise ValueError(f"cenários ausentes em {INSTANTS_FILE}: {', '.join(missing)}")
    if any(i["timestamp"] <= reference.time_range[1] for i in instants):
        raise ValueError("instante de teste dentro da referência")

    both = CsvFeatureExtractor([directory / TRAIN_FILE, directory / EVAL_FILE])
    reference_end = reference.time_range[1]
    ctx = DetectionContext(session=_Session(), node_id=1, feature_extractor=both, features_vec=None,
                           network_node_ids=[1], network_id=1, reference_time=reference_end,
                           reference_policy=ReferencePolicy.empty())
    retry = settings.ML_ONLINE_TRAIN_RETRY_SECONDS
    settings.ML_ONLINE_TRAIN_RETRY_SECONDS = 10 ** 9   # sem nova tentativa durante o teste
    try:
        pipeline = _build_pipeline(cfg, with_lstm)
        if not pipeline.train(ctx):
            status = pipeline.status()
            raise RuntimeError(f"treino falhou: IF={status['primary']['last_error']}; "
                               f"LSTM={(status['secondary'] or {}).get('last_error')}")
        # Cada componente tenta treinar antes de haver referência e fica na espera da
        # nova tentativa. (pipeline.train para no primeiro componente que falha; o
        # outro treinaria sob demanda no instante avaliado.)
        untrained = _build_pipeline(cfg, with_lstm)
        no_reference = replace(ctx, reference_time=reference.time_range[0])
        for runner in (untrained._primary_runner, untrained._secondary_runner):
            if runner is not None and runner.ensure_trained(no_reference):
                raise RuntimeError("componente sem referência não deveria treinar")

        results = []
        for inst in instants:
            target = untrained if inst["cenario"] == "sem_modelo" else pipeline
            try:
                decision = decide(target, both, cfg, inst["timestamp"])
            except Exception as exc:   # erro de execução é estado próprio, nunca normal
                names = DETECTORS if with_lstm else DETECTORS[:1]
                decision = {d: {"estado": "erro", "score_norm": None, "tipos": [],
                                "motivo": f"{type(exc).__name__}: {exc}"} for d in names}
            for detector, d in decision.items():
                results.append({"cenario": inst["cenario"], "episodio": int(inst["episodio"]),
                                "instante": inst["instante"], "timestamp": inst["timestamp"].isoformat(),
                                "criterio": inst["criterio"], "detector": detector, **d})
        calibration = _calibration(pipeline, both, cfg, reference_end)
        status = pipeline.status()
    finally:
        settings.ML_ONLINE_TRAIN_RETRY_SECONDS = retry
    summary = {
        "configuracao": {k: getattr(cfg, k) for k in (
            "if_window_minutes", "if_step_minutes", "if_history_hours", "if_contamination",
            "if_n_estimators", "lstm_window_size", "lstm_window_minutes", "lstm_history_hours",
            "lstm_threshold_percentile", "lstm_score_aggregation", "fusion_alpha", "random_seed")},
        "treino": {"isolation_forest": status["primary"]["training_summary"],
                   "lstm_ae": (status["secondary"] or {}).get("training_summary")},
        "tau_fusao": status["tau"], "tau_status": status["tau_status"],
        "calibracao": calibration,
    }
    return {"resultados": results, "resumo": summary}


def _calibration(pipeline, extractor, cfg, reference_end) -> dict:
    """Alarmes no bloco de calibração (após validação e treino dos dois modelos).

    No LSTM é a taxa dentro da amostra que fixou o limiar p99 (≈1% por construção);
    no IF, que não usa o bloco, é fora da amostra.
    """
    runner = pipeline._secondary_runner
    holdout = getattr(runner, "calibration_set", None)
    if holdout is None or holdout.cutoff is None:
        return {}
    out = {"inicio": holdout.cutoff.isoformat(), "fim": reference_end.isoformat()}
    _, flags, _ = runner.score_val(holdout.X)
    out["lstm_ae"] = {"sequencias": int(len(flags)), "alarmes": int(np.sum(flags))}
    t, alarms, n = holdout.cutoff + timedelta(minutes=cfg.if_window_minutes), 0, 0
    while t <= reference_end:
        d = decide(pipeline, extractor, cfg, t)["isolation_forest"]
        if d["estado"] in ("normal", "anomalia"):
            n += 1
            alarms += d["estado"] == "anomalia"
        t += timedelta(minutes=cfg.if_step_minutes)
    out["isolation_forest"] = {"janelas": n, "alarmes": alarms}
    return out


# ─── Veredito ───────────────────────────────────────────────────────────────

def verdict(scenario: Scenario, rows: list[dict]) -> dict:
    graded = [r for r in rows if r["criterio"]]
    n = len(graded)
    states = [r["estado"] for r in graded]
    counts = {s: states.count(s) for s in STATES if states.count(s)}
    false_positives = missed = 0
    checks = []
    if scenario.expected_state == "normal":
        false_positives = states.count("anomalia")
        checks.append(("falso positivo", false_positives / n, f"≤ {MAX_FALSE_POSITIVE_RATE:.2f}",
                       false_positives / n <= MAX_FALSE_POSITIVE_RATE))
        other = sum(s not in ("normal", "anomalia") for s in states) / n
        checks.append(("estado não decidido", other, "= 0.00", other == 0))
    elif scenario.expected_state == "anomalia":
        detected = [r for r in graded if r["estado"] == "anomalia"]
        missed = n - len(detected)
        checks.append(("detecção", len(detected) / n, f"≥ {MIN_DETECTION_RATE:.2f}",
                       len(detected) / n >= MIN_DETECTION_RATE))
        typed = sum(scenario.expected_type in r["tipos"] for r in detected) / max(len(detected), 1)
        checks.append((f"tipo {scenario.expected_type}", typed, f"≥ {MIN_TYPE_RATE:.2f}", typed >= MIN_TYPE_RATE))
    else:
        rate = states.count(scenario.expected_state) / n
        checks.append((scenario.expected_state, rate, f"= {EXACT_STATE_RATE:.2f}", rate >= EXACT_STATE_RATE))
        normal = states.count("normal") / n
        checks.append(("tratado como normal", normal, "= 0.00", normal == 0))
    informative = {}
    for name in sorted({r["instante"] for r in rows if not r["criterio"]}):
        sub = [r["estado"] for r in rows if r["instante"] == name and not r["criterio"]]
        informative[name] = {s: sub.count(s) for s in STATES if sub.count(s)}
    scores = [r["score_norm"] for r in graded if r["score_norm"] is not None]
    return {"cenario": scenario.key, "descricao": scenario.description, "instantes": n,
            "esperado": scenario.expected_state + (f" ({scenario.expected_type})" if scenario.expected_type else ""),
            "obtido": counts, "falsos_positivos": false_positives, "nao_detectadas": missed,
            "score_norm_mediano": round(float(np.median(scores)), 4) if scores else None,
            "informativo": informative,
            "verificacoes": [{"criterio": c, "valor": round(v, 4), "limite": lim, "aprovado": ok}
                             for c, v, lim, ok in checks],
            "aprovado": all(ok for *_, ok in checks)}


def run(directory: Path, with_lstm: bool = True) -> tuple[dict, dict]:
    """{detector: [veredito por cenário]} e o resultado bruto da avaliação."""
    evaluation = evaluate(directory, with_lstm)
    by = defaultdict(list)
    for r in evaluation["resultados"]:
        by[(r["detector"], r["cenario"])].append(r)
    detectors = [d for d in DETECTORS if any(k[0] == d for k in by)]
    report = {d: [verdict(s, by[(d, s.key)]) for s in SCENARIOS] for d in detectors}
    return report, evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--episodios", type=int, default=30,
                        help="episódios do controle e de cada anomalia (padrão: 30; os demais, 1/3)")
    parser.add_argument("--sem-lstm", action="store_true",
                        help="avalia só o Isolation Forest (ambiente sem PyTorch)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dados", type=Path, help="pasta onde gravar e manter os CSVs sintéticos")
    source.add_argument("--entrada", type=Path, help="pasta com treino.csv, avaliacao.csv e instantes.csv")
    parser.add_argument("--csv", type=Path, help="grava a decisão de cada detector por instante neste CSV")
    parser.add_argument("--json", action="store_true", help="imprime o relatório em JSON")
    args = parser.parse_args()
    if args.episodios < 3:
        parser.error("--episodios deve ser ao menos 3")
    try:
        with tempfile.TemporaryDirectory(prefix="autoteste_") as tmp:
            directory = args.entrada or args.dados or Path(tmp)
            if not args.entrada:
                generate(directory, args.seed, args.episodios)
            report, evaluation = run(directory, with_lstm=not args.sem_lstm)
    except Exception as exc:
        print(f"ERRO de execução: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as fh:
            fields = ["cenario", "episodio", "instante", "timestamp", "criterio", "detector",
                      "estado", "tipos", "score_norm", "motivo"]
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in evaluation["resultados"]:
                writer.writerow({**{k: r[k] for k in fields}, "tipos": "|".join(r["tipos"]),
                                 "criterio": int(r["criterio"]),
                                 "score_norm": "" if r["score_norm"] is None else round(r["score_norm"], 6)})
    passed = all(v["aprovado"] for verdicts in report.values() for v in verdicts)
    summary = evaluation["resumo"]
    if args.json:
        print(json.dumps({"seed": args.seed, "episodios": args.episodios, "aprovado": passed,
                          "resumo": summary, "detectores": report}, ensure_ascii=False, indent=2,
                         default=str))
        return 0 if passed else 1
    origin = f"CSVs de {args.entrada}" if args.entrada else "CSVs sintéticos gerados"
    cfg = summary["configuracao"]
    print(f"Autoteste da detecção — seed {args.seed}; {origin}")
    print(f"Configuração do backend: IF em janelas de {cfg['if_window_minutes']} min "
          f"(contaminação {cfg['if_contamination']}); LSTM-AE com {cfg['lstm_window_size']}×"
          f"{cfg['lstm_window_minutes']} min (limiar p{cfg['lstm_threshold_percentile']:g}); "
          f"fusão α={cfg['fusion_alpha']}, τ={summary['tau_fusao']} ({summary['tau_status']})")
    tr = summary["treino"]
    lstm = tr.get("lstm_ae") or {}
    print(f"Treino: IF {tr['isolation_forest'].get('n_windows')} janelas; LSTM "
          f"{lstm.get('n_train', '—')} sequências (validação: {lstm.get('n_val', '—')}, "
          f"calibração: {lstm.get('n_calib', '—')}); cortes {lstm.get('cutoff', '—')} e "
          f"{lstm.get('calib_cutoff', '—')}")
    cal = summary["calibracao"]
    if cal:
        print(f"Calibração (bloco final da referência, fora do treino): IF "
              f"{cal['isolation_forest']['alarmes']}/{cal['isolation_forest']['janelas']} alarmes; "
              f"LSTM {cal['lstm_ae']['alarmes']}/{cal['lstm_ae']['sequencias']} alarmes")
    for detector, verdicts in report.items():
        ok = sum(v["aprovado"] for v in verdicts)
        print(f"\n== {DETECTOR_LABELS[detector]}: {ok}/{len(verdicts)} cenários aprovados")
        for r in verdicts:
            print(f"[{'APROVADO' if r['aprovado'] else 'REPROVADO'}] {r['cenario']}: {r['descricao']}")
            print(f"    esperado: {r['esperado']}; obtido: {r['obtido']}")
            for c in r["verificacoes"]:
                print(f"    {c['criterio']}: {c['valor']:.2f} (limite {c['limite']}) "
                      f"{'ok' if c['aprovado'] else 'FALHOU'}")
            for name, counts in r["informativo"].items():
                print(f"    (informativo, instante '{name}': {counts})")
        fp = sum(r["falsos_positivos"] for r in verdicts)
        fn = sum(r["nao_detectadas"] for r in verdicts)
        print(f"    falsos positivos: {fp}; anomalias não detectadas: {fn}")
    print(f"\nResultado: {'APROVADO' if passed else 'REPROVADO'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
