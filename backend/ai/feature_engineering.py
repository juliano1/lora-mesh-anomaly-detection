"""
Engenharia de Features para Detecção de Anomalias ML.

Extrai as features de janelas temporais (15 min no IF; subjanelas de 5 min na
sequência do LSTM-AE).

Features extraídas (11):
- rssi_mean (dBm); rssi_std (dB): média, sem ponderação, dos desvios temporais por enlace
- snr_mean (dB); snr_std (dB): média, sem ponderação, dos desvios temporais por enlace
- pdr_mean: índice heurístico de qualidade de sinal 0–10 (não é PDR medido)
- battery_delta: pontos percentuais do indicador desde a última leitura anterior à janela
  (até BATTERY_LOOKBACK antes) ou, sem ela, desde a primeira leitura da janela, até a última
- packet_loss_rate: lacunas de sequência da telemetria na época atual do contador (fração)
- rssi_snr_divergence: |média dos desvios de SNR por enlace − média dos desvios de RSSI por enlace| (dB)
- beacon_interval_std: desvio padrão do intervalo entre telemetrias (s)
- route_change_rate: trocas do parent reportado ÷ registros de parent
- link_asymmetry: média de |RSSI local − RSSI reverso| por enlace (dB)
"""

import math
import numpy as np
import logging
from datetime import datetime, timedelta
from sqlmodel import Session, select
from typing import Optional, List, Tuple
from ai.anomaly_definitions import FEATURE_ORDER

logger = logging.getLogger(__name__)

# Identidade da definição das 11 entradas (v2 = desvios por enlace).
# v3: inversão de chegada não abre época na perda.
# v4: bateria a partir da última leitura anterior à janela (envio condicional do firmware).
FEATURE_SCHEMA_VERSION = "features-v4"

# O firmware envia a bateria quando ela muda 2 p.p. ou a cada 5 min (BATTERY_SEND_INTERVAL).
# A última leitura vale por até 60 min, o contexto de uma decisão do LSTM-AE (12 × 5 min):
# sem leitura mais recente, Δ = 0 significa "nenhuma mudança informada nesse contexto". Com
# perda alta os envios periódicos somem com frequência; com 11 min (um envio perdido), um nó
# com ~50% de perda ficava sem decisão do LSTM na maioria dos ciclos. Uma mudança ≥ 2 p.p.
# cujos envios se perderam todos só aparece na próxima leitura recebida. Sem leitura em
# 60 min, a bateria fica indisponível.
BATTERY_LOOKBACK = timedelta(minutes=60)

# Motivos de indisponibilidade (fora do vetor; chave "_unavailable").
NO_ELIGIBLE_LINK = "no_eligible_link"
NO_LINK_QUALITY = "no_link_quality"
INSUFFICIENT_BATTERY_READINGS = "insufficient_battery_readings"
INSUFFICIENT_DISTINCT_COUNTERS = "insufficient_distinct_counters"
INSUFFICIENT_TELEMETRY = "insufficient_telemetry"
INSUFFICIENT_PARENT_RECORDS = "insufficient_parent_records"
NO_REVERSE_LINK = "no_reverse_link"


# Recuo de sequência aceito como troca de ordem de chegada, não como reset do nó.
_REORDER_TOLERANCE = 2


def packet_loss_estimate(telemetries) -> tuple[float, bool]:
    """Perda por lacunas de sequência na época atual do contador e sua disponibilidade.

    Os nós zeram ``sequence`` e ``uptime`` quando reiniciam. Misturar amostras
    anteriores e posteriores ao reset transforma o maior contador pré-reset em
    pacotes supostamente perdidos. As telemetrias chegam ordenadas por tempo;
    portanto, descartamos os trechos anteriores ao último reset e deduplicamos
    broadcast/unicast somente dentro da época restante.

    A estimativa só é disponível com pelo menos dois contadores
    distintos na época efetiva e intervalo positivo; caso contrário retorna
    ``(0.0, False)`` — o zero é default, não perda nula medida.
    """
    samples = [
        (int(t.sequence), getattr(t, "uptime", None))
        for t in telemetries
        if t.sequence is not None
    ]
    if len(samples) < 2:
        return 0.0, False

    epoch_start = 0
    previous_sequence, previous_uptime = samples[0]
    for index, (sequence, uptime) in enumerate(samples[1:], start=1):
        # uptime está em minutos: uma inversão de chegada na virada do minuto também o faz
        # recuar. Reset de verdade zera os dois contadores, então a sequência precisa
        # recuar além de uma simples troca de ordem.
        uptime_decreased = (
            uptime is not None
            and previous_uptime is not None
            and uptime < previous_uptime
            and previous_sequence - sequence > _REORDER_TOLERANCE
        )
        # Fallback para resets muito rápidos (uptime ainda em zero) ou dados
        # legados sem uptime. A queda precisa voltar a 0/1 e ser maior que uma
        # simples inversão de chegada entre pacotes consecutivos.
        sequence_restarted = sequence <= 1 and previous_sequence - sequence >= 2
        if uptime_decreased or sequence_restarted:
            epoch_start = index

        previous_sequence, previous_uptime = sequence, uptime

    seq_values = sorted({sequence for sequence, _ in samples[epoch_start:]})
    if len(seq_values) < 2:
        return 0.0, False

    expected = seq_values[-1] - seq_values[0]
    actual = len(seq_values) - 1
    if expected <= 0 or actual > expected:
        return 0.0, False
    return float(1.0 - (actual / expected)), True


def _packet_loss_rate_from_telemetries(telemetries) -> float:
    """Compatibilidade: só o valor de :func:`packet_loss_estimate`."""
    return packet_loss_estimate(telemetries)[0]


def _is_finite_number(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def link_temporal_stds(neighbors) -> tuple:
    """Desvios temporais de RSSI/SNR dentro de cada enlace.

    Enlace = vizinho emissor observado pelo nó da janela. Elegível com ≥2 pares
    RSSI/SNR finitos (o mesmo conjunto para as duas estatísticas). Retorna a
    média aritmética, sem ponderação, de ``np.std(ddof=0)`` por enlace elegível,
    ou ``(None, None, stats)`` sem enlace elegível. Média sem ponderação, ddof=0
    e mínimo de 2 são escolhas de projeto desta versão, não parâmetros da
    literatura; 2 observações não comprovam amostras RF independentes.
    """
    links: dict = {}
    for n in neighbors:
        if _is_finite_number(n.rssi) and _is_finite_number(n.snr):
            links.setdefault(n.neighbor_address, []).append((float(n.rssi), float(n.snr)))
    eligible = {addr: pairs for addr, pairs in links.items() if len(pairs) >= 2}
    stats = {
        "observed_links": len({n.neighbor_address for n in neighbors}),
        "eligible_links": len(eligible),
        "observations_per_link": {str(a): len(p) for a, p in sorted(links.items())},
    }
    if not eligible:
        return None, None, stats
    rssi_stds = [float(np.std([p[0] for p in pairs])) for pairs in eligible.values()]
    snr_stds = [float(np.std([p[1] for p in pairs])) for pairs in eligible.values()]
    return float(np.mean(rssi_stds)), float(np.mean(snr_stds)), stats


def _battery_delta(telemetries, baseline, window_end) -> tuple:
    """(Δ, primeira leitura usada, última leitura usada) ou ``(None, None, None)``.

    O firmware só envia a bateria quando ela muda 2 p.p. ou a cada 5 min; numa subjanela
    de 5 min é comum haver uma única leitura. Por isso a referência inicial é a última
    leitura anterior à janela (``baseline``, até BATTERY_LOOKBACK antes do início) e, sem
    ela, a primeira leitura da janela. Sem leitura na janela, a do início só vale se
    ainda tiver até BATTERY_LOOKBACK no fim da janela: Δ = 0 então é "nenhuma mudança de
    2 p.p. informada desde essa leitura", não um default.
    """
    rows = [(t.timestamp, t.battery_level) for t in telemetries
            if t.battery_level is not None and t.battery_level >= 0]
    first = (baseline.timestamp, baseline.battery_level) if baseline is not None else None
    if first is None:
        if len(rows) < 2:
            return None, None, None
        first = rows[0]
    last = rows[-1] if rows else first
    if not rows and (window_end is None or window_end - first[0] > BATTERY_LOOKBACK):
        return None, None, None
    return float(last[1] - first[1]), first[0], last[0]


def compute_window_features(neighbors, telemetries, reverse_by_link: dict,
                            battery_baseline=None, window_end=None) -> Optional[dict]:
    """As 11 features de uma janela, sem I/O.

    Entradas já filtradas por ``(início, fim]`` e pela política de origem,
    ordenadas por ``(timestamp, id)``. ``reverse_by_link``: endereço do vizinho →
    registros em que esse vizinho observa o nó (vazio na timeline).
    ``battery_baseline``: última telemetria com bateria válida até BATTERY_LOOKBACK
    antes do início (ou None); ``window_end``: fim da janela. Além das 11
    entradas, devolve chaves internas (prefixo ``_``) fora do vetor:
    ``_unavailable`` (feature → motivo), ``_link_stats``, ``_asymmetry_links``,
    ``_battery_first_at`` e ``_battery_last_at``.

    Feature listada em ``_unavailable`` não foi observada na janela. O valor numérico
    dela é só apresentação: ``feature_dict_to_vector`` recusa o vetor, e os modelos
    nunca o veem como medida saudável.
    """
    if len(neighbors) < 2:
        return None
    unavailable: dict = {}

    rssi_values = [n.rssi for n in neighbors]
    snr_values = [n.snr for n in neighbors]
    pdr_values = [n.link_quality for n in neighbors if n.link_quality is not None]

    rssi_mean = float(np.mean(rssi_values))
    snr_mean = float(np.mean(snr_values))
    rssi_std, snr_std, link_stats = link_temporal_stds(neighbors)
    if rssi_std is None:
        unavailable["rssi_std"] = NO_ELIGIBLE_LINK
        unavailable["snr_std"] = NO_ELIGIBLE_LINK
    pdr_mean = float(np.mean(pdr_values)) if pdr_values else 5.0
    if not pdr_values:
        unavailable["pdr_mean"] = NO_LINK_QUALITY

    battery_delta, battery_first_at, battery_last_at = _battery_delta(
        telemetries, battery_baseline, window_end)
    if battery_delta is None:
        battery_delta = 0.0
        unavailable["battery_delta"] = INSUFFICIENT_BATTERY_READINGS

    packet_loss_rate, loss_available = packet_loss_estimate(telemetries)
    if not loss_available:
        unavailable["packet_loss_rate"] = INSUFFICIENT_DISTINCT_COUNTERS

    # Diferença entre as médias dos desvios por enlace (fórmula preservada)
    from collections import defaultdict as _dd
    _lrssi: _dd = _dd(list)
    _lsnr: _dd = _dd(list)
    for n in neighbors:
        _lrssi[n.neighbor_address].append(n.rssi)
        _lsnr[n.neighbor_address].append(n.snr)
    _r_stds = [float(np.std(v)) for v in _lrssi.values() if len(v) >= 2]
    _s_stds = [float(np.std(v)) for v in _lsnr.values() if len(v) >= 2]
    rssi_snr_div = abs(float(np.mean(_s_stds)) - float(np.mean(_r_stds))) if (_r_stds and _s_stds) else 0.0
    if not (_r_stds and _s_stds):
        unavailable["rssi_snr_divergence"] = NO_ELIGIBLE_LINK

    beacon_interval_std = 0.0
    if len(telemetries) >= 3:
        timestamps = [t.timestamp for t in telemetries]
        intervals = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        if intervals:
            beacon_interval_std = float(np.std(intervals))
    else:
        unavailable["beacon_interval_std"] = INSUFFICIENT_TELEMETRY

    parent_records = sorted([n for n in neighbors if n.is_parent], key=lambda x: x.timestamp)
    route_change_rate = 0.0
    if len(parent_records) >= 2:
        parents_seq = [n.neighbor_address for n in parent_records]
        changes = sum(1 for i in range(1, len(parents_seq)) if parents_seq[i] != parents_seq[i - 1])
        route_change_rate = float(changes) / len(parent_records)
    else:
        unavailable["route_change_rate"] = INSUFFICIENT_PARENT_RECORDS

    asym_deltas = []
    asymmetry_links = []
    for nb_addr in sorted({n.neighbor_address for n in neighbors}):
        reverse = reverse_by_link.get(nb_addr) or []
        if not reverse:
            continue
        local = [n.rssi for n in neighbors if n.neighbor_address == nb_addr]
        our_rssi = float(np.mean(local))
        their_rssi = float(np.mean([n.rssi for n in reverse]))
        asym_deltas.append(abs(our_rssi - their_rssi))
        # Os dois sentidos do enlace: a diferença não diz qual nó a causa.
        asymmetry_links.append({
            "neighbor": nb_addr, "delta_db": round(abs(our_rssi - their_rssi), 2),
            "rssi_heard_here": round(our_rssi, 2), "rssi_heard_by_neighbor": round(their_rssi, 2),
            "records_here": len(local), "records_by_neighbor": len(reverse),
        })
    link_asymmetry = float(np.mean(asym_deltas)) if asym_deltas else 0.0
    if not asym_deltas:
        unavailable["link_asymmetry"] = NO_REVERSE_LINK

    return {
        "rssi_mean": rssi_mean,
        "rssi_std": rssi_std,
        "snr_mean": snr_mean,
        "snr_std": snr_std,
        "pdr_mean": pdr_mean,
        "battery_delta": battery_delta,
        "packet_loss_rate": packet_loss_rate,
        "rssi_snr_divergence": rssi_snr_div,
        "beacon_interval_std": beacon_interval_std,
        "route_change_rate": route_change_rate,
        "link_asymmetry": link_asymmetry,
        "_unavailable": unavailable,
        "_link_stats": link_stats,
        "_asymmetry_links": asymmetry_links,
        "_battery_first_at": battery_first_at,
        "_battery_last_at": battery_last_at,
    }


def _default_origin(row) -> str:
    """Classificação padrão de origem para a proveniência."""
    return "real" if getattr(row, "data_origin", "real") == "real" else "synthetic"


def _fill_provenance(provenance: dict, neighbors, telemetries, reverse_rows,
                     source_node_ids: set, defaulted: list, classify, extra_rows=()) -> None:
    """Proveniência de uma janela.

    Conta todas as linhas lidas (antes de qualquer filtro de origem): vizinhos
    locais, telemetria, enlaces reversos e ``extra_rows`` (a leitura de bateria
    anterior à janela). Primeira/última observação consideram só os registros
    locais da janela; ``context_start`` é o instante da linha mais antiga usada.
    """
    every = (*neighbors, *telemetries, *reverse_rows, *extra_rows)
    origin_counts: dict = {}
    for row in every:
        category = classify(row)
        origin_counts[category] = origin_counts.get(category, 0) + 1
    local_times = [row.timestamp for row in (*neighbors, *telemetries)]
    provenance.update({
        "synthetic_rows": sum(1 for row in every
                              if getattr(row, "data_origin", "real") != "real"),
        "source_node_ids": source_node_ids,
        "defaulted_features": defaulted,
        "neighbor_record_count": len(neighbors),
        "distinct_neighbor_count": len({n.neighbor_address for n in neighbors}),
        "telemetry_record_count": len(telemetries),
        "reverse_record_count": len(reverse_rows),
        "origin_counts": origin_counts,
        "first_observation": min(local_times) if local_times else None,
        "context_start": min((row.timestamp for row in extra_rows), default=None),
        "last_observation": max(local_times) if local_times else None,
    })


def _is_window_contaminated(node_id: int, time_start, time_end, session) -> bool:
    """
    Retorna True se a janela [time_start, time_end] se sobrepõe com algum
    SimulationRun para o nó — indica dados sintéticos a excluir do treino.
    """
    from models import SimulationRun
    runs = session.exec(
        select(SimulationRun)
        .where(SimulationRun.node_id == node_id)
        .where(SimulationRun.started_at < time_end)
        .where(
            (SimulationRun.ended_at == None) |
            (SimulationRun.ended_at > time_start)
        )
    ).all()
    return len(runs) > 0


class FeatureExtractor:
    """Extrator de features para detecção ML."""

    # Valores válidos para a flag de modo de classificação
    CLASSIFICATION_MODES = frozenset({"contextual", "absolute"})

    def __init__(self):
        # Flag global de modo de classificação (lida de MLSettings via apply_settings)
        # "contextual" → registra o modo "contextual_fallback" e o motivo de não usar
        # contexto (nenhum resíduo é calculado); "absolute" → registra "absolute".
        self.anomaly_classification_mode: str = "contextual"

        self.feature_names = list(FEATURE_ORDER)
    
    def extract_features(self, session: Session, node_id: int,
                        window_minutes: int = 5, since: datetime = None,
                        exclude_sim: bool = False,
                        reference_time: datetime = None) -> dict | None:
        """
        Extrai features da janela recente de um nó, com corte explícito.

        A janela é ``(início, fim]`` com ``fim = reference_time``
        (padrão: agora) e ``início = since`` ou ``fim − window_minutes``. Nenhum
        registro posterior ao corte entra — inclusive enlaces reversos. Mesmo
        cálculo de ``extract_for_window`` (fonte única).

        Args:
            session: SQLModel session
            node_id: ID do nó
            window_minutes: Tamanho da janela em minutos (padrão 5)
            since: Início explícito da janela (substitui window_minutes)
            exclude_sim: True descarta janela sobreposta a SimulationRun
            reference_time: Instante de corte da decisão (padrão: utcnow)

        Returns:
            Dict com features ou None se dados insuficientes
        """
        time_end = reference_time if reference_time is not None else datetime.utcnow()
        time_start = since if since is not None else time_end - timedelta(minutes=window_minutes)
        if exclude_sim and _is_window_contaminated(node_id, time_start, time_end, session):
            logger.debug(f"Node {node_id}: janela contaminada por simulação — excluído do treino")
            return None
        return self.extract_for_window(session, node_id, time_start, time_end)

    def extract_for_window(self, session: Session, node_id: int,
                          window_start: datetime, window_end: datetime,
                          include_synthetic: bool = True,
                          provenance: Optional[dict] = None,
                          origin_classifier=None) -> dict | None:
        """
        Extrai features para a janela ``(window_start, window_end]``.

        Usado pelo sliding window de treino para garantir que treino e inferência
        usem a mesma escala temporal.

        Args:
            include_synthetic: False filtra linhas com data_origin != 'real'.
            provenance: se informado, recebe
                ``synthetic_rows`` — linhas não reais lidas (antes do filtro),
                ``source_node_ids`` — nós cujos dados alimentaram a janela
                (o próprio e os de enlaces reversos),
                ``defaulted_features`` — features preenchidas por default por
                falta de observação (não são medições de saúde),
                neighbor_record_count, distinct_neighbor_count,
                telemetry_record_count, reverse_record_count,
                origin_counts (categoria → linhas), first_observation e
                last_observation (registros locais)
            origin_classifier: função linha → categoria de origem usada só nas
                contagens da proveniência (padrão: real/synthetic por data_origin).
        """
        from models import Neighbor, Telemetry

        neighbors = session.exec(
            select(Neighbor)
            .where(Neighbor.node_id == node_id)
            .where(Neighbor.timestamp > window_start)
            .where(Neighbor.timestamp <= window_end)
            .order_by(Neighbor.timestamp, Neighbor.id)
        ).all()

        classify = origin_classifier or _default_origin
        all_neighbors = list(neighbors)
        reverse_all: list = []
        source_node_ids = {node_id}

        if not include_synthetic:
            neighbors = [n for n in neighbors if getattr(n, "data_origin", "real") == "real"]

        if len(neighbors) < 2:
            if provenance is not None:
                # Janela esparsa não gera features,
                # mas sua origem precisa ser apurada — senão vira lacuna "limpa" e é
                # imputada como saudável. Conta telemetria e enlaces reversos também.
                aux_telemetries, aux_reverse = self._auxiliary_rows(
                    session, node_id, window_start, window_end)
                source_node_ids.update(row.node_id for row in aux_reverse)
                _fill_provenance(provenance, all_neighbors, aux_telemetries, aux_reverse,
                                 source_node_ids, [], classify)
            return None

        telemetries = session.exec(
            select(Telemetry)
            .where(Telemetry.node_id == node_id)
            .where(Telemetry.timestamp > window_start)
            .where(Telemetry.timestamp <= window_end)
            .order_by(Telemetry.timestamp, Telemetry.id)
        ).all()
        all_telemetries = list(telemetries)
        if not include_synthetic:
            telemetries = [t for t in telemetries if getattr(t, "data_origin", "real") == "real"]

        # Enlaces reversos: consultas e proveniência aqui; o cálculo fica no núcleo puro.
        reverse_by_link: dict = {}
        try:
            from models import Node as _Node
            my_node = session.exec(select(_Node).where(_Node.node_id == node_id)).first()
            if my_node and my_node.lora_address:
                my_address = my_node.lora_address
                for nb_addr in sorted({n.neighbor_address for n in neighbors}):
                    nb_node = session.exec(
                        select(_Node).where(_Node.lora_address == nb_addr)
                    ).first()
                    if not nb_node:
                        continue
                    reverse = session.exec(
                        select(Neighbor)
                        .where(Neighbor.node_id == nb_node.node_id)
                        .where(Neighbor.neighbor_address == my_address)
                        .where(Neighbor.timestamp > window_start)
                        .where(Neighbor.timestamp <= window_end)
                        .order_by(Neighbor.timestamp, Neighbor.id)
                    ).all()
                    reverse_all.extend(reverse)
                    if reverse:
                        source_node_ids.add(nb_node.node_id)
                    if not include_synthetic:
                        reverse = [n for n in reverse if getattr(n, "data_origin", "real") == "real"]
                    if reverse:
                        reverse_by_link[nb_addr] = list(reverse)
        except Exception as _e:
            logger.debug(f"Nó {node_id}: link_asymmetry indisponível na janela ({_e})")

        baseline_rows = session.exec(
            select(Telemetry)
            .where(Telemetry.node_id == node_id)
            .where(Telemetry.timestamp > window_start - BATTERY_LOOKBACK)
            .where(Telemetry.timestamp <= window_start)
            .where(Telemetry.battery_level != None)  # noqa: E711
            .where(Telemetry.battery_level >= 0)
            .order_by(Telemetry.timestamp.desc(), Telemetry.id.desc())
            .limit(1)
        ).all()
        all_baseline = list(baseline_rows)
        if not include_synthetic:
            baseline_rows = [t for t in baseline_rows if getattr(t, "data_origin", "real") == "real"]
        baseline = baseline_rows[0] if baseline_rows else None

        result = compute_window_features(neighbors, telemetries, reverse_by_link,
                                         battery_baseline=baseline, window_end=window_end)
        # Valor de apresentação de feature não observada (o vetor é recusado).
        defaulted = [name for name in self.feature_names
                     if name in result["_unavailable"] and result[name] is not None]
        # Modo contextual efetivo (extra — fora do vetor)
        result.update(self._compute_context_residuals(session, node_id))
        if provenance is not None:
            _fill_provenance(provenance, all_neighbors, all_telemetries, reverse_all,
                             source_node_ids, defaulted, classify, extra_rows=all_baseline)
        return result

    def _auxiliary_rows(self, session: Session, node_id: int,
                        window_start: datetime, window_end: datetime) -> tuple:
        """(telemetria do nó, enlaces reversos sobre o nó) na janela ``(início, fim]``.

        Usado quando a janela não tem vizinhos suficientes para features: a
        proveniência ainda precisa refletir toda fonte que alimentaria a janela.
        """
        from models import Neighbor, Telemetry, Node as _Node

        telemetries = session.exec(
            select(Telemetry)
            .where(Telemetry.node_id == node_id)
            .where(Telemetry.timestamp > window_start)
            .where(Telemetry.timestamp <= window_end)
            .order_by(Telemetry.timestamp, Telemetry.id)
        ).all()
        my_node = session.exec(select(_Node).where(_Node.node_id == node_id)).first()
        if not my_node or not my_node.lora_address:
            return list(telemetries), []
        reverse = session.exec(
            select(Neighbor)
            .where(Neighbor.neighbor_address == my_node.lora_address)
            .where(Neighbor.node_id != node_id)
            .where(Neighbor.timestamp > window_start)
            .where(Neighbor.timestamp <= window_end)
            .order_by(Neighbor.timestamp, Neighbor.id)
        ).all()
        return list(telemetries), list(reverse)

    def _compute_context_residuals(self, session: Session, node_id: int) -> dict:
        """Modo contextual efetivo da classificação.

        A distância disponível no banco (``Neighbor.distance_m``) é estimada a
        partir do próprio RSSI (firmware ``estimateDistance`` e ingestão MQTT).
        Prever RSSI por essa distância e compará-lo ao RSSI observado reutiliza a
        mesma medida; por isso o resíduo de RSSI não é calculado até existir
        distância independente documentada. Também não há resíduo do índice de
        qualidade, que nenhuma regra A1–A7 usa. Na prática, o método só registra
        o modo de classificação e, no modo contextual, o motivo do fallback.

        Returns:
            ``{"_classification_mode": "absolute"}`` quando a flag força absoluto;
            senão ``{"_classification_mode": "contextual_fallback",
            "_context_reason": motivo}`` com motivo ``distance_not_independent``,
            ``network_without_context``, ``node_without_network`` ou
            ``context_error``.
        """
        if self.anomaly_classification_mode == "absolute":
            return {"_classification_mode": "absolute"}
        try:
            from models import Node as _Node, Network as _Network
            from ai.propagation import get_propagation_params

            my_node = session.exec(select(_Node).where(_Node.node_id == node_id)).first()
            if not my_node or not my_node.network_id:
                reason = "node_without_network"
            else:
                network = session.get(_Network, my_node.network_id)
                has_context = bool(network) and get_propagation_params(network)["has_context"]
                reason = "distance_not_independent" if has_context else "network_without_context"
        except Exception as _e:
            logger.debug(f"Node {node_id}: contexto indisponível ({_e})")
            reason = "context_error"
        return {"_classification_mode": "contextual_fallback", "_context_reason": reason}

    def get_available_history_hours(self, session: Session) -> float:
        """
        Retorna quantas horas de dados existem no banco (para todos os nós).

        Usado por train() para adaptar o período de busca ao volume real disponível,
        evitando queries desnecessárias para períodos sem dados.

        Returns:
            Horas de dados disponíveis (0.0 se nenhum dado)
        """
        from models import Neighbor
        oldest = session.exec(
            select(Neighbor).order_by(Neighbor.timestamp.asc()).limit(1)
        ).first()
        if not oldest:
            return 0.0
        delta = datetime.utcnow() - oldest.timestamp
        return delta.total_seconds() / 3600.0

    def extract_sliding_windows(self, session: Session, node_id: int,
                                history_hours: int = 24,
                                window_minutes: int = 60,
                                overlap_pct: float = 0.5,
                                end_time: Optional[datetime] = None,
                                reference_policy=None,
                                exclusion_stats=None) -> List[dict]:
        """
        Gera múltiplos vetores de features por nó usando janelas deslizantes.

        ``end_time`` fixa o fim do histórico (padrão: agora, legado). Com
        ``reference_policy``, janelas contaminadas (simulação histórica, linhas
        sintéticas, enlaces reversos de nós simulados) são excluídas e contadas em
        ``exclusion_stats`` (Counter) por motivo. Passo = max(1, int(janela·(1−overlap))).
        """
        step_minutes = max(1, int(window_minutes * (1 - overlap_pct)))
        now = end_time if end_time is not None else datetime.utcnow()
        history_start = now - timedelta(hours=history_hours)

        windows = []
        window_start = history_start
        while window_start + timedelta(minutes=window_minutes) <= now:
            window_end = window_start + timedelta(minutes=window_minutes)
            features = None
            if reference_policy is None:
                features = self.extract_for_window(session, node_id, window_start, window_end)
            elif reference_policy.node_contaminated([node_id], window_start, window_end):
                if exclusion_stats is not None:
                    exclusion_stats["simulation_overlap"] += 1
            else:
                prov: dict = {}
                features = self.extract_for_window(
                    session, node_id, window_start, window_end, provenance=prov
                )
                reason = (
                    reference_policy.check_window(node_id, window_start, window_end, prov)
                    if features else None
                )
                if reason:
                    if exclusion_stats is not None:
                        exclusion_stats[reason] += 1
                    features = None
            if features:
                windows.append(features)
            window_start += timedelta(minutes=step_minutes)

        logger.debug(
            f"Node {node_id}: {len(windows)} janelas de treino geradas "
            f"({history_hours}h, janela={window_minutes}min, step={step_minutes}min)"
        )
        return windows
