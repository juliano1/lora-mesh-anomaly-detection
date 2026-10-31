import json
import asyncio
import logging
import math
import re
import paho.mqtt.client as mqtt
import threading
from sqlmodel import Session, select
from sqlalchemy import text as sql_text, func as sa_func
from database import engine
from models import Node, Telemetry, Neighbor, MessageLog
from datetime import datetime, timedelta
from config import settings
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Union, Dict
from ai.anomaly_detector import AnomalyDetector

# Configuração de Logging
logger = logging.getLogger(__name__)

# Instância global do detector de anomalias (inicializada em start_mqtt_client)
anomaly_detector: AnomalyDetector = None

# Parâmetros do modelo log-distance de perda de percurso usados pelo firmware.
# A estimativa do backend (estimate_distance) usa os parâmetros de ai.propagation.
RSSI_REF_1M = -52   # RSSI a 1 m (calibrado para Heltec LoRa 32 V2; igual a RSSI_REF_1M do firmware)
PATH_LOSS_EXPONENT = 3.0  # Expoente n do firmware (PATH_LOSS_EXPONENT=3.0, indoor)
DISTANCE_REF_M = 1.0     # Distância de referência (m)

def estimate_distance(rssi: int, network=None) -> float:
    """
    Estima distância em metros a partir do RSSI.

    Se a rede tiver contexto de ambiente cadastrado, usa o perfil de
    propagação contextual (expoente n + perda por paredes) em vez do fallback fixo.
    Sem contexto, usa o padrão de ai.propagation (n=2.7, sem perda por paredes).

    Args:
        rssi:    RSSI em dBm.
        network: objeto Network (opcional) com o contexto de ambiente.
    """
    from ai.propagation import get_propagation_params, estimate_distance_contextual
    params = get_propagation_params(network)
    return estimate_distance_contextual(
        rssi,
        n=params["n"],
        wall_loss_db_total=params["wall_loss_db_total"],
        reference_rssi_1m=params["reference_rssi_1m"],
    )

# Limites de entrada do uplink. O beacon real tem < 1 KiB e no máximo
# MAX_NEIGHBORS_IN_BEACON (2) vizinhos; os limites deixam folga sem aceitar abuso.
MAX_MQTT_PAYLOAD_BYTES = 8192
MAX_NEIGHBORS_PER_PACKET = 16
_TOPIC_RE = re.compile(r"^lab/(\d{1,6})/(uplink|send_result)$")
_LORA_ADDR = r"^(0[xX])?[0-9a-fA-F]{1,4}$"


def extract_network_id(topic: str) -> Optional[int]:
    """Extrai network_id do tópico MQTT; só aceita lab/<id>/uplink e lab/<id>/send_result.

    Ex: 'lab/3/uplink' → 3
        'lab/1/send_result' → 1
    """
    match = _TOPIC_RE.match(topic)
    return int(match.group(1)) if match else None


def _reject_non_finite(constant: str):
    """json.loads aceita NaN/Infinity por padrão; telemetria não finita é recusada."""
    raise ValueError(f"constante JSON não finita: {constant}")


def parse_uplink_json(payload_str: str) -> dict:
    """Decodifica o uplink: objeto JSON sem NaN/Infinity e com lista de vizinhos limitada."""
    data = json.loads(payload_str, parse_constant=_reject_non_finite)
    if not isinstance(data, dict):
        raise ValueError("uplink deve ser um objeto JSON")
    neighbors = data.get("n")
    if neighbors is not None and (not isinstance(neighbors, list) or len(neighbors) > MAX_NEIGHBORS_PER_PACKET):
        raise ValueError("lista de vizinhos inválida ou acima do limite")
    return data


def _finite_number(value) -> Optional[float]:
    """Valor numérico finito do payload interno (temp/hum) ou None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def build_lora_address_map(session: Session, network_id: Optional[int] = None) -> tuple[dict, dict]:
    """
    Constrói mapeamento bidirecional a partir do banco de dados.
    Quando network_id fornecido, inclui apenas nós da rede especificada.
    As chaves de lora_to_node ficam em minúsculas, para comparação sem diferenciar maiúsculas.

    Returns:
        tuple: (lora_to_node, node_to_lora) dicionários de mapeamento
    """
    lora_to_node = {}
    node_to_lora = {}

    query = select(Node).where(Node.lora_address.is_not(None))
    if network_id is not None:
        query = query.where(Node.network_id == network_id)
    nodes = session.exec(query).all()

    for node in nodes:
        if node.lora_address:
            lora_to_node[node.lora_address.lower()] = node.node_id
            node_to_lora[node.node_id] = node.lora_address  # preserva case original

    logger.debug(f"Mapeamento LoRa construído: {len(lora_to_node)} nós mapeados")
    return lora_to_node, node_to_lora

# Cores fixas por nó, usadas nas arestas que o nó reporta
NODE_COLORS = [
    '#dc2626',  # Node 1 (Gateway) — Vermelho
    '#eab308',  # Node 2 — Amarelo
    '#3b82f6',  # Node 3 — Azul
    '#22c55e',  # Node 4 — Verde
    '#a855f7',  # Node 5 — Roxo
    '#f97316',  # Node 6 — Laranja
    '#06b6d4',  # Node 7 — Ciano
    '#ec4899',  # Node 8 — Rosa
    '#14b8a6',  # Node 9 — Teal
    '#f43f5e',  # Node 10 — Rose
    '#8b5cf6',  # Node 11 — Violet
    '#84cc16',  # Node 12 — Lime
]

def get_node_color(node_id: int) -> str:
    """Retorna cor fixa do nó pelo seu ID."""
    if node_id is None or node_id < 1:
        return '#999999'
    return NODE_COLORS[(node_id - 1) % len(NODE_COLORS)]

def get_color_by_rssi(rssi: Optional[int]) -> str:
    """
    Retorna cor hex baseada na qualidade do sinal RSSI.
    
    Faixas:
    - Verde (#22c55e): RSSI > -70 dBm (excelente)
    - Amarelo (#eab308): -70 a -85 dBm (bom)
    - Laranja (#f97316): -85 a -95 dBm (fraco)
    - Vermelho (#ef4444): < -95 dBm (muito fraco)
    - Cinza (#999999): sem dados
    
    Args:
        rssi: RSSI em dBm (negativo)
    
    Returns:
        str: Código de cor hexadecimal
    """
    if rssi is None or rssi == 0:
        return "#999999"  # Cinza (sem dados)
    elif rssi > -70:
        return "#22c55e"  # Verde (excelente)
    elif rssi > -85:
        return "#eab308"  # Amarelo (bom)
    elif rssi > -95:
        return "#f97316"  # Laranja (fraco)
    else:
        return "#ef4444"  # Vermelho (muito fraco)

def ensure_gateway_exists(session: Session):
    """Garante que o Gateway (node_id=1) existe no banco de dados"""
    gateway = session.get(Node, 1)
    if not gateway:
        gateway = Node(
            node_id=1,
            name="Gateway 1",
            status="online",
            last_seen=datetime.utcnow()
        )
        session.add(gateway)
        session.commit()
        logger.info("✅ Gateway (node_id=1) criado no banco de dados")

# Modelo de Validação de Entrada (DTO)
class NeighborData(BaseModel):
    """Modelo para cada vizinho no array 'n'"""
    a: Optional[str] = Field(default=None, pattern=_LORA_ADDR, description="Endereço LoRa do vizinho em hex (ex: '0001')")
    r: Optional[int] = Field(default=None, ge=-200, le=50, description="RSSI para este vizinho em dBm")
    s: Optional[float] = Field(default=None, ge=-50, le=50, allow_inf_nan=False, description="SNR para este vizinho em dB")
    p: bool = Field(default=False, description="True se for o parent (rota ativa)")
    h: Optional[int] = Field(default=None, ge=0, le=255, description="Número de saltos (hops)")
    d: Optional[float] = Field(default=None, ge=0, le=1_000_000, allow_inf_nan=False, description="Distância estimada em metros")
    l: Optional[int] = Field(default=None, ge=0, le=10, description="Índice de qualidade do enlace (0–10), calculado pelo firmware a partir de RSSI/SNR; não é PDR medido")
    
    def is_valid(self) -> bool:
        """Verifica se o vizinho tem os campos mínimos (endereço e RSSI)."""
        return self.a is not None and self.r is not None

class LoRaPacket(BaseModel):
    # Relato de múltiplos enlaces
    i: Optional[str] = Field(default=None, max_length=16, description="ID do nó (ex: 'N2')")
    a: Optional[str] = Field(default=None, pattern=_LORA_ADDR, description="Endereço LoRa local em hex (ex: '0002')")
    n: Optional[list[NeighborData]] = Field(default=None, max_length=MAX_NEIGHBORS_PER_PACKET, description="Array de vizinhos diretos")
    u: Optional[int] = Field(default=None, ge=0, description="Uptime em minutos")
    q: Optional[int] = Field(default=None, ge=0, description="Número de sequência")
    b: Optional[int] = Field(default=None, ge=-2, le=100, description="Nível de bateria 0-100%, -1=USB, -2=leitura inválida")
    nearest: Optional[str] = Field(default=None, max_length=16, description="Endereço LoRa do vizinho mais próximo")
    sim: Optional[str] = Field(default=None, pattern=r"^A[1-7]$", description="Código da anomalia simulada (ex.: 'A2')")

    # Campos antigos (compatibilidade)
    sender: Optional[int] = Field(default=None, description="ID do nó remetente")
    node_id: Optional[int] = Field(default=None, description="ID do nó (alternativa)")
    parent_id: Optional[int] = Field(default=None, description="ID do nó que encaminhou a mensagem")
    rssi: Optional[int] = Field(default=0, ge=-140, le=0)
    snr: Optional[float] = Field(default=0.0, ge=-50, le=50, allow_inf_nan=False)
    packet_id: Optional[int] = None
    payload: Union[Dict, str, None] = None # Payload flexível
    
    # Campos do Gateway
    id: Optional[str] = Field(default=None, description="ID string do nó")
    seq: Optional[int] = Field(default=None, description="Número de sequência")
    uptime: Optional[int] = Field(default=None, description="Uptime em segundos")
    gw_id: Optional[str] = Field(default=None, description="ID do Gateway")
    gw_rssi: Optional[int] = Field(default=None, ge=-200, le=50, description="RSSI no Gateway")
    gw_snr: Optional[float] = Field(default=None, ge=-50, le=50, allow_inf_nan=False, description="SNR no Gateway")
    gw_toa: Optional[int] = Field(default=None, description="Tempo no ar (time on air)")
    gw_timestamp: Optional[int] = Field(default=None, description="Timestamp do Gateway")
    
    def get_node_id(self) -> Optional[int]:
        """Retorna o node_id, extraindo de 'id' ou 'i' se necessário"""
        if self.node_id is not None:
            return self.node_id
        if self.sender is not None:
            return self.sender
        # Extrair de 'i' (formato "N2" -> 2)
        if self.i and self.i.startswith("N"):
            try:
                return int(self.i[1:])
            except ValueError:
                pass
        # Extrair de 'id' (formato "NODE_2" -> 2)
        if self.id and self.id.startswith("NODE_"):
            try:
                return int(self.id.split("_")[1])
            except (ValueError, IndexError):
                pass
        return None

def build_topology(session: Session, network_id: Optional[int] = None) -> dict:
    """
    Constrói a topologia da rede (nós e arestas no formato de grafo usado por GET /topology).
    Usa a tabela neighbors para criar múltiplos edges com cores.

    Se network_id fornecido: filtra nós e links daquela rede.
    Se None: retorna todos os nós (compatibilidade).
    """
    nodes_data = []
    edges_data = []

    # Carregar objeto Network para perfil contextual de distância
    _topology_network = None
    if network_id is not None:
        from models import Network as _Network
        _topology_network = session.get(_Network, network_id)

    # Buscar nós (filtrado por rede se network_id fornecido)
    node_query = select(Node)
    if network_id is not None:
        node_query = node_query.where(Node.network_id == network_id)
    all_nodes = session.exec(node_query).all()

    # Buscar timestamp da última leitura de bateria por nó
    battery_timestamps = {}
    try:
        bat_rows = session.execute(sql_text(
            "SELECT node_id, MAX(timestamp) as last_bat FROM telemetry "
            "WHERE battery_level IS NOT NULL GROUP BY node_id"
        )).all()
        battery_timestamps = {row[0]: row[1].isoformat() + "Z" if row[1] else None for row in bat_rows}
    except Exception as e:
        logger.warning(f"Erro ao buscar timestamps de bateria: {e}")

    # Construir mapeamento LoRa <-> Node ID (escopo da rede)
    lora_to_node, node_to_lora = build_lora_address_map(session, network_id=network_id)
    
    # Threshold para status online — calculado dinamicamente a partir do last_seen.
    # Não depende do campo node.status salvo no banco (que pode estar desatualizado
    # se o NodeMonitor tiver um atraso ou se o backend tiver reiniciado recentemente).
    # Usa o mesmo valor de NODE_TIMEOUT_SECONDS para manter consistência.
    online_threshold = datetime.utcnow() - timedelta(seconds=settings.NODE_TIMEOUT_SECONDS)

    def _node_status(node) -> str:
        if node is None:
            return "online"
        if node.last_seen is None:
            return "offline"
        return "online" if node.last_seen > online_threshold else "offline"

    node_status_by_id = {node.node_id: _node_status(node) for node in all_nodes}
    offline_node_ids = {nid for nid, status in node_status_by_id.items() if status == "offline"}

    # Conjunto de IDs de gateway para uso posterior (is_parent check)
    gateway_node_ids = {node.node_id for node in all_nodes if node.is_gateway}

    # Adicionar todos os nós usando is_gateway do banco (suporte multi-rede)
    for node in all_nodes:
        lora_addr = node.lora_address or "0x?"
        node_type = "gateway" if node.is_gateway else "sensor"
        # Gateways usam o nome cadastrado; sensores usam label curto Nx
        label = (node.name or f"N{node.node_id}") if node.is_gateway else f"N{node.node_id}"
        nodes_data.append({
            "data": {
                "id": str(node.node_id),
                "label": label,
                "type": node_type,
                "battery": node.battery_level,
                "battery_updated_at": battery_timestamps.get(node.node_id),
                "status": node_status_by_id.get(node.node_id, _node_status(node)),
                "lora_address": lora_addr,
                "node_color": get_node_color(node.node_id),
                "led_state": node.led_state or "off",
                "sim_mode": str(node.sim_mode).lower(),
                "sim_anomaly": node.sim_anomaly or "",
                "sim_started_at": node.sim_started_at.isoformat() if node.sim_started_at else None,
            }
        })
    
    # Buscar vizinhos recentes (últimos 30 minutos) apenas dos nós desta rede.
    # O filtro por node_id impede que vizinhos de outra rede criem edges cruzados.
    time_threshold = datetime.utcnow() - timedelta(minutes=30)
    node_ids_in_network = {n.node_id for n in all_nodes}
    neighbor_query = (
        select(Neighbor)
        .where(Neighbor.timestamp > time_threshold)
        .order_by(Neighbor.timestamp.desc())
    )
    if node_ids_in_network:
        neighbor_query = neighbor_query.where(Neighbor.node_id.in_(node_ids_in_network))
    recent_neighbors = session.exec(neighbor_query).all()

    logger.debug(f"Encontrados {len(recent_neighbors)} vizinhos recentes (últimos 30 min)")
    
    # Arestas direcionais: uma por (nó que reporta → vizinho observado), com a cor fixa
    # do nó que reporta. Se dois nós se reportam, o par gera duas arestas, uma em cada
    # sentido; não deduplicar.
    seen_edges = set()
    
    for neighbor in recent_neighbors:
        
        # Converter endereço LoRa do vizinho para node_id (lowercase para match)
        neighbor_node_id = lora_to_node.get(neighbor.neighbor_address.lower())
        
        if neighbor_node_id is None:
            logger.warning(f"Vizinho com endereço desconhecido: {neighbor.neighbor_address}")
            continue

        # Remover links onde a ORIGEM está offline (sem dados recentes para reportar).
        # Links cujo DESTINO está offline são preservados: o destino já sai com
        # status offline e a aresta mostra a última topologia de roteamento conhecida.
        # Remover também por destino offline faria, por exemplo, o nó 3 (online) perder
        # a aresta 3→4 quando o parent 4 está offline e ficar sem parent visível.
        if neighbor.node_id in offline_node_ids:
            continue
        
        # Edge direcional: source (quem reporta) → target (vizinho observado)
        edge_key = f"{neighbor.node_id}-{neighbor_node_id}"
        
        if edge_key not in seen_edges:
            # Cor fixa do nó que reporta (fonte)
            node_color = get_node_color(neighbor.node_id)
            
            # Gateway é raiz da malha — nunca deve gerar parent edges.
            # O firmware marca um vizinho como parent na routing table do gateway,
            # mas isso não faz sentido topológico (gateway é o destino final).
            is_parent = neighbor.is_parent and neighbor.node_id not in gateway_node_ids

            # Determinar estilo de linha: sólido (parent) vs pontilhado (peer)
            line_style = "solid" if is_parent else "dashed"
            
            # Estilo pelo índice de qualidade do enlace (link_quality, 0–10), se houver:
            # l=0–4 → dotted, l=5–7 → dashed, l=8–10 → solid.
            # NOTA: Aplicar pdr_line_style SOMENTE para edges parent (rotas ativas).
            # Peer edges devem SEMPRE permanecer dashed para diferenciar de rotas.
            pdr_line_style = None
            if is_parent and neighbor.link_quality is not None:
                if neighbor.link_quality <= 4:
                    pdr_line_style = "dotted"
                elif neighbor.link_quality <= 7:
                    pdr_line_style = "dashed"
                else:
                    pdr_line_style = "solid"
            
            # Label compacto: RSSI + distância se disponível
            if neighbor.distance_m and neighbor.distance_m > 0:
                label = f"{neighbor.rssi}dBm ~{neighbor.distance_m:.0f}m"
            else:
                label = f"{neighbor.rssi}dBm"
            
            edges_data.append({
                "data": {
                    "id": edge_key,
                    "source": str(neighbor.node_id),
                    "target": str(neighbor_node_id),
                    "rssi": neighbor.rssi,
                    "snr": neighbor.snr,
                    "is_parent": is_parent,
                    "label": label,
                    "color": node_color,
                    "line_style": line_style,
                    "pdr_line_style": pdr_line_style,
                    "link_quality": neighbor.link_quality,   # índice de qualidade do enlace 0-10
                    "source_node_id": neighbor.node_id,
                    "hops": neighbor.hops,
                    "distance_m": neighbor.distance_m
                }
            })
            seen_edges.add(edge_key)
            
            logger.debug(f"Edge direcional: {neighbor.node_id} → {neighbor_node_id} "
                        f"(RSSI: {neighbor.rssi}, parent: {neighbor.is_parent}, "
                        f"cor: {node_color}, estilo: {line_style})")
    
    # ── Fallback: edges a partir de Telemetry.parent_id ──────────────
    # Quando nós sensores não enviam vizinhos (ex: após reboot, n:[]),
    # usamos o parent_id da última telemetria para manter os links visíveis.
    # Também usamos os vizinhos reportados pelo gateway para enriquecer:
    # se GW vê N2 e N5, e N5 reportou parent=N2 na telemetria, criamos 5→2.
    
    nodes_with_edges = set()
    for ek in seen_edges:
        src, tgt = ek.split("-")
        nodes_with_edges.add(int(src))
    
    # Nós online que NÃO reportaram vizinhos recentes (exclui gateways)
    online_nodes = {
        node_id for node_id, status in node_status_by_id.items()
        if status == "online" and node_id not in gateway_node_ids
    }
    missing_nodes = online_nodes - nodes_with_edges
    
    if missing_nodes:
        logger.info(f"🔗 Fallback: nós sem vizinhos recentes: {missing_nodes}")
        
        # Estratégia 1: última telemetria com parent_id (sem limite de tempo,
        # pois o nó está online — detectado pelo gateway)
        for nid in list(missing_nodes):
            last_tel = session.exec(
                select(Telemetry)
                .where(Telemetry.node_id == nid)
                .where(Telemetry.parent_id != None)
                .order_by(Telemetry.timestamp.desc())
            ).first()
            
            if last_tel and last_tel.parent_id:
                parent_id = last_tel.parent_id

                # Fallback não recria edge se a ORIGEM (nid) está offline.
                # Target offline é permitido — edge mostra último parent conhecido.
                if nid in offline_node_ids:
                    continue

                edge_key = f"{nid}-{parent_id}"
                
                if edge_key not in seen_edges:
                    node_color = get_node_color(nid)
                    rssi_val = last_tel.rssi or 0
                    # Usa perfil contextual da rede quando disponível
                    distance = estimate_distance(rssi_val, network=_topology_network) if rssi_val and rssi_val < 0 else None
                    
                    if distance and distance > 0:
                        label = f"{rssi_val}dBm ~{distance:.0f}m"
                    else:
                        label = f"{rssi_val}dBm" if rssi_val else "rota"
                    
                    edges_data.append({
                        "data": {
                            "id": edge_key,
                            "source": str(nid),
                            "target": str(parent_id),
                            "rssi": rssi_val,
                            "snr": last_tel.snr,
                            "is_parent": True,
                            "label": label,
                            "color": node_color,
                            "line_style": "solid",
                            "pdr_line_style": None,
                            "link_quality": None,
                            "source_node_id": nid,
                            "hops": None,
                            "distance_m": distance
                        }
                    })
                    seen_edges.add(edge_key)
                    logger.info(f"🔗 Fallback edge: {nid} → {parent_id} (parent via telemetria, RSSI: {rssi_val})")
        
        # Estratégia 2: vizinhos históricos recentes (limite 2h para evitar is_parent
        # stale de topologias antigas causarem ciclos na reconstrução de caminho)
        hist_time_threshold = datetime.utcnow() - timedelta(hours=2)
        still_missing = missing_nodes - {int(ek.split("-")[0]) for ek in seen_edges if int(ek.split("-")[0]) in missing_nodes}
        
        for nid in still_missing:
            hist_neighbors = session.exec(
                select(Neighbor)
                .where(Neighbor.node_id == nid)
                .where(Neighbor.timestamp > hist_time_threshold)
                .order_by(Neighbor.timestamp.desc())
            ).all()
            
            hist_seen = set()
            for hn in hist_neighbors:
                neighbor_node_id = lora_to_node.get(hn.neighbor_address.lower())
                if neighbor_node_id is None:
                    continue

                # Fallback histórico não cria edge se a ORIGEM está offline.
                # Target offline é permitido para preservar topologia conhecida.
                if hn.node_id in offline_node_ids:
                    continue

                edge_key = f"{hn.node_id}-{neighbor_node_id}"
                if edge_key not in seen_edges and edge_key not in hist_seen:
                    node_color = get_node_color(hn.node_id)
                    hist_is_parent = hn.is_parent and hn.node_id not in gateway_node_ids
                    line_style = "solid" if hist_is_parent else "dashed"
                    
                    if hn.distance_m and hn.distance_m > 0:
                        label = f"{hn.rssi}dBm ~{hn.distance_m:.0f}m"
                    else:
                        label = f"{hn.rssi}dBm"
                    
                    edges_data.append({
                        "data": {
                            "id": edge_key,
                            "source": str(hn.node_id),
                            "target": str(neighbor_node_id),
                            "rssi": hn.rssi,
                            "snr": hn.snr,
                            "is_parent": hist_is_parent,
                            "label": label,
                            "color": node_color,
                            "line_style": line_style,
                            "pdr_line_style": None,
                            "link_quality": hn.link_quality,
                            "source_node_id": hn.node_id,
                            "hops": hn.hops,
                            "distance_m": hn.distance_m
                        }
                    })
                    seen_edges.add(edge_key)
                    hist_seen.add(edge_key)
                    logger.info(f"🔗 Fallback histórico: {hn.node_id} → {neighbor_node_id} "
                               f"(vizinho de {(datetime.utcnow() - hn.timestamp).seconds}s atrás)")
    
    # ── Fim Fallback ─────────────────────────────────────────────────

    # ── Detecção e quebra de ciclos em parent edges ───────────────────────────
    # Constrói mapa: source -> (idx_em_edges_data, target) para parent edges.
    # Apenas o primeiro parent por nó é registrado (comportamento esperado do protocolo).
    parent_idx_map = {}
    for i, edge in enumerate(edges_data):
        if edge["data"]["is_parent"]:
            src = edge["data"]["source"]
            if src not in parent_idx_map:
                parent_idx_map[src] = (i, edge["data"]["target"])

    broken_sources = set()
    for start_src in list(parent_idx_map.keys()):
        if start_src in broken_sources:
            continue

        path = [start_src]
        visited_order = {start_src: 0}
        current = start_src

        for _ in range(30):
            if current not in parent_idx_map:
                break
            _, nxt = parent_idx_map[current]
            if nxt == "1":  # chegou ao gateway — sem ciclo
                break
            if nxt in visited_order:
                # Ciclo encontrado — determinar nós envolvidos no loop
                cycle_start_idx = visited_order[nxt]
                cycle_nodes = path[cycle_start_idx:]

                # Selecionar a edge mais fraca no ciclo para quebrar.
                # lq=None ocorre apenas em edges de fallback (telemetria histórica
                # sem link_quality). Tratá-las com -1 (abaixo de qualquer lq real 0-10)
                # é intencional: preferimos quebrar links incertos antes de links com
                # qualidade medida, mesmo que ruim. Isso é conservador e evita remover
                # uma edge com dados reais quando há uma alternativa sem dados.
                candidates = []
                for cn in cycle_nodes:
                    if cn in parent_idx_map:
                        idx, tgt = parent_idx_map[cn]
                        lq  = edges_data[idx]["data"].get("link_quality")
                        rssi = edges_data[idx]["data"].get("rssi")
                        candidates.append((
                            lq   if lq   is not None else -1,
                            rssi if rssi is not None else -999,
                            cn, idx
                        ))

                if candidates:
                    candidates.sort()  # menor lq/rssi = link mais fraco
                    _, _, break_src, break_idx = candidates[0]
                    _, break_tgt = parent_idx_map[break_src]

                    edges_data[break_idx]["data"]["is_parent"]     = False
                    edges_data[break_idx]["data"]["line_style"]    = "dashed"
                    edges_data[break_idx]["data"]["pdr_line_style"] = None

                    # Remover do mapa para que travessias posteriores (ex.: nós
                    # fora do ciclo que chegam a break_src via sua própria cadeia)
                    # encontrem "sem parent" e parem, evitando quebras duplas.
                    del parent_idx_map[break_src]

                    logger.warning(
                        f"🔄 Ciclo detectado e quebrado: edge parent {break_src} → {break_tgt} "
                        f"convertida em peer "
                        f"(lq={edges_data[break_idx]['data'].get('link_quality')}, "
                        f"rssi={edges_data[break_idx]['data'].get('rssi')})"
                    )
                    broken_sources.update(cycle_nodes)
                break

            path.append(nxt)
            visited_order[nxt] = len(path) - 1
            current = nxt
    # ── Fim detecção de ciclos ────────────────────────────────────────────────

    # Filtrar edges que referenciam nós inexistentes no nodes_data.
    # Isso pode ocorrer quando há nós duplicados por endereço LoRa (ex: após
    # edição de rede que cria novos registros com os mesmos endereços dos nós
    # ativos que ainda têm network_id=None). O lora_to_node pode mapear para
    # node_ids fora do conjunto de nodes_data, gerando arestas sem nó de origem
    # ou destino; elas são descartadas.
    valid_node_ids = {nd["data"]["id"] for nd in nodes_data}
    edges_data = [
        e for e in edges_data
        if e["data"]["source"] in valid_node_ids and e["data"]["target"] in valid_node_ids
    ]

    # Calcular vizinho mais próximo de cada nó (a partir dos edges)
    nearest_map = {}  # node_id -> {"address": "0x...", "distance_m": float, "node_id": int}
    for edge in edges_data:
        src = int(edge["data"]["source"])
        dist = edge["data"].get("distance_m")
        if dist and dist > 0:
            if src not in nearest_map or dist < nearest_map[src]["distance_m"]:
                nearest_map[src] = {
                    "node_id": int(edge["data"]["target"]),
                    "distance_m": round(dist, 1)
                }
    
    # Injetar nearest nos nodes_data
    for node_data in nodes_data:
        nid = int(node_data["data"]["id"])
        if nid in nearest_map:
            node_data["data"]["nearest"] = nearest_map[nid]
    
    # Calcular estatísticas
    active_nodes = len([n for n in all_nodes if n.status == "online"])
    gw_count = len(gateway_node_ids)
    stats = {
        "gateway_count": gw_count,
        "sensor_count": len(all_nodes) - gw_count,
        "active_connections": len(edges_data)
    }
    
    logger.info(f"Topologia construída: {len(nodes_data)} nós, {len(edges_data)} edges")
    
    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "stats": stats,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

def process_message_sync(payload_str: str, network_id: Optional[int] = None):
    """
    Recebe o JSON cru do MQTT e grava no banco. Síncrona: roda na thread do
    cliente MQTT, chamada por on_message.
    Processa o array de vizinhos do beacon.
    network_id extraído do tópico para validação e associação.
    """
    try:
        logger.debug(f"🔍 RAW PAYLOAD: {payload_str}")

        # 1. Validação e Parsing com Pydantic
        try:
            raw_data = parse_uplink_json(payload_str)

            # Pré-processamento: limpa listas de vizinhos truncadas antes da validação.
            # Sensores com firmware antigo enviam beacons truncados, com entradas
            # incompletas como {"a": null} ou {"a": "0003"} (sem r/s).
            if "n" in raw_data and isinstance(raw_data["n"], list):
                valid_neighbors = []
                for nb in raw_data["n"]:
                    if isinstance(nb, dict) and nb.get("a") and nb.get("r") is not None:
                        valid_neighbors.append(nb)
                    else:
                        logger.debug(f"Vizinho incompleto filtrado (truncamento): {nb}")
                if len(valid_neighbors) < len(raw_data["n"]):
                    logger.warning(f"⚠️ {len(raw_data['n']) - len(valid_neighbors)} vizinho(s) "
                                  f"incompleto(s) removido(s) (beacon truncado do sensor com firmware antigo)")
                raw_data["n"] = valid_neighbors if valid_neighbors else None
            
            packet = LoRaPacket(**raw_data)
        except ValidationError as e:
            logger.warning(f"Dados inválidos (Schema incorreto): {e.error_count()} erro(s)")
            logger.debug(f"Payload recebido: {payload_str}")
            return
        except ValueError as e:  # inclui json.JSONDecodeError
            logger.warning(f"Uplink recusado ({len(payload_str)} bytes): {e}")
            return

        # Detectar beacons truncados (estouro do StaticJsonDocument no gateway)
        if "raw_data" in raw_data:
            logger.warning(f"⚠️ BEACON TRUNCADO detectado! O Gateway não conseguiu parsear "
                          f"o beacon completo. raw_data presente ({len(str(raw_data.get('raw_data', '')))} chars). "
                          f"Verifique StaticJsonDocument no firmware do GW.")

        # Extração de dados
        node_id = packet.get_node_id()
        if node_id is None:
            logger.warning(f"Não foi possível extrair node_id do pacote: {payload_str[:200]!r}")
            return
        
        # Extrair endereço LoRa local do nó (campo 'a')
        lora_address = None
        if packet.a:
            # Normalizar formato: adicionar "0x" se não tiver, e lowercase
            addr = packet.a.lower()
            lora_address = addr if addr.startswith("0x") else f"0x{addr}"
        
        # Extrair uptime e sequence (campos 'u' e 'q')
        uptime = packet.u
        sequence = packet.q if packet.q is not None else packet.seq
            
        # Usar gw_rssi/gw_snr se disponível, senão rssi/snr direto
        # (para mensagens antigas sem vizinhos)
        rssi = packet.gw_rssi if packet.gw_rssi is not None else packet.rssi
        snr = packet.gw_snr if packet.gw_snr is not None else packet.snr
        parent_id_old = packet.parent_id  # parent_id antigo (int)
        
        inner_payload = packet.payload if isinstance(packet.payload, dict) else {}
        temp = _finite_number(inner_payload.get("temp"))
        hum = _finite_number(inner_payload.get("hum"))
        bat = next((int(v) for k in ("bat", "bat_lvl", "battery")
                    if (v := _finite_number(inner_payload.get(k))) is not None and -2 <= v <= 100), None)
        
        # Bateria pelo campo 'b' do beacon (enviado por todos os nós)
        # b >= 0: percentual real; b == -1: alimentação USB; b == -2: leitura inválida/sem sensor
        if packet.b is not None:
            if packet.b >= 0:
                bat = packet.b
                logger.debug(f"Node {packet.get_node_id()}: battery={bat}% (LiPo)")
            elif packet.b == -1:
                bat = -1  # marcador de alimentação USB
                logger.debug(f"Node {packet.get_node_id()}: USB powered (no battery)")
            else:
                bat = -2  # Sensor inválido — propagado para DB para atualizar timestamp
                logger.debug(f"Node {packet.get_node_id()}: battery sensor invalid (b={packet.b})")

        # 2. Interação com Banco de Dados
        with Session(engine) as session:
            # Validar rede se network_id fornecido
            network = None  # Objeto Network para perfil de propagação
            if network_id is not None:
                from models import Network
                network = session.get(Network, network_id)
                if not network or not network.active:
                    logger.warning(f"Mensagem descartada: rede {network_id} não existe ou está inativa")
                    return
                # Com rede no tópico, a identidade vem só do endereço LoRa cadastrado;
                # o campo 'i' do pacote não basta (evita gravar em nó de outra rede).
                if not lora_address:
                    logger.warning(f"Mensagem descartada: uplink da rede {network_id} sem endereço LoRa")
                    return
                if lora_address:
                    node_check = session.exec(
                        select(Node).where(
                            sa_func.lower(Node.lora_address) == lora_address,
                            Node.network_id == network_id
                        )
                    ).first()
                    if not node_check:
                        logger.warning(
                            f"Mensagem descartada: dispositivo {lora_address} não pertence à rede {network_id}"
                        )
                        return

            # Garantir que o Gateway existe
            ensure_gateway_exists(session)

            # A. Atualizar ou Criar Nó (Upsert)
            # Quando network_id está disponível, usa node_check (já validado acima pelo
            # lora_address + network_id) para evitar colapso de identidade via campo 'i'
            # do gateway (que reporta i="N1" para todas as mensagens que ele relayou).
            if network_id is not None and lora_address:
                node = node_check
                node_id = node.node_id
            else:
                node = session.get(Node, node_id)
                if not node:
                    node = Node(node_id=node_id, name=f"Node {node_id}")

            node.last_seen = datetime.utcnow()
            node.status = "online"
            if bat is not None:
                node.battery_level = bat

            # Sincronizar o estado de simulação pelo campo "sim" do beacon:
            # o beacon é a fonte de verdade, pois o firmware controla o próprio estado.
            sim_field = packet.sim
            # data_origin baseado no campo "sim"
            origin = _get_data_origin(sim_field)

            session.add(node)

            # O recebimento de qualquer uplink implica gateway operacional.
            # Atualiza heartbeat do gateway da rede para evitar falso offline.
            if network_id is not None and not node.is_gateway:
                gw = session.exec(
                    select(Node).where(
                        Node.network_id == network_id,
                        Node.is_gateway == True  # noqa: E712
                    )
                ).first()
                if gw:
                    gw.last_seen = datetime.utcnow()
                    gw.status = "online"
                    session.add(gw)
            elif network_id is None and node_id != 1:
                gateway = session.get(Node, 1)
                if gateway:
                    gateway.last_seen = datetime.utcnow()
                    gateway.status = "online"
                    session.add(gateway)

            session.commit()
            session.refresh(node)

            # B. Processar array de vizinhos
            # Obter mapa de endereços LoRa para converter parent hex -> node_id (escopo da rede)
            lora_to_node, _ = build_lora_address_map(session, network_id=network_id)
            
            parent_node_id = None  # parent_id como inteiro (node_id)
            
            if packet.n and len(packet.n) > 0:
                logger.info(f"Processando {len(packet.n)} vizinhos de Node {node_id}")
                
                # Retenção de vizinhos: manter 90 dias de histórico para
                # exportação e análise offline. A topologia em
                # tempo real já filtra por timestamp na query, então manter
                # registros antigos não afeta o desempenho da topologia.
                retention_threshold = datetime.utcnow() - timedelta(days=90)
                old_neighbors = session.exec(
                    select(Neighbor)
                    .where(Neighbor.node_id == node_id)
                    .where(Neighbor.timestamp < retention_threshold)
                ).all()

                for old_neighbor in old_neighbors:
                    session.delete(old_neighbor)

                # Processar cada vizinho
                for neighbor_data in packet.n:
                    # Ignorar vizinhos incompletos (de beacons truncados)
                    if not neighbor_data.is_valid():
                        logger.debug(f"Skipping invalid neighbor: {neighbor_data}")
                        continue
                    
                    # Normalizar endereço do vizinho: adicionar "0x" e lowercase
                    addr = neighbor_data.a.lower()
                    neighbor_addr = addr if addr.startswith("0x") else f"0x{addr}"
                    
                    # Verificar se o vizinho existe no mapa LoRa->Node
                    neighbor_node_id = lora_to_node.get(neighbor_addr)
                    
                    # Se o vizinho tem um node_id conhecido, garantir que existe na tabela Node
                    if neighbor_node_id:
                        neighbor_node = session.get(Node, neighbor_node_id)
                        if not neighbor_node:
                            # Criar nó novo - status "unknown" até que ele envie telemetria diretamente
                            neighbor_node = Node(
                                node_id=neighbor_node_id,
                                name=f"Node {neighbor_node_id}",
                                lora_address=neighbor_addr,
                                status="unknown",
                                last_seen=datetime.utcnow()
                            )
                            session.add(neighbor_node)
                            session.commit()
                            logger.info(f"✅ Node {neighbor_node_id} criado automaticamente (descoberto como vizinho)")
                        else:
                            # NÃO atualizar last_seen do vizinho aqui!
                            # O last_seen só deve ser atualizado quando o próprio nó envia telemetria.
                            # Outro nó pode ter o vizinho em cache (routing table) mesmo após ele desligar.
                            pass
                    
                    # Distância: usa 'd' do firmware se houver; senão, estima pelo RSSI
                    # Passa perfil da rede para usar n e wall_loss contextuais
                    distance = neighbor_data.d
                    if distance is None and neighbor_data.r is not None:
                        distance = estimate_distance(neighbor_data.r, network=network)
                    
                    # Criar ou atualizar registro de vizinho
                    neighbor = Neighbor(
                        node_id=node_id,
                        neighbor_address=neighbor_addr,
                        rssi=neighbor_data.r,
                        snr=neighbor_data.s,
                        is_parent=neighbor_data.p,
                        hops=neighbor_data.h,
                        distance_m=distance,            # do beacon ou estimada
                        link_quality=neighbor_data.l,   # índice de qualidade do enlace 0-10
                        timestamp=datetime.utcnow(),
                        data_origin=origin,
                    )
                    
                    session.add(neighbor)
                    
                    logger.debug(f"Vizinho {neighbor_addr}: RSSI {neighbor_data.r}, "
                               f"SNR {neighbor_data.s}, parent={neighbor_data.p}")
                    
                    # Se for o parent, converter para node_id e salvar
                    if neighbor_data.p:
                        parent_node_id = neighbor_node_id  # Já obtido acima
                        if parent_node_id is None:
                            logger.warning(f"Parent LoRa {neighbor_addr} sem node_id conhecido")
                        rssi = neighbor_data.r  # Usar RSSI do parent
                        snr = neighbor_data.s   # Usar SNR do parent
                
                session.commit()
                logger.info(f"✅ {len(packet.n)} vizinhos salvos para Node {node_id}")
            
            # Compatibilidade: se não há array de vizinhos, usar parent_id antigo
            elif parent_id_old is not None:
                parent_node_id = parent_id_old  # Já é inteiro

            # C. Salvar Histórico (Telemetry)
            telemetry = Telemetry(
                node_id=node_id,
                parent_id=parent_node_id,  # Sempre inteiro (node_id)
                rssi=rssi or 0,
                snr=snr or 0.0,
                packet_id=packet.packet_id,
                uptime=uptime,
                sequence=sequence,
                battery_level=bat,
                temperature=temp,
                humidity=hum,
                payload_raw=payload_str,
                data_origin=origin,
            )
            session.add(telemetry)

            # Incrementar telemetry_count em tempo real
            if origin == "synthetic" and node.current_sim_run_id:
                _increment_telemetry_count(node.current_sim_run_id, session)

            # Sincronizar estado de simulação via campo "sim"
            if sim_field and not node.sim_mode:
                if _should_activate_sim(node, sim_field):
                    node.sim_mode = True
                    node.sim_anomaly = sim_field
                    node.sim_started_at = datetime.utcnow()
                    node.sim_ended_at = None
                    logger.info(f"SIM detected in beacon from N{node_id}: {sim_field}")
                else:
                    logger.debug(f"N{node_id}: beacon residual ignorado (grace period 30s)")
            elif not sim_field and node.sim_mode and node.current_sim_run_id:
                # Firmware encerrou simulação — fechar SimulationRun.
                # Carência de 90 s após o início: até lá o SIM_START (ou o reenvio do
                # watchdog, aos 60 s) pode ainda não ter chegado ao firmware, e um beacon
                # sem 'sim' não indica que a simulação terminou.
                from models import SimulationRun
                run = session.get(SimulationRun, node.current_sim_run_id)
                if run:
                    elapsed = (datetime.utcnow() - run.started_at).total_seconds()
                    if elapsed >= 90:
                        # telemetry_count == 0 → firmware nunca recebeu (nem o retry do watchdog)
                        reason = "firmware_rejected" if run.telemetry_count == 0 else "firmware_stopped"
                        _close_simulation_run(run, reason, session)
                        logger.info(f"N{node_id}: simulação encerrada ({reason}, elapsed={elapsed:.0f}s, beacons={run.telemetry_count})")
                    else:
                        logger.debug(f"N{node_id}: beacon sem 'sim' ignorado — grace period início ({elapsed:.0f}s < 90s)")
            elif not sim_field and node.sim_mode:
                # Legado: nó em sim_mode sem current_sim_run_id (versões antigas)
                node.sim_mode = False
                node.sim_ended_at = datetime.utcnow()

            session.commit()

            # A detecção roda no loop ML em background, não aqui.
            logger.info(f"✅ Dados salvos do Nó {node_id}: RSSI {rssi} dBm, SNR {snr} dB, "
                       f"parent={parent_node_id or 'N/A'}")

    except Exception as e:
        logger.exception(f"Erro crítico ao processar mensagem: {e}")

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback quando conecta ao broker (também a cada reconexão: refaz as assinaturas)."""
    service_state["mqtt_connected"] = rc == 0
    if rc == 0:
        logger.info(f"📡 Conectado ao Broker: {settings.MQTT_BROKER}")
        # Wildcard para receber de todas as redes
        client.subscribe("lab/+/uplink")
        client.subscribe("lab/+/send_result")
        logger.info("📡 Subscrito aos tópicos: lab/+/uplink, lab/+/send_result")
    else:
        logger.error(f"❌ Falha na conexão MQTT: código {rc}")

def on_message(client, userdata, msg):
    """Callback quando recebe mensagem"""
    try:
        if len(msg.payload) > MAX_MQTT_PAYLOAD_BYTES:
            logger.warning(f"Mensagem MQTT descartada: {len(msg.payload)} bytes em {msg.topic!r}")
            return
        payload = msg.payload.decode("utf-8")
        topic = msg.topic
        logger.debug(f"📨 Mensagem recebida do tópico {topic}: {payload}")

        # Extrair network_id do tópico
        network_id = extract_network_id(topic)
        if network_id is None:
            logger.warning(f"Tópico MQTT sem network_id válido: {topic}")
            return

        if topic.endswith("/send_result"):
            process_send_result(payload, network_id)
        elif topic.endswith("/uplink"):
            process_message_sync(payload, network_id)
    except Exception as e:
        logger.error(f"Erro ao processar mensagem MQTT: {e}")

def on_disconnect(client, userdata, flags=None, rc=None, properties=None):
    """Callback quando desconecta - compatível com v2"""
    service_state["mqtt_connected"] = False
    disconnect_rc = rc if rc is not None else flags  # Compatibilidade v1/v2
    if disconnect_rc != 0:
        logger.warning("⚠️ Desconectado do broker. Tentando reconectar...")

# Cliente MQTT global
mqtt_client = None
monitor_task = None
# Estado exposto em /health: conexão com o broker e detector
# (not_started | running | failed).
service_state = {"mqtt_connected": False, "detector": "not_started"}

def _load_ml_settings(session):
    """Carrega MLSettings do banco ou cria defaults.
    Força ml_fast_detection=False (campo obsoleto; a camada foi removida)."""
    from models import MLSettings
    ml_settings = session.get(MLSettings, 1)
    if not ml_settings:
        ml_settings = MLSettings(id=1)
        session.add(ml_settings)
        session.commit()
        session.refresh(ml_settings)
    # Garante que fast detection está sempre desabilitado,
    # mesmo que o banco tenha sido persistido com True em sessão anterior.
    if ml_settings.ml_fast_detection:
        ml_settings.ml_fast_detection = False
        session.add(ml_settings)
        session.commit()
        logger.warning(
            "ml_fast_detection legado=True forçado para False (camada removida)."
        )
    return ml_settings


def start_mqtt_client(loop):
    """Inicia o cliente MQTT em uma thread separada"""
    global mqtt_client, monitor_task, anomaly_detector

    # As migrações rodam antes, no lifespan (main.run_startup_migrations), com falha fechada.
    try:
        with Session(engine) as session:
            ml_settings = _load_ml_settings(session)
            logger.info(
                f"MLSettings carregadas — primário={ml_settings.ml_online_primary}, "
                f"combo={ml_settings.ml_online_combo}"
            )
        # Detector de anomalias com as configurações do banco
        anomaly_detector = AnomalyDetector(engine=engine, ml_settings=ml_settings)
        anomaly_detector.start_ml_loop(interval_seconds=60)
        service_state["detector"] = "running"
        logger.info("Detector de anomalias iniciado (pipeline online, ciclo de 60 s)")
    except Exception as exc:
        # A ingestão segue sem detector; /health mostra "degraded".
        service_state["detector"] = "failed"
        logger.error(f"❌ Detector não iniciou: {exc}", exc_info=True)
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.on_disconnect = on_disconnect

    # Autenticação MQTT, quando MQTT_USER e MQTT_PASS estão definidos
    if settings.MQTT_USER and settings.MQTT_PASS:
        mqtt_client.username_pw_set(settings.MQTT_USER, settings.MQTT_PASS)

    # Iniciar Node Monitor (health check) no event loop principal
    from node_monitor import node_monitor
    
    # Conectar o monitor ao MQTT client para publicação
    def mqtt_publish_wrapper(topic, payload):
        """Wrapper para publicação MQTT thread-safe"""
        if mqtt_client and mqtt_client.is_connected():
            mqtt_client.publish(topic, payload)
    
    node_monitor.set_mqtt_publisher(mqtt_publish_wrapper)

    async def start_monitor():
        """Wrapper assíncrono para iniciar o monitor"""
        # Aguardar um pouco para garantir que o MQTT está conectado
        await asyncio.sleep(5)
        logger.info("🩺 Iniciando Node Monitor...")
        await node_monitor.start(None)
    
    monitor_task = asyncio.run_coroutine_threadsafe(start_monitor(), loop)
    
    # connect_async + loop_forever: broker fora do ar na inicialização ou queda depois
    # não encerram a thread; o paho tenta de novo com espera de 1 s a 60 s.
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=60)
    try:
        mqtt_client.connect_async(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
        mqtt_client.loop_forever(retry_first_connection=True)
    except Exception as e:
        logger.error(f"❌ Laço MQTT encerrado: {e}", exc_info=True)
    finally:
        service_state["mqtt_connected"] = False

def stop_mqtt_client():
    """Para o cliente MQTT e o Node Monitor"""
    global mqtt_client, monitor_task
    
    # Parar o Node Monitor
    try:
        from node_monitor import node_monitor
        node_monitor.stop()
        logger.info("🛑 Node Monitor parado")
    except Exception as e:
        logger.error(f"Erro ao parar Node Monitor: {e}")
    
    # Parar o cliente MQTT
    if mqtt_client:
        mqtt_client.disconnect()
        mqtt_client.loop_stop()


# ==========================================
# Publish to Downlink + Send Result Processing
# ==========================================

def publish_to_downlink(payload: dict, network_id: Optional[int]) -> bool:
    """
    Publica mensagem no tópico MQTT downlink para o Gateway da rede.
    Tópico inclui network_id: lora/mesh/{network_id}/command
    Thread-safe — usa o mqtt_client global.

    Args:
        payload: Dict com o comando para o gateway (será serializado para JSON)
        network_id: ID da rede do nó de destino (obrigatório; sem rede não publica)

    Returns:
        bool: True se publicado com sucesso
    """
    global mqtt_client

    if network_id is None:
        logger.error("❌ Nó sem rede: comando de downlink não enviado.")
        return False
    if mqtt_client is None or not mqtt_client.is_connected():
        logger.error("❌ MQTT não conectado. Não é possível enviar para downlink.")
        return False

    try:
        json_payload = json.dumps(payload)
        topic = f"lora/mesh/{network_id}/command"
        result = mqtt_client.publish(topic, json_payload)

        if result.rc == 0:
            logger.info(f"📤 Downlink publicado em {topic}: {json_payload}")
            return True
        else:
            logger.error(f"❌ Falha ao publicar downlink (rc={result.rc})")
            return False
    except Exception as e:
        logger.error(f"❌ Erro ao publicar downlink: {e}")
        return False


_SEND_RESULT_STATUSES = {"sent", "delivered", "failed"}


def process_send_result(payload_str: str, network_id: Optional[int] = None):
    """
    Processa resultado de envio reportado pelo Gateway via MQTT (lab/{network_id}/send_result).
    Atualiza o MessageLog correspondente.
    
    Formato esperado do Gateway:
    {
        "type": "send_result",
        "msg_id": 42,
        "dest": "0x0002",
        "status": "delivered" | "sent" | "failed",
        "rtt_ms": 245,
        "error": "timeout"
    }
    """
    try:
        data = json.loads(payload_str, parse_constant=_reject_non_finite)

        if not isinstance(data, dict) or data.get("type") != "send_result":
            return

        msg_id = data.get("msg_id")
        status = data.get("status", "sent")
        rtt_ms = data.get("rtt_ms")
        error = data.get("error")
        if (not isinstance(msg_id, int) or isinstance(msg_id, bool)
                or status not in _SEND_RESULT_STATUSES
                or (rtt_ms is not None and (not isinstance(rtt_ms, int) or not 0 <= rtt_ms <= 600_000))
                or (error is not None and not isinstance(error, str))):
            logger.warning("send_result recusado: campos fora do formato esperado")
            return
        error = error[:200] if error else None

        logger.info(f"📬 Send result: msg_id={msg_id}, status={status}, rtt={rtt_ms}ms")
        
        # Atualizar MessageLog no banco
        if msg_id:
            with Session(engine) as session:
                message = session.get(MessageLog, msg_id)
                if message:
                    message.status = status
                    message.updated_at = datetime.utcnow()
                    if rtt_ms is not None:
                        message.rtt_ms = rtt_ms
                    if error:
                        message.error_message = error
                    session.add(message)
                    session.commit()
                    logger.info(f"✅ MessageLog {msg_id} atualizado: status={status}")
                else:
                    logger.warning(f"⚠️ MessageLog {msg_id} não encontrado")
        
    except ValueError:  # inclui json.JSONDecodeError
        logger.error(f"❌ JSON inválido em send_result ({len(payload_str)} bytes)")
    except Exception as e:
        logger.exception(f"❌ Erro ao processar send_result: {e}")


# ─── Ciclo de vida das execuções de simulação ──────────────────────────────

def _get_data_origin(sim_field) -> str:
    """Retorna 'synthetic' se beacon tem campo 'sim', 'real' caso contrário."""
    return "synthetic" if sim_field else "real"


def _should_activate_sim(node, sim_field: str) -> bool:
    """
    Retorna True se uma reativação de simulação é legítima.
    Carência de 30 s: rejeita beacons residuais logo após o encerramento
    da simulação.
    """
    if not node.sim_ended_at:
        return True
    elapsed = (datetime.utcnow() - node.sim_ended_at).total_seconds()
    return elapsed >= 30


def _increment_telemetry_count(sim_run_id: int, session) -> None:
    """Incrementa telemetry_count do SimulationRun ativo em tempo real.

    Na 1ª telemetria sintética, grava anomaly_onset_at uma única vez: o início
    real da anomalia, a partir do qual se mede a latência do detector
    (separada da latência total, que inclui o atraso do firmware).
    """
    from models import SimulationRun
    run = session.get(SimulationRun, sim_run_id)
    if run and run.ended_at is None:
        run.telemetry_count += 1
        if run.anomaly_onset_at is None:
            run.anomaly_onset_at = datetime.utcnow()
        session.add(run)
        session.flush()


def _close_simulation_run(run, ended_reason: str, session) -> None:
    """Encerra um SimulationRun e sincroniza o estado do Node."""
    from models import Node
    run.ended_at = datetime.utcnow()
    run.ended_reason = ended_reason
    session.add(run)
    node = session.get(Node, run.node_id)
    if node:
        node.sim_mode = False
        node.sim_anomaly = None
        node.sim_ended_at = datetime.utcnow()
        node.current_sim_run_id = None
        session.add(node)
    session.commit()


def _simulation_watchdog_tick(eng=None) -> None:
    """Encerra runs expirados e órfãos. Chamado pelo watchdog loop no lifespan."""
    from models import SimulationRun
    from sqlmodel import Session, select
    _engine = eng or engine
    with Session(_engine) as session:
        now = datetime.utcnow()

        # 1. Timeout: runs cujo tempo planejado já passou (Python-side — cross-database)
        open_runs = session.exec(
            select(SimulationRun).where(SimulationRun.ended_at == None)
        ).all()
        expired = [
            r for r in open_runs
            if (now - r.started_at).total_seconds() >= r.duration_planned
        ]
        for run in expired:
            _close_simulation_run(run, "timeout", session)
            logger.info(f"SimulationRun #{run.id} encerrado por timeout")

        # 2. Orphan retry: ainda sem beacons após 60s → reenviar SIM_START uma vez
        RETRY_AFTER = timedelta(seconds=60)
        GIVE_UP_AFTER = timedelta(seconds=120)
        candidates = session.exec(
            select(SimulationRun)
            .where(SimulationRun.ended_at == None)
            .where(SimulationRun.telemetry_count == 0)
            .where(SimulationRun.started_at < now - RETRY_AFTER)
        ).all()
        for run in candidates:
            elapsed = (now - run.started_at).total_seconds()
            if elapsed >= GIVE_UP_AFTER.total_seconds():
                # Retry já aconteceu (ou foi pulado) — fechar definitivamente
                _close_simulation_run(run, "firmware_rejected", session)
                logger.warning(f"SimulationRun #{run.id} encerrado como firmware_rejected após {elapsed:.0f}s")
            else:
                # Primeira detecção (60-120s): reenviar SIM_START
                from models import Node
                node = session.get(Node, run.node_id)
                if node and node.lora_address:
                    payload = {
                        "cmd": "send",
                        "dest": node.lora_address,
                        "msg": f"SIM_START:{run.anomaly}:{run.duration_planned}",
                        "msg_id": 9039,
                    }
                    ok = publish_to_downlink(payload, node.network_id)
                    logger.warning(
                        f"SimulationRun #{run.id} sem beacons em {elapsed:.0f}s — "
                        f"retry SIM_START {'enviado' if ok else 'FALHOU'}"
                    )
